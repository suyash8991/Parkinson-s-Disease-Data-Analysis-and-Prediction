import pandas as pd
import optuna
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    classification_report,
    make_scorer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from functools import partial
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold, 
    cross_validate
)
import traceback as tb
from sklearn.preprocessing import StandardScaler
from library.utils import save_model
from library.model_evaluation import ModelEvaluation
from config import config
from config.logger_config import Logger
from config.logger_config import log_decorator
from imblearn.pipeline import Pipeline


class ModelTraining:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with preprocessed dataset.
        
        Args:
            data (pd.DataFrame): Preprocessed dataset for model training
        """
        try:
            # Get logger instance
            self.logger = Logger().get_logger()

            # Configure Optuna logging
            optuna.logging.set_verbosity(optuna.logging.INFO)
            
            # Get Optuna logger and configure it
            optuna_logger = logging.getLogger("optuna")
            
            # Ensure Optuna logs are captured by the main logger
            if not optuna_logger.handlers:
                for handler in self.logger.handlers:
                    optuna_logger.addHandler(handler)


            self.data = data
            self.model_definitions = self._get_model_definitions()
            self.trained_models = {}
            self.logger.info("Able to initialize Model Training")
        except:
            self.logger.error("Data was not able to initialize Model Training ",tb.format_exc())

    def _get_model_definitions(self):
        """
        Define model configurations and hyperparameter search spaces.
        
        Returns:
            dict: Model definitions with their respective hyperparameter distributions
        """
        model_definitions = {
            "randomforest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": (50, 300),
                    "max_depth": (3, 15),
                    "min_samples_split": (2, 10),
                    "min_samples_leaf": (1, 4),
                    "class_weight": [None, 'balanced']
                },
                "type": "classification"
            },
            "decisiontree": {
                "model": DecisionTreeClassifier,
                "params": {
                    "max_depth": (3,15),
                    "min_samples_split": (2, 10),
                    "min_samples_leaf": (1, 4),
                    "class_weight": [None, 'balanced']
                },
                "type": "classification"
            },
            "xgboost": {
                "model": XGBClassifier,
                "params": {
                    "n_estimators": (50, 300),
                    "max_depth": (5,20),
                    "learning_rate": (0.01, 0.2),
                    "subsample": (0.5, 1.0),
                },
                "type": "classification"
            }
        }
        return model_definitions

    @log_decorator
    def split_data(self, X, y):
        """
        Splits data into train, validation, and test sets with stratification.

        Args:
            X (pd.DataFrame): Feature columns
            y (pd.Series): Target column

        Returns:
            tuple: Train and test data splits
        """
        try:
            # First split: separate test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=config.RANDOM_STATE,
                stratify=y
            )


            # Log dataset sizes
            self.logger.info(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
            self.logger.info(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")

            # Log class distribution
            self.logger.info("Class Distribution:")
            self.logger.info(f"Training set:\n{y_train.value_counts(normalize=True)}")
            self.logger.info(f"Test set:\n{y_test.value_counts(normalize=True)}")
            return X_train, X_test, y_train, y_test
        except:
            self.logger.error("Data was not able to initialize Model Training ",tb.format_exc())
        return None,None, None,None


    def _objective(self, trial, model_class, X_train, y_train):
        """
        Objective function for Optuna hyperparameter optimization.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            model_class (type): Model class to optimize
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target        
        Returns:
            float: Validation score for optimization
        """
        try:
            # Get model-specific hyperparameters
            model_name = model_class.__name__.lower()
            model_config = next(
                (name for name, config in self.model_definitions.items() 
                if config['model'].__name__.lower() == model_name), 
                None
            )

            if not model_config:
                raise ValueError(f"No configuration found for {model_name}")

            # Suggest hyperparameters
            params = {}
            for param_name, param_range in self.model_definitions[model_config]['params'].items():
                if isinstance(param_range, tuple):
                    # Numeric parameters
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical parameters
                    params[param_name] = trial.suggest_categorical(param_name, param_range)

            # Initialize and train model
            model = model_class(**params)
            # model.fit(X_train_resampled, y_train_resampled)
            pipeline = Pipeline([
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),  # Apply SMOTE within each training fold
        ("model", model)
    ])
        # Stratified K-Fold for imbalanced data
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

            # Cross-validation scoring
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf,scoring=make_scorer(f1_score, average="weighted"),n_jobs=-1)

            return cv_scores.mean() 
            
        except:
            self.logger.error("_Objective function failed to run ",tb.format_exc())
        

    def optimize_hyperparameters(self, X, y, n_trials=200):
        """
        Perform hyperparameter optimization for all defined models.
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
           
            optimized_models = {}

            # Iterate through model definitions
            for model_config_name, model_config in self.model_definitions.items():
                model_class = model_config['model']
                
                # Create a study object and optimize the objective function
                study = optuna.create_study(direction='maximize')
                
                # Partial function to pass to Optuna
                objective = partial(
                    self._objective, 
                    model_class=model_class, 
                    X_train=X_train, 
                    y_train=y_train, 
                )
                
                # Run the optimization
                study.optimize(objective, n_trials=n_trials)

                # Best trial information
                best_trial = study.best_trial
                self.logger.info(f"Best {model_config_name} trial:")
                self.logger.info(f" Value: {best_trial.value}")
                self.logger.info(" Params: ")
                for key, value in best_trial.params.items():
                    self.logger.info(f" {key}: {value}")

                # Train model with best parameters
                best_model = model_class(**best_trial.params)
                print("Model in train best ",best_model)
                # Apply SMOTE to full training data
                smote = SMOTE(random_state=config.RANDOM_STATE)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                best_model.fit(X_train_resampled, y_train_resampled)

                # Package model data for comprehensive evaluation
                model_data = {
                    'model': best_model,
                    'X_test': X_test,
                    'y_test': y_test,
                    'model_name': model_config_name,
                    'best_params': best_trial.params
                }

                # Evaluate model using enhanced ModelEvaluation
                evaluator = ModelEvaluation(model_data)
                evaluation_results = evaluator.evaluate()

                # Save the model
                save_model(best_model, f"{model_config_name}_optimized_model.pkl")

                # Store optimized model details
                optimized_models[model_config_name] = {
                    **model_data,
                    'best_score': best_trial.value,
                    'evaluation_results': evaluation_results
                }
        except:
            self.logger.error("Issue in optimize parameters function",tb.format_exc())

        return optimized_models

    @log_decorator
    def train_models(self):
        """
        Train multiple models with Optuna hyperparameter tuning.
        
        Returns:
            dict: Trained models with their configurations
        """
        # Prepare data
        try:
            X = self.data.drop('status', axis=1)
            y = self.data['status']

            # Perform hyperparameter optimization
            self.trained_models = self.optimize_hyperparameters(X, y)
        except:
            self.logger.error("Data was not able to execute train models ",tb.format_exc())

        return self.trained_models