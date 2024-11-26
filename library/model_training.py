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
    confusion_matrix, 
    classification_report
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
                    "max_depth": (3, 20),
                    "min_samples_split": (2, 10),
                    "min_samples_leaf": (1, 4),
                    "class_weight": [None, 'balanced']
                },
                "type": "classification"
            },
            "decisiontree": {
                "model": DecisionTreeClassifier,
                "params": {
                    "max_depth": (3, 20),
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
                    "max_depth": (3, 20),
                    "learning_rate": (0.01, 0.3),
                    "subsample": (0.5, 1.0),
                    "colsample_bytree": (0.5, 1.0),
                    "scale_pos_weight": (0.5, 2.0)
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
            tuple: Train, validation, and test data splits
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=config.RANDOM_STATE,
            stratify=y
        )

        # Second split: separate train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=0.2, 
            random_state=config.RANDOM_STATE,
            stratify=y_temp
        )

        # Log dataset sizes
        self.logger.info(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
        self.logger.info(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.2f}%)")
        self.logger.info(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")

        # Log class distribution
        self.logger.info("Class Distribution:")
        self.logger.info(f"Training set:\n{y_train.value_counts(normalize=True)}")
        self.logger.info(f"Validation set:\n{y_val.value_counts(normalize=True)}")
        self.logger.info(f"Test set:\n{y_test.value_counts(normalize=True)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    @log_decorator
    def _validate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Comprehensive model validation with multiple metrics.
        
        Args:
            model: Trained model
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Target sets
        
        Returns:
            dict: Validation metrics
        """
        # Predict on validation and test sets
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Probability predictions for AUC
        try:
            y_val_prob = model.predict_proba(X_val)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_val_prob = y_val_pred
            y_test_prob = y_test_pred

        # Compute metrics for validation set
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, average='weighted'),
            'recall': recall_score(y_val, y_val_pred, average='weighted'),
            'f1_score': f1_score(y_val, y_val_pred, average='weighted'),
        }

        # Add AUC if binary classification
        try:
            val_metrics['roc_auc'] = roc_auc_score(y_val, y_val_prob)
        except:
            pass

        # Compute metrics for test set
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, average='weighted'),
            'recall': recall_score(y_test, y_test_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
        }

        # Add AUC if binary classification
        try:
            test_metrics['roc_auc'] = roc_auc_score(y_test, y_test_prob)
        except:
            pass

        # Detailed classification report
        val_report = classification_report(y_val, y_val_pred)
        test_report = classification_report(y_test, y_test_pred)

        # Confusion matrices
        val_cm = confusion_matrix(y_val, y_val_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        cv_scores = cross_validate(
            model, 
            np.concatenate([X_train, X_val]), 
            np.concatenate([y_train, y_val]), 
            cv=cv, 
            scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        )

        return {
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'validation_report': val_report,
            'test_report': test_report,
            'validation_confusion_matrix': val_cm,
            'test_confusion_matrix': test_cm,
            'cross_validation_scores': {
                metric: scores.mean() 
                for metric, scores in cv_scores.items() 
                if metric.startswith('test_')
            }
        }

    @log_decorator
    def _objective(self, trial, model_class, X_train, X_val, y_train, y_val):
        """
        Objective function for Optuna hyperparameter optimization.
        
        Args:
            trial (optuna.Trial): Optuna trial object
            model_class (type): Model class to optimize
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            y_train (pd.Series): Training target
            y_val (pd.Series): Validation target
        
        Returns:
            float: Validation score for optimization
        """
        # Apply SMOTE to training data
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

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
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Calculate multiple metrics
        return f1_score(y_val, y_pred, average='weighted')

    @log_decorator
    def optimize_hyperparameters(self, X, y, n_trials=100):
        """
        Perform hyperparameter optimization for all defined models.
        """
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

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
                X_val=X_val, 
                y_train=y_train, 
                y_val=y_val
            )
            
            # Run the optimization
            study.optimize(objective, n_trials=n_trials)

            # Best trial information
            best_trial = study.best_trial
            self.logger.info(f"Best {model_config_name} trial:")
            self.logger.info(f"  Value: {best_trial.value}")
            self.logger.info("  Params: ")
            for key, value in best_trial.params.items():
                self.logger.info(f"    {key}: {value}")

            # Train model with best parameters
            best_model = model_class(**best_trial.params)
            
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

        return optimized_models

    @log_decorator
    def train_models(self):
        """
        Train multiple models with Optuna hyperparameter tuning.
        
        Returns:
            dict: Trained models with their configurations
        """
        # Prepare data
        X = self.data.drop('status', axis=1)
        y = self.data['status']

        # Perform hyperparameter optimization
        self.trained_models = self.optimize_hyperparameters(X, y)

        return self.trained_models