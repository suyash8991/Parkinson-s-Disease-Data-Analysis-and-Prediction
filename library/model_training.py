import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from functools import partial
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.model_selection import train_test_split
from library.utils import save_model
from config import config





class ModelTraining:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with preprocessed dataset.
        """
        self.data = data
        self.model = None


    def _get_model_definitions(self):
    # Define models
        model_definitions = {
            "randomforest": {
                "model": RandomForestClassifier,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(50, 300, step=50),
                    "max_depth": optuna.distributions.IntDistribution(3, 20),
                    "min_samples_split": optuna.distributions.IntDistribution(2, 10),
                },
            },
            "decisionTree": {
                "model": DecisionTreeClassifier,
                "params": {
                    "max_depth": optuna.distributions.IntDistribution(3, 20),
                    "min_samples_split": optuna.distributions.IntDistribution(2, 10),
                },
            },
            "xgboost": {
                "model": XGBClassifier,
                "params": {
                    "n_estimators": optuna.distributions.IntDistribution(50, 300, step=50),
                    "max_depth": optuna.distributions.IntDistribution(3, 20),
                    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
                },
            },
        }            



    def split_data(self,X,y):
        """
        Splits data into train test and split

        Args:
            X (dataframe): feature columns
            y (dataframe): targe column

        Returns:
            tuple: Train, validation and test data
        """

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=config.RANDOM_STATE
        )

        X_train,X_val,y_train,y_val = train_test_split(
            X_temp, y_temp, 
            test_size=0.2, 
            random_state=config.RANDOM_STATE,
            stratify=y_temp
        )
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")


        return X_train,X_val,X_test,y_train,y_val,y_test

    def train_models(self):
        """
        Train multiple models and return them along with train-test split.
        """
        print("IN TRAIN")
        self.data['status'].value_counts()
        X = self.data.drop('status', axis=1)  # Drop target column
        y = self.data['status']  # Target column

        # Split data into train, calidation and test
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X,y)

        numerical_cols = list(self.data.select_dtypes(include=['float64']).columns)
        print( "PRINT ",numerical_cols)
        trained_models = {}  # Store results for each model

        # Apply SMOTE for upsampling the minority class in the training data
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        for model_name, model in models.items():
            # Train the model
            print(f"Training {model_name} model...")
            model.fit(X_train_resampled, y_train_resampled)
            
            # Save the trained model
            save_model(model, f"{model_name}_parkinson_model.pkl")

            trained_models[model_name] = {
                "model": model,
                "X_train": X_train,
                "X_validate": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_validate":y_val,
                "y_test": y_test
            }

        return trained_models
