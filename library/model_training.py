import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.model_selection import train_test_split
from library.utils import save_model
from config import config
# Define models
models = {
    "randomforest": RandomForestClassifier(),
    "decisionTree": DecisionTreeClassifier(),
    "xgboost": XGBClassifier()
}

class ModelTraining:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with preprocessed dataset.
        """
        self.data = data
        self.model = None

    def train_models(self):
        """
        Train multiple models and return them along with train-test split.
        """
        print("IN TRAIN")
        self.data['status'].value_counts()
        X = self.data.drop('status', axis=1)  # Drop target column
        y = self.data['status']  # Target column

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
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
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }

        return trained_models
