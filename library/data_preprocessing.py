import pandas as pd
import numpy as np
import traceback as tb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from config import config
from config.logger_config import Logger
from config.logger_config import log_decorator

class DataPreprocessing:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with dataset.
        """
        self.logger = Logger().get_logger()
        try:
            self.data = data
            self.logger.info("DataPreprocessing initialized successfully")
        except:
            self.logger.info(tb.format_exc())
        
    @log_decorator
    def drop_columns(self, columns_to_drop):
        """
        Drop irrelevant columns.
        """
        self.data = self.data.drop(columns=columns_to_drop, axis=1)
        print(f"Dropped columns: {columns_to_drop}")

    @log_decorator
    def standardize_features(self):
        """
        Standardize numerical features.
        """
        numeric_cols = self.data.select_dtypes(include=['float64']).columns
        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        print("Standardized numerical features.")


    @log_decorator
    def feature_importance(self):
        """
        Evaluate feature importance using tree-based methods.
        """
        X = self.data.drop('status', axis=1)
        y = self.data['status']

        model = ExtraTreesClassifier()
        model.fit(X, y)
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("Feature importance:\n", feature_importance)

    @log_decorator
    def handle_outliers(self):
        """
        Handle outliers by capping values at 1.5 * IQR range.
        """
        for column in self.data.select_dtypes(include=['float64']):
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.data[column] = np.clip(self.data[column], lower_bound, upper_bound)
        print("Handled outliers for numerical columns.")




    @log_decorator
    def get_data(self):
        """
        Return the processed dataset.
        """
        return self.data


