import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from config.logger_config import Logger
from config.logger_config import log_decorator
from ydata_profiling import ProfileReport
import traceback as tb

class ParkinsonEDA:

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA class with the dataset.

        :param data: DataFrame containing the Parkinson's dataset.
        """
        self.logger = Logger().get_logger()
        try:
            self.data = data
        except e:
            self.logger.error("Error in accessing eda for data")
    
    @log_decorator
    def ydata_profiling(self):
        profile = ProfileReport(self.data)
        profile.to_file("./data/profile_report.html")
    @log_decorator
    def summarize_data(self):
        """
        Print a summary of the dataset.
        """
        print("************ Dataset Overview ************")
        print(self.data.head(), "\n")
        print("Dataset Info:")
        print(self.data.info(), "\n")
        print("Summary Statistics:")
        print(self.data.describe(), "\n")

    @log_decorator
    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        try:
            missing = self.data.isnull().sum()
            print("Missing Values:")
            print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.\n")
        except:
            self.logger.error("Missing value error ",tb.format_exc())


    @log_decorator
    def class_distribution(self):
        """
        Display the distribution of the target class.
        """
        sns.countplot(x='status', data=self.data)
        plt.title("Class Distribution")
        plt.xlabel("status (0 = No Parkinson's, 1 = Parkinson's)")
        plt.ylabel("Count")
        plt.show()

    @log_decorator
    def correlation_matrix(self):
        """
        Display a heatmap of the correlation matrix.
        Only numeric columns are considered.
        """
        try:
            numeric_data = self.data.select_dtypes(include=['number'])  # Exclude non-numeric columns
            corr_matrix = numeric_data.corr()  # Compute correlation matrix
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Correlation Matrix")
            plt.show()
        except:
            self.logger.error("Correlation matrix couldnt be generated ",tb.format_exc())


