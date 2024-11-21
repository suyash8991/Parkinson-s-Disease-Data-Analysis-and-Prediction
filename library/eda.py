import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample


class ParkinsonEDA:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA class with the dataset.

        :param data: DataFrame containing the Parkinson's dataset.
        """
        self.data = data

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

    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        missing = self.data.isnull().sum()
        print("Missing Values:")
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.\n")

    def class_distribution(self):
        """
        Display the distribution of the target class.
        """
        sns.countplot(x='status', data=self.data)
        plt.title("Class Distribution")
        plt.xlabel("status (0 = No Parkinson's, 1 = Parkinson's)")
        plt.ylabel("Count")
        plt.show()

    def correlation_matrix(self):
        """
        Display a heatmap of the correlation matrix.
        Only numeric columns are considered.
        """
        numeric_data = self.data.select_dtypes(include=['number'])  # Exclude non-numeric columns
        corr_matrix = numeric_data.corr()  # Compute correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()


