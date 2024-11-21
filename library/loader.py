import pandas as pd


class DataLoader:
    def __init__(self):
        self.data = None

    def load_csv(self,file_path: str) -> pd.DataFrame:
        """ -->update this 
        Load a CSV file into a pandas DataFrame.

        :param file_path: Path to the CSV file.
        :return: Loaded DataFrame.
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}.")
            self.data['status']= self.data['status'].apply(lambda x : 1 if x>0.5 else 0)
            print("Data now ",self.data['status'].head())
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
