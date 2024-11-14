import pandas as pd
from sklearn.model_selection import train_test_split


# We create a parent class which reads the data into a data frame
class DataReader:
    def __init__(self, csv_path: str):
        """
        Initialize the DataReader with a CSV file path.
        """
        self.csv_path = csv_path
        self.data = None

    def read_data(self):
        """
        Reads a CSV file and stores it in the data attribute.
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Data successfully read from {self.csv_path}")
        except FileNotFoundError:
            print(f"File not found: {self.csv_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return self.data


# We create a child class which splits the data into training and test dataframes
class DataSplitter(DataReader):
    def __init__(self, csv_path: str, test_size: float = 0.2, random_state: int = None):
        """
        Initialize DataSplitter with CSV path, test size, and random state for reproducibility.
        """
        super().__init__(csv_path)  # Initialize the parent class
        self.test_size = test_size
        self.random_state = random_state

    # DataSplitter is a DataReader that, additionally, creates the following method
    def split_data(self):
        """
        Splits the data into training and validation sets.
        Requires that `read_data` has already been called to load the data.
        Drops the first column after the split.
        """
        if self.data is None:
            print("No data loaded. Call 'read_data' first.")
            return None, None

        try:
            train_data, validation_data = train_test_split(
                self.data, test_size = self.test_size, random_state = self.random_state
            )
            print(f"Data successfully split with {self.test_size*100}% validation set.")

            # Drop the first column from both train and validation sets
            train_data = train_data.iloc[:, 1:]
            validation_data = validation_data.iloc[:, 1:]
            print("First column dropped from both training and validation sets.")

        except Exception as e:
            print(f"An error occurred while splitting the data: {e}")

        return train_data, validation_data