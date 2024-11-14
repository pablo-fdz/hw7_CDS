import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class NaNRemover:
    def __init__(self, dataframe: pd.DataFrame, columns_to_clean: list):
        """
        Initialize the NaNRemover with a dataframe.
        """
        self.input_dataframe = dataframe
        self.columns_to_clean = columns_to_clean
        self.imputer = None

    def remove_na(self):

        """
        Takes a Pandas DataFrame and drops the rows with NAs of the specified subset of columns.

        Returns:
        - df_clean: DataFrame without NAs in the specified columns.
        """

        # Check if the input types are the adequate ones
        if not isinstance(self.input_dataframe, pd.DataFrame):
            raise TypeError(f"Expected target to be a dataframe, got {type(self.input_dataframe)} instead.")
        if not isinstance(self.columns_to_clean, list) or not all(isinstance(f, str) for f in self.columns_to_clean):
            raise TypeError(f"Expected features to be a list of strings, got {type(self.columns_to_clean)} instead.")
        
        # Deletion of the rows with NAs
        clean_dataframe = self.input_dataframe.dropna(subset = self.columns_to_clean)

        return clean_dataframe
    
    # This is done so that we can directly call a method after creating an instance
    def __call__(self):
        return self.remove_na()