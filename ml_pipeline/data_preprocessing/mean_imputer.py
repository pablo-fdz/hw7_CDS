import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class MeanImputer:
    def __init__(self, dataframe: pd.DataFrame, columns_to_impute: list):
        """
        Initialize the NaNRemover with a dataframe.
        """
        self.input_dataframe = dataframe
        self.columns_to_impute = columns_to_impute

    def fill_na_mean(self):

        """
        This method takes a Pandas DataFrame and fills with the mean the rows with 
        NAs of the specified subset of columns.

        Returns:
        - df_imputed: DataFrame without NAs in the specified columns.
        """

        # Check if the input types are the adequate ones
        if not isinstance(self.input_dataframe, pd.DataFrame):
            raise TypeError(f"Expected target to be a dataframe, got {type(self.input_dataframe)} instead.")
        if not isinstance(self.columns_to_impute, list) or not all(isinstance(f, str) for f in self.columns_to_impute):
            raise TypeError(f"Expected features to be a list of strings, got {type(self.columns_to_impute)} instead.")
        
        # Fill missing values with mean for the specified subset of columns
        imputed_dataframe = self.input_dataframe.copy()
        for col in self.columns_to_impute:
            if col in imputed_dataframe.columns:
                mean_value = imputed_dataframe[col].mean()
                imputed_dataframe[col].fillna(mean_value, inplace=True)

        return imputed_dataframe