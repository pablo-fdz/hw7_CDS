import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from abc import ABCMeta, abstractmethod

class Transformer(metaclass = ABCMeta):
    def __init__(self, dataframe: pd.DataFrame, columns_to_transform: list):
        self.dataframe = dataframe
        self.columns_to_transform = columns_to_transform
    
    @abstractmethod
    def transform(self):
        return NotImplementedError

class OHE(Transformer):
    def __init__(self, dataframe: pd.DataFrame, columns_to_transform: list):
        super().__init__(dataframe, columns_to_transform)  # Initialize the parent class
        self.encoded_columns = None

    def transform(self):

        # Creates dummies and drop the first dummy
        dummies = pd.get_dummies(self.dataframe.loc[:, self.columns_to_transform], drop_first = True)
        # Store the names of the dummy columns created
        self.encoded_columns = dummies.columns.tolist()
         # We drop the input columns from the original data frame
        df_dropped = self.dataframe.drop(labels = self.columns_to_transform, axis = 1)
        # We join the dummies
        transformed_dataframe = df_dropped.join(dummies)

        return transformed_dataframe

    def get_encoded_columns(self):
        return self.encoded_columns

class Standardize(Transformer):
    def __init__(self, dataframe: pd.DataFrame, columns_to_transform: list):
        super().__init__(dataframe, columns_to_transform)  # Initialize the parent class

    def transform(self):
        
        transformed_dataframe = self.dataframe.copy()

        for col in self.columns_to_transform:
            try:
                transformed_dataframe[col] = (transformed_dataframe[col] - transformed_dataframe[col].mean()) / transformed_dataframe[col].std()
            except TypeError:
                print(f'Column {col} is not numerical.')

        return transformed_dataframe
