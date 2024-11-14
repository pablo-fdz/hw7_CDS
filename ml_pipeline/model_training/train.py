import pandas as pd
import warnings
import joblib
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self, training_dataframe: pd.DataFrame, feature_columns: list, target_column: str):
        # We set the input attributes as private with "_"
        self._training_dataframe = training_dataframe
        self._feature_columns = feature_columns
        self._target_column = target_column
        # We set as public attribute the model which we're training
        self.model = LogisticRegression()
        self.fitted_model = None

    def train_model(self):
        self.fitted_model = self.model.fit(
            X = self._training_dataframe[self._feature_columns], 
            y = self._training_dataframe[self._target_column]
            )

    def save_model(self, filepath):
        joblib.dump({'model': self.fitted_model, 'num_features': len(self._feature_columns)}, 'model.pkl')
        print(f"Model saved at {filepath}")