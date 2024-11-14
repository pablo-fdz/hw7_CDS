import pandas as pd
from ml_pipeline.data_reader import DataReader, DataSplitter
from ml_pipeline.data_preprocessing import NaNRemover, MeanImputer, OHE, Standardize
from ml_pipeline.model_training import ModelTrainer

# Step 1: Load data

## Create an instance of DataSplitter
data_splitter = DataSplitter("sample_diabetes_mellitus_data.csv", test_size=0.3, random_state=42)

## Read the data from CSV. The child class has access to the attributes and methods of the parent class
df = data_splitter.read_data()

## Split the data into training and validation sets
train_df, val_df = data_splitter.split_data()

# Step 2: Preprocess data

## We create a preprocessor class that removes those rows that contain NaN values 
## in the columns: age, gender, ethnicity.

### Remove null values from the training dataset
clean_train_df = NaNRemover(train_df, 
                            columns_to_clean = ['age', 'gender', 'ethnicity']).remove_na()

## Create a preprocessor class that fills NaN with the mean value of the column 
## in the columns: height, weight and bmi

imputed_train_df = MeanImputer(clean_train_df, ['height', 'weight', 'bmi']).fill_na_mean()

## Create at least two feature classes that transform some of the columns in the data set.
## Below, we OHE the ethnicity column and standardize the bmi column

### Create an instance of OHE
ohe = OHE(imputed_train_df, ['ethnicity', 'gender'])
### We encode the ethnicity column
ohe_df = ohe.transform()
### Access encoded columns
ohe_columns = ohe.get_encoded_columns()

### Create an instance of Standardize
standardize = Standardize(ohe_df, ['bmi'])

### We encode the ethnicity column
standardized_df = standardize.transform()

# Step 3: Train model
feature_columns = ['age', 'bmi', 'height', 'weight'] + ohe_columns
target_column = 'diabetes_mellitus'
trainer = ModelTrainer(standardized_df, feature_columns, target_column)
trainer.train_model()

# Step 4: Save model
trainer.save_model("model.pkl")