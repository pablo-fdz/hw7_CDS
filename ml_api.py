from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
# Load the model and feature count from the pickle file
model_data = joblib.load("model.pkl")
model = model_data['model']
num_features = model_data['num_features'] # Number of features used in training

# Define the expected input structure
class ModelInput(BaseModel):
    values: list

# Create prediction endpoint
@app.post("/predict")
async def predict(input_data: ModelInput):
    # Check if the input features match the expected count
    if len(input_data.values) != num_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {num_features} features, but received {len(input_data.features)}."
        )
    else:
        pass
    # Now, we pass the input features to make a prediction
    try:
        # Prepare features for prediction
        input_features = np.array(input_data.values).reshape(1, -1)
        prediction = model.predict(input_features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))