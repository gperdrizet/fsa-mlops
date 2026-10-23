from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the saved model
current_dir = Path(__file__).resolve().parent
model = joblib.load(f'{current_dir}/breast_cancer_model.pkl')

app = FastAPI()

# Define the input data structure using Pydantic
class InputData(BaseModel):
    features: list

# Define the prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Debugging: print the incoming features
    print("Received features:", data.features)

    # Check if features are provided
    if len(data.features) == 0:
        return {"error": "No features provided"}

    # Convert the input features into a numpy array and reshape it for the model
    features = np.array(data.features).reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(features)
    print(prediction)
    # Return the prediction result
    return {"prediction": int(prediction[0])}
