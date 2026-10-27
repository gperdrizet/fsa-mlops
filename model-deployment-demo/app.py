<<<<<<< HEAD
'''Model deployment for Breast Cancer classification using FastAPI'''

=======
>>>>>>> d91ba9f6edf6b0e9d6d7403b9636ca9129242bfb
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the saved model
current_dir = Path(__file__).resolve().parent
<<<<<<< HEAD
model = joblib.load(current_dir / 'breast_cancer_model.pkl')
=======
model = joblib.load(f'{current_dir}/breast_cancer_model.pkl')
>>>>>>> d91ba9f6edf6b0e9d6d7403b9636ca9129242bfb

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
<<<<<<< HEAD

=======
    
>>>>>>> d91ba9f6edf6b0e9d6d7403b9636ca9129242bfb
    # Make the prediction
    prediction = model.predict(features)
    print(prediction)
    # Return the prediction result
    return {"prediction": int(prediction[0])}
