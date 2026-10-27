'''API for Breast Cancer classification using the trained Random Forest model'''

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
current_dir = Path(__file__).resolve().parent
model = joblib.load(current_dir / "model.pkl")

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict/")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
