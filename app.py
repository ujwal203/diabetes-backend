from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from fastapi.middleware.cors import CORSMiddleware
# Load saved model and scaler once when app starts
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define expected input schema using Pydantic
class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to DataFrame to avoid feature name warning
    input_df = pd.DataFrame([data.dict()])
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Convert prediction to human-readable label
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    return {"prediction": result}
