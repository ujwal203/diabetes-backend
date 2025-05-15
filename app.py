from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# 1. CORS Configuration (Production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://calm-froyo-f2f2db.netlify.app",
        "http://localhost:3000"  # For local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Manual CORS Headers (Double Protection)
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update({
        "Access-Control-Allow-Origin": "https://calm-froyo-f2f2db.netlify.app",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    })
    return response

# 3. Explicit OPTIONS Handler
@app.options("/predict", include_in_schema=False)
async def options_predict():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "https://calm-froyo-f2f2db.netlify.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

# Load model
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

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
async def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.dict()])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        return {
            "prediction": "Diabetic" if prediction[0] == 1 else "Not Diabetic",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def health_check():
    return {
        "status": "API is healthy",
        "cors_configured": True,
        "allowed_origin": "https://calm-froyo-f2f2db.netlify.app"
    }