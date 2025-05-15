from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# ================= CORS CONFIGURATION =================
# Production domains
PRODUCTION_DOMAINS = [
    "https://calm-froyo-f2f2db.netlify.app",
    # Add other production domains if needed
]

# Development domains
DEVELOPMENT_DOMAINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# Combine based on environment
allowed_origins = PRODUCTION_DOMAINS + (DEVELOPMENT_DOMAINS if os.getenv("ENV") == "development" else [])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manual CORS for preflight requests
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = ", ".join(allowed_origins)
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Explicit OPTIONS handler
@app.options("/predict")
async def options_predict():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": ", ".join(allowed_origins),
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

# ================= MODEL LOADING =================
try:
    model = joblib.load('diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ================= API ENDPOINTS =================
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
async def predict(data: PatientData, request: Request):
    try:
        # Validate input
        input_data = data.dict()
        if any(value < 0 for value in input_data.values()):
            raise HTTPException(status_code=422, detail="Negative values not allowed")
        
        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        return {
            "prediction": "Diabetic" if prediction[0] == 1 else "Not Diabetic",
            "status": "success",
            "input_data": input_data  # For debugging
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def health_check():
    return {
        "status": "API is healthy",
        "cors_enabled": True,
        "allowed_origins": allowed_origins,
        "model_loaded": model is not None
    }

# ================= STARTUP VALIDATION =================
@app.on_event("startup")
async def startup_event():
    print(f"Allowed origins: {allowed_origins}")
    print("Model loaded successfully" if model else "Model failed to load")