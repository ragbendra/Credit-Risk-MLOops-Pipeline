"""
FastAPI application for Credit Risk Prediction.

Endpoints:
- POST /predict - Get credit risk prediction
- GET /health - Health check
- GET /docs - Swagger documentation (auto-generated)

Usage:
    uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

from .schemas import CreditRequest, CreditResponse, HealthResponse


# Configuration
MODEL_PATH = Path("models/best_model.pkl")
MODEL_VERSION = "1.0.0"
API_VERSION = "1.0.0"

# Global model instance
_model = None
_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    global _model, _start_time
    
    _start_time = datetime.now()
    
    # Load model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("   API will start but /predict will return 503")
    
    yield
    
    # Cleanup on shutdown
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="MLOps pipeline for credit default prediction using German Credit Dataset",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (allow all for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    
    Returns service status, whether model is loaded, and API version.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version=API_VERSION,
    )


@app.post("/predict", response_model=CreditResponse, tags=["Prediction"])
def predict(request: CreditRequest):
    """
    Predict credit risk for a loan application.
    
    Input is validated by Pydantic - invalid values will return 422.
    
    - **prediction**: 'good' or 'bad' credit risk
    - **probability**: Probability of default (0.0-1.0)
    - **risk_score**: Risk score (0-100, higher = more risky)
    """
    # Check if model is loaded
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model file exists at models/best_model.pkl"
        )
    
    # Convert request to feature array
    # Order must match training feature order!
    features = np.array([[
        request.status,
        request.duration,
        request.credit_history,
        request.purpose,
        request.amount,
        request.savings,
        request.employment,
        request.installment_rate,
        request.personal_status,
        request.other_debtors,
        request.residence,
        request.property,
        request.age,
        request.other_plans,
        request.housing,
        request.existing_credits,
        request.job,
        request.dependents,
        request.telephone,
        request.foreign_worker,
    ]])
    
    # Make prediction
    try:
        prob = _model.predict_proba(features)[0][1]  # Probability of class 1 (bad)
        prediction = "bad" if prob > 0.5 else "good"
        
        return CreditResponse(
            prediction=prediction,
            probability=round(float(prob), 4),
            risk_score=int(prob * 100),
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/", tags=["Root"])
def root():
    """Root endpoint - redirects to docs."""
    return {
        "message": "Credit Risk Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# For running directly with python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
