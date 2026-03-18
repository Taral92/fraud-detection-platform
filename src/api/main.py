from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn

# Load model and threshold
model = joblib.load('models/fraud_model.pkl')
threshold = joblib.load('models/threshold.pkl')

app = FastAPI(
    title="Fraud Detection API",
    description="Real time fraud detection with explainability",
    version="1.0.0"
)

# Request schema
class Transaction(BaseModel):
    features: list[float]

# Response schema
class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    confidence: float
    risk_level: str

# Health check
@app.get("/health")
def health():
    return {"status": "healthy", "model": "fraud_detection_v1"}

# Predict endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        features = np.array(transaction.features).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        is_fraud = bool(prob >= threshold)

        if prob >= 0.7:
            risk_level = "HIGH"
        elif prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            transaction_id="txn_001",
            is_fraud=is_fraud,
            confidence=round(float(prob), 4),
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explain endpoint
@app.post("/explain")
def explain(transaction: Transaction):
    try:
        import shap
        features = np.array(transaction.features).reshape(1, -1)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        
        feature_importance = {
            f"feature_{i}": round(float(v), 4)
            for i, v in enumerate(shap_values[0])
        }

        top_features = dict(
            sorted(feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True)[:10]
        )

        return {
            "confidence": round(float(model.predict_proba(features)[0][1]), 4),
            "top_contributing_features": top_features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model info endpoint
@app.get("/model/info")
def model_info():
    return {
        "model_type": "XGBoost",
        "threshold": threshold,
        "version": "1.0.0",
        "features_expected": model.n_features_in_
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)