import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# Load model at startup
MODEL_PATH = "models/xgb_churn_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None  # run train.py first

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Real-time churn risk scoring for telecom customers. "
                "Built by Yousuf Patel | M.Sc. ISM Munich",
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=72, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    recommended_action: str
    monthly_revenue_at_risk: float


RISK_TIERS = {
    (0.0, 0.3): ("Low Risk 🟢", "Standard retention email campaign"),
    (0.3, 0.6): ("Medium Risk 🟡", "Proactive call from retention specialist"),
    (0.6, 1.0): ("High Risk 🔴", "Immediate VIP intervention + discount offer"),
}


def get_risk_tier(prob: float):
    for (low, high), (tier, action) in RISK_TIERS.items():
        if low <= prob < high:
            return tier, action
    return "High Risk 🔴", "Immediate VIP intervention + discount offer"


def preprocess_input(customer: CustomerFeatures) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    data = customer.dict()
    df = pd.DataFrame([data])

    # Derived features
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"],
    )
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(v not in ["No", "No internet service", "No phone service"] for v in row),
        axis=1,
    )

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "model_loaded": model is not None}


@app.post("/predict", response_model=ChurnResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures, customer_id: str = "CUST-001"):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    df = preprocess_input(customer)
    prob = float(model.predict_proba(df)[:, 1][0])
    tier, action = get_risk_tier(prob)

    return ChurnResponse(
        customer_id=customer_id,
        churn_probability=round(prob, 4),
        risk_tier=tier,
        recommended_action=action,
        monthly_revenue_at_risk=round(customer.MonthlyCharges * prob, 2),
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for i, customer in enumerate(customers):
        df = preprocess_input(customer)
        prob = float(model.predict_proba(df)[:, 1][0])
        tier, action = get_risk_tier(prob)
        results.append({
            "customer_id": f"CUST-{i+1:04d}",
            "churn_probability": round(prob, 4),
            "risk_tier": tier,
            "recommended_action": action,
        })

    results.sort(key=lambda x: x["churn_probability"], reverse=True)
    return {"total_customers": len(results), "predictions": results}
