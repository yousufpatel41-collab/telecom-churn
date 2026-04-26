"""
preprocess.py
Feature engineering pipeline for Telecom Churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix TotalCharges (whitespace → NaN → impute with median)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID (not predictive)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Tenure buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"],
    )

    # Charge per month ratio
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Count of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(v not in ["No", "No internet service", "No phone service"]
                        for v in row), axis=1
    )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def get_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def run_pipeline(filepath: str):
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return get_features_target(df)


if __name__ == "__main__":
    X, y = run_pipeline("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"Features: {X.shape}, Target distribution:\n{y.value_counts()}")
