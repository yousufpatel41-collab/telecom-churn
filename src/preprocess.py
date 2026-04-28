"""preprocess.py — Feature engineering pipeline for Telecom Churn dataset."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def engineer_features(df):
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0,12,24,48,60,72],
        labels=["0-1yr","1-2yr","2-4yr","4-5yr","5+yr"]
    )
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    service_cols = ["PhoneService","MultipleLines","InternetService",
                    "OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]
    df["num_services"] = df[service_cols].apply(
        lambda r: sum(v not in ["No","No internet service","No phone service"] for v in r), axis=1)
    return df


def encode_categoricals(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object","category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def run_pipeline(filepath):
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return df.drop(columns=["Churn"]), df["Churn"]
