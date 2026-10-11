"""Fitted preprocessing shared by training and inference."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/churn_pipeline.pkl"
DATA_PATH = ROOT / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    y = df.pop("Churn").map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Churn must contain only Yes and No")
    return prepare_features(df), y.astype(int)

def prepare_features(df):
    df = df.drop(columns=["customerID"], errors="ignore").copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def build_preprocessor():
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
        ("categorical", Pipeline([
            ("missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), make_column_selector(dtype_include=object)),
    ])
