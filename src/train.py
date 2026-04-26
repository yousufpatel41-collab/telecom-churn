"""
train.py
Train XGBoost churn classifier + generate SHAP explanations.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

from preprocess import run_pipeline

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/xgb_churn_model.pkl"
OUTPUT_DIR = "outputs"

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*50}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="XGBoost")
    ax.set_title("ROC Curve — Churn Prediction")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No Churn", "Churn"], ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    return auc


def generate_shap(model, X_test, feature_names):
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance — Churn Drivers")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved to {OUTPUT_DIR}/shap_summary.png")

    # Waterfall for top churn risk customer
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=feature_names,
    )
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall_sample.png", dpi=150, bbox_inches="tight")
    plt.close()


def simulate_revenue_impact(model, X_test, y_test, monthly_arpu=45.0):
    """Simulate preventable revenue loss by retention tier."""
    y_prob = model.predict_proba(X_test)[:, 1]
    results = X_test.copy()
    results["churn_prob"] = y_prob
    results["actual_churn"] = y_test.values

    # Risk tiers
    results["risk_tier"] = pd.cut(
        results["churn_prob"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    # Simulate €-at-risk (12-month horizon)
    results["revenue_at_risk"] = results["churn_prob"] * monthly_arpu * 12

    summary = results.groupby("risk_tier").agg(
        customers=("churn_prob", "count"),
        avg_churn_prob=("churn_prob", "mean"),
        total_revenue_at_risk=("revenue_at_risk", "sum"),
    ).round(2)

    print("\n📊 Revenue Impact Simulation:")
    print(summary)
    return results, summary


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X, y = run_pipeline(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Churn rate — Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    print("\nTraining XGBoost model...")
    model = train_model(X_train, y_train)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nCV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    evaluate_model(model, X_test, y_test, X.columns.tolist())
    generate_shap(model, X_test, X.columns.tolist())
    simulate_revenue_impact(model, X_test, y_test)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved to {MODEL_PATH}")
