"""Train and evaluate a complete preprocessing/model pipeline."""
import hashlib
import json
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from preprocess import ROOT, DATA_PATH, MODEL_PATH, load_data, build_preprocessor

def build_model():
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, eval_metric="auc", random_state=42, n_jobs=2)),
    ])

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = build_model()
    scores = cross_val_score(model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= .5).astype(int)
    metrics = {
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "random_state": 42, "train_rows": len(X_train), "test_rows": len(X_test),
        "cv_roc_auc_mean": float(scores.mean()), "cv_roc_auc_std": float(scores.std()),
        "roc_auc": float(roc_auc_score(y_test, prob)), "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred)), "f1": float(f1_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "threshold": .5,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
