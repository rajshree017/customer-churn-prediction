"""
Customer Churn Prediction — LightGBM + SHAP
---------------------------------------------
Dataset: Telco Customer Churn (Kaggle)
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay,
)
import warnings
warnings.filterwarnings("ignore")

# ── Load & clean ──────────────────────────────────────────────
def load_data(path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df

# ── Feature engineering ───────────────────────────────────────
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AvgMonthlyCharge"]    = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargePerService"]    = df["MonthlyCharges"] / (
        df[["PhoneService","InternetService","OnlineSecurity",
            "OnlineBackup","DeviceProtection","TechSupport",
            "StreamingTV","StreamingMovies"]]
        .apply(lambda col: col.map({"Yes": 1, "No": 0}).fillna(0))
        .sum(axis=1) + 1
    )
    df["IsLongTermContract"]  = df["Contract"].isin(["One year","Two year"]).astype(int)
    df["HasMultipleServices"] = (
        df[["OnlineSecurity","OnlineBackup","TechSupport"]].eq("Yes").sum(axis=1) >= 2
    ).astype(int)
    return df

# ── Encode categoricals ───────────────────────────────────────
def encode(df: pd.DataFrame):
    le = LabelEncoder()
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# ── Train ─────────────────────────────────────────────────────
def train(X_train, y_train, X_val, y_val):
    params = dict(
        objective      = "binary",
        metric         = "auc",
        learning_rate  = 0.05,
        num_leaves     = 31,
        max_depth      = 6,
        min_child_samples = 20,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 5,
        n_estimators   = 1000,
        early_stopping_rounds = 50,
        verbose        = -1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    return model

# ── Evaluate ──────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    print(f"\nROC-AUC : {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Stay","Churn"]))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=120); plt.close()
    print("[✓] roc_curve.png saved")

# ── SHAP explainability ───────────────────────────────────────
def explain(model, X_test: pd.DataFrame):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(vals, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=120); plt.close()
    print("[✓] shap_summary.png saved")

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df = feature_engineer(df)
    df = encode(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    model = train(X_tr, y_tr, X_val, y_val)
    evaluate(model, X_test, y_test)
    explain(model, X_test)

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
    print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
