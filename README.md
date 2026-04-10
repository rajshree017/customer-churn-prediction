# 📉 Customer Churn Prediction — LightGBM + SHAP

Predict which customers will churn using gradient boosting with full SHAP explainability.

## 📁 Folder Structure
```
5_churn_prediction/
├── train.py              # Feature engineering, training, SHAP
├── requirements.txt
└── README.md
```

## 🚀 Setup & Run
```bash
pip install -r requirements.txt

# Download dataset from Kaggle:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place CSV in this folder, then:
python train.py
```

## 📊 Expected Results
| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.85 |
| F1 (Churn class) | ~0.62 |

Outputs: `roc_curve.png`, `shap_summary.png`

## 🧠 What You'll Learn
- Feature engineering for tabular data
- LightGBM with early stopping
- Threshold tuning for imbalanced classes
- SHAP values for model explainability
