"""
Module C: Classification and Regression Metrics
Reference Deck: Module_5_Session_M5-3.pptx
Authoritative requirements:
- Classification metrics: Accuracy paradox, Precision, Recall, Specificity, F_beta (F0.5, F1, F2)
- Harmonic vs Arithmetic mean
- Regression metrics: MAE (L1), MSE (L2), RMSE, MAPE (zero caveats), R2 score
- Huber loss formulation and delta = 1.345 threshold
- Exact 10-Student Placement & Salary Package Case Study (Slide 9-12)
  - S08 False Positive & $45k Outlier impact
  - SAE = $65.0k, SSE = 2,113.0k^2, MAE = $6.50k, RMSE = $14.54k, R2 = 0.8973
- Self-assessment concept checks
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import HuberRegressor


def get_case_study_data() -> pd.DataFrame:
    """Exact 10-student dataset from Slide 9."""
    data = {
        "Student": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"],
        "CGPA": [8.5, 7.2, 9.1, 6.8, 8.0, 7.0, 8.8, 7.5, 8.2, 6.9],
        "Coding": [85, 65, 92, 58, 72, 61, 89, 68, 80, 59],
        "TruePlaced": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "PredProb": [0.88, 0.25, 0.95, 0.15, 0.72, 0.40, 0.91, 0.60, 0.82, 0.20],
        "PredPlaced": [1, 0, 1, 0, 1, 0, 1, 1, 1, 0],  # S08 is False Alarm (p=0.60 >= 0.50)
        "TrueSalary": [85.0, 0.0, 110.0, 0.0, 75.0, 0.0, 95.0, 0.0, 80.0, 0.0],
        "PredSalary": [82.0, 0.0, 105.0, 0.0, 70.0, 0.0, 90.0, 45.0, 78.0, 0.0]  # S08 outlier $45k
    }
    df = pd.DataFrame(data)
    df["AbsError"] = np.abs(df["TrueSalary"] - df["PredSalary"])
    df["SqError"] = (df["TrueSalary"] - df["PredSalary"]) ** 2
    return df


def compute_f_beta(precision: float, recall: float, beta: float) -> float:
    """Generalized F-beta formula: (1 + beta^2) * (P * R) / (beta^2 * P + R)."""
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return float((1 + b2) * (precision * recall) / (b2 * precision + recall))


def compute_huber_loss(residuals: np.ndarray, delta=1.345) -> np.ndarray:
    """Huber Loss formula: 0.5*e^2 if |e|<=delta else delta*(|e| - 0.5*delta)."""
    abs_e = np.abs(residuals)
    linear_mask = abs_e > delta
    loss = np.empty_like(residuals, dtype=float)
    loss[~linear_mask] = 0.5 * (residuals[~linear_mask] ** 2)
    loss[linear_mask] = delta * (abs_e[linear_mask] - 0.5 * delta)
    return loss


def run_module_c_walkthrough() -> dict:
    df = get_case_study_data()

    y_true_c = df["TruePlaced"].values
    y_pred_c = df["PredPlaced"].values

    # 1. Classification Metrics
    cm = confusion_matrix(y_true_c, y_pred_c)  # [[TN, FP], [FN, TP]] -> [[4, 1], [0, 5]]
    tn, fp, fn, tp = cm.ravel()
    acc = float(accuracy_score(y_true_c, y_pred_c))
    prec = float(precision_score(y_true_c, y_pred_c))
    rec = float(recall_score(y_true_c, y_pred_c))
    f1 = float(f1_score(y_true_c, y_pred_c))
    spec = float(tn / (tn + fp))
    f05 = compute_f_beta(prec, rec, beta=0.5)
    f2 = compute_f_beta(prec, rec, beta=2.0)

    # 2. Regression Metrics
    y_true_s = df["TrueSalary"].values
    y_pred_s = df["PredSalary"].values
    residuals = y_true_s - y_pred_s

    sae = float(np.sum(np.abs(residuals)))
    sse = float(np.sum(residuals ** 2))
    mae = float(mean_absolute_error(y_true_s, y_pred_s))
    mse = float(mean_squared_error(y_true_s, y_pred_s))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_s, y_pred_s))

    # Huber Regressor fit
    huber = HuberRegressor(epsilon=1.35)
    huber.fit(y_pred_s.reshape(-1, 1), y_true_s)

    return {
        "case_study_table": df.to_dict(orient="records"),
        "classification": {
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "f05_score": round(f05, 4),
            "f2_score": round(f2, 4),
            "accuracy_paradox_demo": {
                "dummy_majority_accuracy": 0.50,
                "dummy_recall": 0.0,
                "note": "On balanced 50% data, dummy achieves 50%. On 99:1 imbalance, dummy achieves 99% accuracy but 0% recall!"
            }
        },
        "regression": {
            "sae_k": round(sae, 2),
            "sse_k2": round(sse, 2),
            "mae_k": round(mae, 2),
            "mse_k2": round(mse, 2),
            "rmse_k": round(rmse, 2),
            "rmse_to_mae_ratio": round(rmse / mae, 2),
            "r2_score": round(r2, 4),
            "huber_intercept": round(float(huber.intercept_), 2),
            "huber_coef": round(float(huber.coef_[0]), 4),
            "outlier_analysis": "S08 produces a $45k error. In MAE, it contributes 45/10 = $4.5k. In MSE, it contributes 45^2/10 = 202.5k^2, causing RMSE to jump to $14.54k (2.24x MAE)."
        },
        "quiz": [
            {"q": "Why does MAE optimize for the conditional median while MSE optimizes for the mean?", "a": "MAE uses an L1 linear penalty whose derivative is constant (+/- 1), so the zero-gradient point occurs at the median. MSE uses an L2 quadratic penalty whose derivative is linear, so the zero-gradient point occurs at the expected mean."},
            {"q": "How does Huber loss transition between MSE and MAE?", "a": "It uses MSE (quadratic 0.5*e^2) for errors |e| <= delta for smooth gradients, and MAE (linear delta*(|e|-0.5*delta)) for errors |e| > delta to bound outlier gradients."}
        ]
    }
