"""
FreshChain - Module 2/3: Logistic Regression Analysis for Fruit Spoilage Prediction.

Covers:
1. Binary Classification formulation: Spoilage Quality (1 = Bad / Spoiled, 0 = Good / Fresh)
2. Mathematical foundation:
   - Log-odds (logit): ln(p / (1 - p)) = beta_0 + sum(beta_i * X_i)
   - Sigmoid activation function: p = 1 / (1 + exp(-z))
   - Maximum Likelihood Estimation & Log Loss (Binary Cross-Entropy)
   - Odds Ratios: OR_i = exp(beta_i) (interpretable environmental risk factors)
3. Standardized feature scaling (StandardScaler) for fair coefficient comparison
4. Evaluation metrics: Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss
5. Decision Threshold Sensitivity: Precision-Recall trade-off in cold-chain logistics
6. Plots: ROC Curve, Sigmoid Probability Distribution, Confusion Matrix, Odds Ratio Bar Chart
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, log_loss, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "logistic_regression"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
CATEGORICAL_COLS = ["Fruit"]


def to_serializable(obj):
    """Recursively convert NumPy data types to standard Python types for JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


def load_and_split_data():
    """Load clean data, standardize numeric features, one-hot encode categories."""
    df = pd.read_csv(DATA_PATH)
    
    # 1 = Spoiled (Bad), 0 = Fresh (Good)
    y = (df["Class"] == "Bad").astype(int)
    
    # One-hot encode Fruit with drop_first=True
    cat_df = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    
    # Split 70% train, 15% val, 15% test
    X_raw = pd.concat([df[NUMERIC_COLS], cat_df], axis=1)
    
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Fit StandardScaler strictly on training set
    scaler = StandardScaler()
    scaler.fit(X_train_raw[NUMERIC_COLS])
    
    def transform_split(df_split):
        num_scaled = pd.DataFrame(
            scaler.transform(df_split[NUMERIC_COLS]),
            columns=NUMERIC_COLS,
            index=df_split.index
        )
        cat_part = df_split.drop(columns=NUMERIC_COLS)
        return pd.concat([num_scaled, cat_part], axis=1)
        
    X_train = transform_split(X_train_raw)
    X_val = transform_split(X_val_raw)
    X_test = transform_split(X_test_raw)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, list(X_train.columns)


def evaluate_logistic():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_split_data()
    
    # 1. Fit Logistic Regression with L2 Regularization (C=1.0)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predicted probabilities: P(Spoiled=1)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # Standard metrics at default threshold 0.5
    acc = float(accuracy_score(y_test, y_test_pred))
    prec = float(precision_score(y_test, y_test_pred))
    rec = float(recall_score(y_test, y_test_pred))
    f1 = float(f1_score(y_test, y_test_pred))
    auc = float(roc_auc_score(y_test, y_test_proba))
    loss = float(log_loss(y_test, y_test_proba))
    cm = confusion_matrix(y_test, y_test_pred).tolist()
    
    # Coefficients and Odds Ratios
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    coef_records = []
    for feat, coef, odds in zip(feature_names, coefficients, odds_ratios):
        coef_records.append({
            "Feature": feat,
            "Coefficient_Beta": round(float(coef), 4),
            "Odds_Ratio_exp_Beta": round(float(odds), 4),
            "Impact_Interpretation": (
                f"Multiplies odds of spoilage by {odds:.2f}x per 1 std dev increase"
                if odds >= 1.0 else
                f"Reduces odds of spoilage by {(1.0 - odds)*100:.1f}% per 1 std dev increase"
            )
        })
    coef_records = sorted(coef_records, key=lambda x: abs(x["Coefficient_Beta"]), reverse=True)
    
    # Threshold sensitivity analysis (0.2 to 0.8)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thresh_table = []
    for t in thresholds:
        pred_t = (y_test_proba >= t).astype(int)
        thresh_table.append({
            "Threshold": t,
            "Accuracy": round(float(accuracy_score(y_test, pred_t)), 4),
            "Precision": round(float(precision_score(y_test, pred_t, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, pred_t, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y_test, pred_t, zero_division=0)), 4)
        })
        
    # -------------------------------------------------------------
    # Visualization 1: ROC Curve
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#17a888", linewidth=2.5, label=f"Logistic Regression (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Random Baseline (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Visualization 2: Sigmoid Probability Distribution vs. Log-Odds (z)
    # -------------------------------------------------------------
    z_scores = model.decision_function(X_test)
    sort_idx = np.argsort(z_scores)
    z_sorted = z_scores[sort_idx]
    p_sorted = y_test_proba[sort_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(z_scores, y_test_proba, c=y_test.values, cmap="coolwarm", alpha=0.4, s=20, label="Test Instances")
    ax.plot(z_sorted, p_sorted, color="#12172b", linewidth=2, label="Sigmoid Curve: σ(z) = 1 / (1 + e⁻ᶻ)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Default Threshold (p = 0.5)")
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Logistic Sigmoid Function & Predicted Spoilage Probability", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Log-Odds / Decision Score (z = β₀ + ∑βᵢXᵢ)", fontsize=11)
    ax.set_ylabel("Predicted Probability of Spoilage (p)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "sigmoid_distribution.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Visualization 3: Confusion Matrix Heatmap
    # -------------------------------------------------------------
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    ax.set_title("Logistic Regression Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fresh (0)", "Spoiled (1)"], fontsize=10)
    ax.set_yticklabels(["Fresh (0)", "Spoiled (1)"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    for r in range(2):
        for c in range(2):
            val = cm_arr[r, c]
            color = "white" if val > cm_arr.max() / 2 else "black"
            ax.text(c, r, f"{val}", ha="center", va="center", color=color, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Visualization 4: Odds Ratios Bar Chart
    # -------------------------------------------------------------
    df_or = pd.DataFrame(coef_records).sort_values(by="Odds_Ratio_exp_Beta", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#ef6f6c" if o < 1.0 else "#17a888" for o in df_or["Odds_Ratio_exp_Beta"]]
    ax.barh(df_or["Feature"], df_or["Odds_Ratio_exp_Beta"], color=colors, height=0.55)
    ax.axvline(1.0, color="#12172b", linestyle="--", linewidth=1.5, label="Baseline Odds Ratio (OR = 1.0)")
    ax.set_title("Logistic Regression Odds Ratios (exp(β))", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Odds Ratio (Multiplicative Risk Factor)", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "odds_ratios.png", dpi=300)
    plt.close()
    
    summary = {
        "model_name": "Logistic Regression Classifier",
        "sample_counts": {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test)
        },
        "intercept_beta_0": round(float(model.intercept_[0]), 4),
        "coefficients_and_odds_ratios": coef_records,
        "metrics_at_threshold_0_5": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "log_loss": round(loss, 4),
            "confusion_matrix": cm
        },
        "threshold_sensitivity": thresh_table,
        "viva_takeaway": (
            "Logistic Regression performs binary classification by wrapping a linear combination of features "
            "in the Sigmoid activation function σ(z) = 1 / (1 + e⁻ᶻ). It outputs calibrated probabilities between 0 and 1. "
            f"The model achieves an ROC-AUC of {auc*100:.2f}% and an F1-Score of {f1*100:.2f}%. "
            "Unlike Linear Regression which uses MSE, Logistic Regression is optimized using Maximum Likelihood Estimation via Log Loss."
        )
    }
    
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(to_serializable(summary), f, indent=2)
        
    print(f"[logistic_regression] Saved reports and plots to {REPORT_DIR}")
    return summary


if __name__ == "__main__":
    evaluate_logistic()
