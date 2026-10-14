"""
M2 · Session 9-10 — Feature Scaling & Standardization.

Demonstrates, on this project's numeric features, exactly what the session
deck covers: Min-Max normalization, Standardization (Z-score), Robust
scaling, a before/after comparison on one sample row, and a demonstration
that a distance-based model (KNN) benefits from scaling while the tree
model the final pipeline actually uses does not.

Run standalone with: python src/scaling.py
Outputs: reports/scaling/summary.json
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data_cleaning import run as clean_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "scaling"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]


def before_after_table(df: pd.DataFrame) -> dict:
    """Show one sample row's raw values next to Min-Max, Standard, and
    Robust scaled versions — the "before vs. after, one row" table from
    the session deck.
    """
    sample = df[NUMERIC_COLS].iloc[[0]]

    mm = MinMaxScaler().fit(df[NUMERIC_COLS])
    ss = StandardScaler().fit(df[NUMERIC_COLS])
    rs = RobustScaler().fit(df[NUMERIC_COLS])

    return {
        "raw": sample.iloc[0].round(3).to_dict(),
        "minmax_scaled": dict(zip(NUMERIC_COLS, mm.transform(sample)[0].round(3))),
        "standard_scaled": dict(zip(NUMERIC_COLS, ss.transform(sample)[0].round(3))),
        "robust_scaled": dict(zip(NUMERIC_COLS, rs.transform(sample)[0].round(3))),
    }


def scaler_comparison_table() -> dict:
    return {
        "Min-Max Scaler": {"output_range": "[0, 1] fixed", "uses": "min, max",
                             "outlier_sensitivity": "High", "best_for": "Bounded inputs, neural networks"},
        "Standard Scaler": {"output_range": "Unbounded, mean 0", "uses": "mean, std dev",
                              "outlier_sensitivity": "Moderate", "best_for": "Linear/distance-based models"},
        "Robust Scaler": {"output_range": "Unbounded, median 0", "uses": "median, IQR",
                            "outlier_sensitivity": "Low", "best_for": "Data with real outliers"},
    }


def does_scaling_change_results(df: pd.DataFrame) -> dict:
    """The session's headline demo: KNN accuracy with vs without scaling,
    same split, same k. This is what motivates 'scaling matters for
    distance-based models' concretely rather than just asserting it.
    """
    X = df[NUMERIC_COLS]
    y = (df["Class"] == "Good").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Unscaled
    knn_raw = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))

    # Scaled — fit on train, transform on test (never the other way round)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn_scaled = KNeighborsClassifier(n_neighbors=7).fit(X_train_scaled, y_train)
    acc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))

    return {
        "model": "KNN (k=7)",
        "accuracy_unscaled": round(acc_raw, 4),
        "accuracy_standardized": round(acc_scaled, 4),
        "accuracy_gain_pts": round((acc_scaled - acc_raw) * 100, 2),
    }


def scale_sensitivity_by_model() -> dict:
    return {
        "scaling_matters": ["KNN", "K-Means / clustering", "SVM", "Logistic & Linear Regression", "PCA", "Neural networks"],
        "scale_invariant": ["Decision Trees", "Random Forest", "Gradient Boosting (XGBoost, LightGBM)", "Naive Bayes (usually)"],
    }


def run() -> dict:
    df = clean_run(save=False)
    report = {
        "before_after_sample_row": before_after_table(df),
        "scaler_comparison": scaler_comparison_table(),
        "knn_scaling_experiment": does_scaling_change_results(df),
        "scale_sensitivity_by_model": scale_sensitivity_by_model(),
    }
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[scaling] KNN scaling experiment:", report["knn_scaling_experiment"])
    print(f"[scaling] Done. summary.json saved to {REPORT_DIR}")
    return report


if __name__ == "__main__":
    run()
