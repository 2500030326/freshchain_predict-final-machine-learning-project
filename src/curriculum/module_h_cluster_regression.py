"""
Module H: Cluster-Augmented Regression
Reference Deck: Cluster_Augmented_Regression (1).pptx
Authoritative requirements:
- Exact 5-House Dataset:
    House 1: L=1,   S=1000, P=150, Cluster=A
    House 2: L=2,   S=1200, P=180, Cluster=A
    House 3: L=8,   S=1000, P=300, Cluster=B
    House 4: L=9,   S=1200, P=340, Cluster=B
    House 5: L=8.5, S=1100, P=320, Cluster=B
- Step 1: K-Means (K=2) on Location L -> Centroids 1.5 and 8.5
- Step 2: Baseline OLS (Size only):
    S_bar = 1100, P_bar = 258, beta_1 = 0.175, beta_0 = 65.5
    P_hat = 65.5 + 0.175 * S
    Residuals: [-90.5, -95.5, +59.5, +64.5, +62.0] -> R2 approx 0.04
- Diagnosis: Omitted-variable bias (sign pattern in residuals)
- Step 3: Augmented Regression (Size + Cluster + Interaction):
    P_hat = beta_0 + beta_1*S + beta_2*C + beta_3*(S*C)
    Combined equation: P_hat = 0.15*S + 100*C + 0.05*(S*C)
    All residuals = 0.0 -> R2 = 1.0
- Interactive house price predictor & regression line visualizer
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score


def get_5house_dataset() -> pd.DataFrame:
    """Exact 5-house dataset from Slide 2."""
    data = {
        "House": [1, 2, 3, 4, 5],
        "Location_L": [1.0, 2.0, 8.0, 9.0, 8.5],
        "Size_S": [1000.0, 1200.0, 1000.0, 1200.0, 1100.0],
        "Price_P": [150.0, 180.0, 300.0, 340.0, 320.0],
        "Cluster": ["A", "A", "B", "B", "B"]
    }
    return pd.DataFrame(data)


def compute_baseline_ols(df: pd.DataFrame) -> dict:
    """Step 2 manual OLS calculations from Slide 4 & 5."""
    S = df["Size_S"].values
    P = df["Price_P"].values

    s_bar = float(np.mean(S))  # 1100.0
    p_bar = float(np.mean(P))  # 258.0

    cov_sp = float(np.sum((S - s_bar) * (P - p_bar)))  # 7000.0
    var_s = float(np.sum((S - s_bar) ** 2))            # 40000.0

    beta_1 = cov_sp / var_s  # 0.175
    beta_0 = p_bar - beta_1 * s_bar  # 65.5

    p_pred = beta_0 + beta_1 * S
    residuals = P - p_pred

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((P - p_bar) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)

    return {
        "s_mean": s_bar,
        "p_mean": p_bar,
        "numerator_cov": cov_sp,
        "denominator_var": var_s,
        "beta_0_intercept": round(beta_0, 2),
        "beta_1_slope": round(beta_1, 4),
        "fitted_equation": f"P_hat = {beta_0:.1f} + {beta_1:.3f} * S",
        "predictions": [round(float(p), 1) for p in p_pred],
        "residuals": [round(float(r), 1) for r in residuals],
        "r2_score": round(r2, 4)
    }


def compute_augmented_regression(df: pd.DataFrame) -> dict:
    """Step 3 cluster-augmented regression with interaction from Slide 7, 8, 9."""
    S = df["Size_S"].values
    P = df["Price_P"].values
    # C = 0 for A, 1 for B
    C = (df["Cluster"] == "B").astype(float).values
    SC = S * C

    X_aug = np.column_stack([S, C, SC])
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_aug, P)

    # Hand-derived coefficients from slide 9:
    # beta_0 = 0.0, beta_1 = 0.15, beta_2 = 100.0, beta_3 = 0.05
    p_pred = reg.predict(X_aug)
    residuals = P - p_pred
    r2 = float(r2_score(P, p_pred))

    return {
        "beta_0": round(float(reg.intercept_), 2),
        "beta_1_size": round(float(reg.coef_[0]), 4),
        "beta_2_cluster": round(float(reg.coef_[1]), 2),
        "beta_3_interaction": round(float(reg.coef_[2]), 4),
        "augmented_equation": "P_hat = 0.15*S + 100*C + 0.05*(S*C)",
        "predictions": [round(float(p), 1) for p in p_pred],
        "residuals": [round(float(r), 1) for r in residuals],
        "r2_score": round(r2, 4)
    }


def predict_house_price(size_sqft: float, cluster: str) -> dict:
    """Live interactive prediction function."""
    c_val = 1.0 if cluster.upper() == "B" else 0.0
    # Baseline
    p_base = 65.5 + 0.175 * size_sqft
    # Augmented
    p_aug = 0.15 * size_sqft + 100.0 * c_val + 0.05 * (size_sqft * c_val)
    return {
        "size_sqft": size_sqft,
        "cluster": cluster,
        "baseline_predicted_price_k": round(p_base, 2),
        "augmented_predicted_price_k": round(p_aug, 2),
        "difference_k": round(p_aug - p_base, 2)
    }


def run_module_h_walkthrough() -> dict:
    df = get_5house_dataset()

    # Step 1: K-Means on Location L
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(df[["Location_L"]])
    centroids = sorted(km.cluster_centers_.ravel().tolist())

    # Step 2: Baseline OLS
    base_res = compute_baseline_ols(df)

    # Step 3: Augmented Model
    aug_res = compute_augmented_regression(df)

    # Table comparing both models per house
    table_comparison = []
    for idx, row in df.iterrows():
        table_comparison.append({
            "House": int(row["House"]),
            "Location_L": float(row["Location_L"]),
            "Size_S": float(row["Size_S"]),
            "Actual_Price": float(row["Price_P"]),
            "Cluster": row["Cluster"],
            "Baseline_Pred": base_res["predictions"][idx],
            "Baseline_Residual": base_res["residuals"][idx],
            "Augmented_Pred": aug_res["predictions"][idx],
            "Augmented_Residual": aug_res["residuals"][idx],
        })

    return {
        "dataset_table": df.to_dict(orient="records"),
        "step1_kmeans": {
            "cluster_a_centroid": round(centroids[0], 2),  # 1.5
            "cluster_b_centroid": round(centroids[1], 2),  # 8.5
            "convergence": "Immediate on iteration 1 (all points already closest to their group centroid)."
        },
        "step2_baseline_model": base_res,
        "step3_augmented_model": aug_res,
        "comparison_table": table_comparison,
        "diagnosis_omitted_variable_bias": {
            "cluster_a_residuals": "[-90.5, -95.5] (All negative -> Model consistently over-predicts)",
            "cluster_b_residuals": "[+59.5, +64.5, +62.0] (All positive -> Model consistently under-predicts)",
            "root_cause": "Location/Cluster drives price but was omitted from the baseline model."
        },
        "takeaway": "The jump from R2 = 0.04 to R2 = 1.0 shows Location/Cluster was the dominant driver of price — not a minor refinement, but a variable the baseline model could not afford to omit."
    }
