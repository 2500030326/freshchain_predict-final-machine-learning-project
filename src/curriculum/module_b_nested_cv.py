"""
Module B: Stratified and Nested Cross-Validation
Reference Deck: module_5_session_M5_S2 (L61-62).pptx
Authoritative requirements:
- Standard K-Fold failure under imbalance vs Stratified K-Fold stability
- Hyperparameter selection bias & tuning leakage
- Nested CV double-loop architecture (Outer=Generalization, Inner=Tuning)
- Model-fit count formula: N_fits = K_outer * (K_inner * |Lambda| + 1)
- Worked Placement Benchmark trace (N=1,000, 67% Placed / 33% Not Placed)
- Scikit-Learn Leakage-Free Nested Stratified Pipeline
- Self-assessment concept checks
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def compute_nested_fits_count(k_outer: int, k_inner: int, num_grid_params: int) -> int:
    """Formula from Slide 8 & 18: N_fits = K_outer * (K_inner * |Lambda| + 1)."""
    return k_outer * (k_inner * num_grid_params + 1)


def generate_placement_benchmark(n_samples=1000, random_state=42) -> tuple[pd.DataFrame, pd.Series]:
    """Generates benchmark dataset with exact 67% Placed / 33% Not Placed ratio from Slide 9."""
    np.random.seed(random_state)
    n_placed = int(0.67 * n_samples)
    n_unplaced = n_samples - n_placed

    cgpa_p = np.random.normal(8.0, 0.8, n_placed).clip(6.0, 10.0)
    apt_p = np.random.normal(78, 10, n_placed).clip(50, 100)
    code_p = np.random.normal(82, 9, n_placed).clip(55, 100)
    soft_p = np.random.normal(80, 8, n_placed).clip(50, 100)
    intern_p = np.random.poisson(2, n_placed)
    proj_p = np.random.poisson(3, n_placed)
    y_p = np.ones(n_placed, dtype=int)

    cgpa_u = np.random.normal(6.8, 0.7, n_unplaced).clip(5.0, 8.5)
    apt_u = np.random.normal(60, 12, n_unplaced).clip(30, 85)
    code_u = np.random.normal(55, 12, n_unplaced).clip(25, 80)
    soft_u = np.random.normal(65, 10, n_unplaced).clip(40, 85)
    intern_u = np.random.poisson(0.5, n_unplaced)
    proj_u = np.random.poisson(1.5, n_unplaced)
    y_u = np.zeros(n_unplaced, dtype=int)

    X = pd.DataFrame({
        "CGPA": np.concatenate([cgpa_p, cgpa_u]),
        "AptitudeTestScore": np.concatenate([apt_p, apt_u]),
        "CodingTestScore": np.concatenate([code_p, code_u]),
        "SoftSkillsRating": np.concatenate([soft_p, soft_u]),
        "Internships": np.concatenate([intern_p, intern_u]),
        "Projects": np.concatenate([proj_p, proj_u]),
    })
    y = pd.Series(np.concatenate([y_p, y_u]), name="PlacementStatus")
    return X, y


def run_module_b_walkthrough() -> dict:
    X, y = generate_placement_benchmark(n_samples=1000)

    # 1. Verification of class proportion allocation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_proportions = []
    for f_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        y_val = y.iloc[val_idx]
        p_rate = float(y_val.mean())
        fold_proportions.append({
            "fold": f_idx,
            "total_val": len(val_idx),
            "placed_count": int(y_val.sum()),
            "not_placed_count": int((y_val == 0).sum()),
            "placed_pct": round(p_rate * 100, 1),
            "not_placed_pct": round((1 - p_rate) * 100, 1)
        })

    # 2. Formula calculation: K_out=5, K_in=3, |Lambda|=4 (C in 0.01, 0.1, 1.0, 10.0)
    c_grid = [0.01, 0.1, 1.0, 10.0]
    n_fits = compute_nested_fits_count(k_outer=5, k_inner=3, num_grid_params=len(c_grid))

    # 3. Leakage-free nested pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=pipeline, param_grid={'model__C': c_grid}, cv=inner_cv, scoring='f1')
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    nested_f1_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1', n_jobs=-1)

    return {
        "dataset_summary": {
            "total_samples": 1000,
            "placed_ratio": 67.0,
            "not_placed_ratio": 33.0
        },
        "fold_allocation_table": fold_proportions,
        "n_fits_calculation": {
            "k_outer": 5,
            "k_inner": 3,
            "grid_size": len(c_grid),
            "formula": "K_outer * (K_inner * |Lambda| + 1)",
            "total_fits": n_fits
        },
        "nested_f1_scores": [round(float(s), 4) for s in nested_f1_scores],
        "unbiased_mean_f1": round(float(nested_f1_scores.mean()), 4),
        "f1_confidence_margin": round(float(nested_f1_scores.std() * 2), 4),
        "slide_reported_benchmark": {
            "outer_scores": [0.971, 0.932, 0.949, 0.932, 0.931],
            "reported_mean_f1": 0.9430,
            "reported_margin": 0.0308,
            "optimal_c": 1.00
        },
        "quiz": [
            {"q": "What happens if standard K-fold is used on a 99:1 imbalanced dataset?", "a": "Validation folds may contain 0 positive samples, causing Precision/Recall and ROC-AUC evaluation routines to crash or fail."},
            {"q": "Why is reporting the best inner-loop CV score optimistic?", "a": "Because the hyperparameters were selected specifically to maximize that validation set, resulting in selection bias."},
            {"q": "How many total model fits occur in Nested CV with K_out=5, K_in=3, and 10 grid options?", "a": "N_fits = 5 * (3 * 10 + 1) = 155 total model fits."}
        ]
    }
