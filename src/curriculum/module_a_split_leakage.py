"""
Module A: Training, Validation, Testing and Data Leakage
Reference Deck: module_5_session_M5_S1 (L59-60).pptx
Authoritative requirements:
- Train, validation, test set roles and pairwise-disjointness
- Empirical Risk vs True Risk definitions: R_emp(f) = (1/n) sum L(f(x_i), y_i)
- 3-way partition (approx 60/20/20) and holdout discipline
- Data leakage taxonomy (preprocessing, temporal, target contamination)
- Scikit-learn Pipeline leakage prevention
- 4-Fold cross-validation walkthrough on ExampleDataSetPP (10 students)
- Learning curves and validation curves diagnosis
- Self-assessment quiz and production rules
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline


def get_example_placement_dataset() -> pd.DataFrame:
    """10-student reference dataset from Slide 12."""
    data = {
        "Student": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"],
        "CGPA": [8.5, 7.2, 9.1, 6.8, 8.0, 7.0, 8.8, 7.5, 8.2, 6.9],
        "Aptitude": [82, 65, 90, 55, 75, 60, 88, 70, 80, 58],
        "Coding": [85, 60, 92, 50, 78, 58, 89, 68, 82, 55],
        "Communication": [80, 70, 85, 65, 75, 68, 82, 72, 78, 62],
        "Internship": [2, 0, 3, 0, 1, 0, 2, 1, 2, 0],
        "Projects": [3, 1, 4, 1, 2, 1, 3, 2, 3, 1],
        "Placed": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Balanced 50% placed
    }
    return pd.DataFrame(data)


def compute_empirical_risk(y_true: np.ndarray, y_pred: np.ndarray, loss_type="0-1") -> float:
    """Calculates empirical risk: R_emp(f) = (1/n) * sum(L(f(x_i), y_i))."""
    if loss_type == "0-1":
        return float(np.mean(y_true != y_pred))
    elif loss_type == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    raise ValueError(f"Unknown loss_type: {loss_type}")


def run_module_a_walkthrough() -> dict:
    df = get_example_placement_dataset()
    features = ["CGPA", "Aptitude", "Coding", "Communication", "Internship", "Projects"]
    X = df[features]
    y = df["Placed"]

    # 1. Vault Split: 80% Dev (S01-S08), 20% Vault Test (S09-S10)
    # Using fixed index split matching slide 13
    dev_mask = df["Student"].isin(["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"])
    X_dev, y_dev = X[dev_mask], y[dev_mask]
    X_test, y_test = X[~dev_mask], y[~dev_mask]

    # 2. Leakage-free pipeline
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

    # 3. 4-Fold Stratified CV on Dev set (2 students per validation fold)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_dev, y_dev, cv=skf, scoring="accuracy")

    # 4. Final fit on full dev set and evaluate on vault-locked test set
    pipeline.fit(X_dev, y_dev)
    test_score = float(pipeline.score(X_test, y_test))
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    # 5. Model weights from pipeline
    weights = dict(zip(features, [round(float(w), 3) for w in pipeline.named_steps["logisticregression"].coef_[0]]))

    return {
        "dataset": df.to_dict(orient="records"),
        "dev_students": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"],
        "test_students": ["S09", "S10"],
        "cv_scores_4fold": [round(float(s), 4) for s in cv_scores],
        "mean_cv_accuracy": round(float(cv_scores.mean()), 4),
        "test_accuracy": round(test_score, 4),
        "test_predictions": [
            {"student": "S09", "actual": int(y_test.iloc[0]), "prob": round(float(test_probs[0]), 3), "pred": int(test_probs[0] >= 0.5)},
            {"student": "S10", "actual": int(y_test.iloc[1]), "prob": round(float(test_probs[1]), 3), "pred": int(test_probs[1] >= 0.5)},
        ],
        "learned_weights": weights,
        "quiz": [
            {"q": "Why is validation score optimistically biased?", "a": "Because hyperparameter decisions were made to explicitly maximize it."},
            {"q": "What happens if you run StandardScaler before train_test_split?", "a": "Data Leakage! Test statistics corrupt the training distribution."},
            {"q": "When should you use Stratified K-Fold over Standard K-Fold?", "a": "For imbalanced classification to prevent class-starved folds."}
        ]
    }
