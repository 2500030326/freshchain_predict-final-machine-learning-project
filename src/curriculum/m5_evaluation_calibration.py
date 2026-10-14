"""
Module 5: Model Evaluation, Selection, and Calibration
======================================================
Complete, self-contained runnable implementations covering:
1. Train / Validation / Test split discipline & data leakage prevention.
2. Stratified K-Fold and Nested Cross-Validation (unbiased model selection).
3. Classification metrics: Accuracy paradox, Confusion Matrix, ROC-AUC vs. PR-AUC under imbalance.
4. Regression metrics: RMSE, MAE, MAPE, R2, and outlier sensitivity stress-test.
5. Probability Calibration: Reliability diagrams, Platt Scaling, Isotonic Regression, and Brier Score.
6. Hyperparameter Search: Grid Search vs. Random Search (Bergstra-Bengio).
7. Learning Curves & Validation Curves: Diagnosing Under-fitting vs. Over-fitting.
8. Statistical Significance: McNemar's test (classifiers) & Paired Bootstrap (regressors).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.stats import uniform, randint, chi2


# =====================================================================
# 1. THREE-WAY SPLIT DISCIPLINE & DATA LEAKAGE PREVENTION
# =====================================================================

def demo_split_discipline():
    print("\n" + "=" * 60)
    print("DEMO 1: THREE-WAY SPLIT DISCIPLINE & LEAKAGE PREVENTION")
    print("=" * 60)

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    # 1. 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Total Dataset:  {len(X)} samples")
    print(f"Training Set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%) -> Parameter fitting")
    print(f"Validation Set: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%) -> Hyperparameter tuning & early stopping")
    print(f"Test Set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%) -> HELD OUT: Evaluated once at the very end")


# =====================================================================
# 2. STRATIFIED & NESTED CROSS-VALIDATION
# =====================================================================

def demo_nested_cross_validation():
    print("\n" + "=" * 60)
    print("DEMO 2: NESTED CROSS-VALIDATION (UNBIASED ERROR ESTIMATION)")
    print("=" * 60)

    X, y = make_classification(n_samples=300, n_features=15, n_informative=5, random_state=42)

    # Inner CV: Tunes hyperparameters
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    inner_clf = GridSearchCV(
        estimator=LogisticRegression(solver="liblinear"),
        param_grid=param_grid,
        cv=inner_cv,
        scoring="roc_auc",
    )

    # Outer CV: Evaluates generalization performance
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nested_scores = cross_val_score(inner_clf, X=X, y=y, cv=outer_cv, scoring="roc_auc")

    # Non-nested (optimistically biased)
    inner_clf.fit(X, y)
    non_nested_score = inner_clf.best_score_

    print(f"Nested CV AUC Scores (5 Outer Folds):   {[round(s, 4) for s in nested_scores]}")
    print(f"Nested CV Unbiased Generalization Mean:  {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")
    print(f"Non-Nested CV Best Score (Optimistic):   {non_nested_score:.4f}")
    print("Nested CV eliminates selection bias by never tuning on the evaluation fold.")


# =====================================================================
# 3. CLASSIFICATION METRICS UNDER CLASS IMBALANCE
# =====================================================================

def demo_imbalance_metrics_roc_vs_pr_auc():
    print("\n" + "=" * 60)
    print("DEMO 3: ACCURACY PARADOX & ROC-AUC VS. PR-AUC (95:5 IMBALANCE)")
    print("=" * 60)

    # 95% Negative (Class 0), 5% Positive (Class 1)
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        weights=[0.95, 0.05],
        flip_y=0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Naive constant model: Always predicts majority class (0)
    naive_preds = np.zeros_like(y_test)
    print(f"Naive Zero-Predictor Accuracy: {accuracy_score(y_test, naive_preds):.4f} (Accuracy lies!)")
    print(f"Naive Zero-Predictor Recall:   {recall_score(y_test, naive_preds, zero_division=0):.4f}")

    # Real classifier
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    print("\n--- Model Performance on Imbalanced Test Set ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision_score(y_test, preds):.4f} | Recall: {recall_score(y_test, preds):.4f} | F1: {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f} (Can look deceptively high due to large TN pool)")
    print(f"PR-AUC:    {pr_auc:.4f}  (Focuses strictly on minority positive class)")


# =====================================================================
# 4. REGRESSION METRICS & OUTLIER SENSITIVITY STRESS-TEST
# =====================================================================

def demo_regression_metrics_outlier_stress():
    print("\n" + "=" * 60)
    print("DEMO 4: REGRESSION METRICS & OUTLIER SENSITIVITY (RMSE VS. MAE)")
    print("=" * 60)

    np.random.seed(42)
    y_true = np.random.uniform(50, 100, size=200)
    y_pred = y_true + np.random.normal(0, 3.0, size=200)  # Clean small errors

    rmse_clean = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_clean = mean_absolute_error(y_true, y_pred)
    mape_clean = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("--- Clean Scenario ---")
    print(f"Clean RMSE: {rmse_clean:.4f} | Clean MAE: {mae_clean:.4f} | Clean MAPE: {mape_clean:.2f}%")

    # Inject 3 extreme outliers
    y_pred_corrupted = y_pred.copy()
    y_pred_corrupted[0] += 120.0
    y_pred_corrupted[1] += 150.0
    y_pred_corrupted[2] -= 100.0

    rmse_corrupted = np.sqrt(mean_squared_error(y_true, y_pred_corrupted))
    mae_corrupted = mean_absolute_error(y_true, y_pred_corrupted)

    print("\n--- After Adding 3 Extreme Outliers (Outlier Sensitivity) ---")
    print(f"Corrupted RMSE: {rmse_corrupted:.4f} (Jumped by {((rmse_corrupted - rmse_clean)/rmse_clean)*100:+.1f}%)")
    print(f"Corrupted MAE:  {mae_corrupted:.4f}  (Jumped by {((mae_corrupted - mae_clean)/mae_clean)*100:+.1f}%)")
    print("Conclusion: RMSE penalizes outliers quadratically (L2); MAE is robust (L1).")


# =====================================================================
# 5. PROBABILITY CALIBRATION: PLATT SCALING & ISOTONIC REGRESSION
# =====================================================================

def demo_probability_calibration():
    print("\n" + "=" * 60)
    print("DEMO 5: PROBABILITY CALIBRATION (PLATT SCALING & ISOTONIC)")
    print("=" * 60)

    X, y = make_classification(n_samples=1500, n_features=12, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Uncalibrated base model (Gradient Boosting / Random Forest often output distorted probabilities)
    base_model = GradientBoostingClassifier(random_state=42)
    base_model.fit(X_train, y_train)
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, raw_probs)

    # 1. Platt Scaling (Sigmoid) with cross-validation
    platt = CalibratedClassifierCV(estimator=GradientBoostingClassifier(random_state=42), method="sigmoid", cv=3)
    platt.fit(X_train, y_train)
    platt_probs = platt.predict_proba(X_test)[:, 1]
    brier_platt = brier_score_loss(y_test, platt_probs)

    # 2. Isotonic Regression (Non-parametric) with cross-validation
    isotonic = CalibratedClassifierCV(estimator=GradientBoostingClassifier(random_state=42), method="isotonic", cv=3)
    isotonic.fit(X_train, y_train)
    iso_probs = isotonic.predict_proba(X_test)[:, 1]
    brier_iso = brier_score_loss(y_test, iso_probs)

    print(f"Raw Base Model Brier Score:       {brier_raw:.4f} (Lower is better)")
    print(f"Platt Scaling (Sigmoid) Brier:    {brier_platt:.4f}")
    print(f"Isotonic Regression Brier:        {brier_iso:.4f}")
    print("Calibration ensures predicted P(y=1)=0.7 corresponds to a true 70% empirical hit rate.")


# =====================================================================
# 6. HYPERPARAMETER SEARCH: GRID SEARCH VS. RANDOM SEARCH
# =====================================================================

def demo_hyperparameter_search():
    print("\n" + "=" * 60)
    print("DEMO 6: GRID SEARCH VS. RANDOM SEARCH (BERGSTRA-BENGIO)")
    print("=" * 60)

    import time

    X, y = make_classification(n_samples=600, n_features=10, random_state=42)

    rf = RandomForestClassifier(random_state=42)

    # Grid Search
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 8],
        "min_samples_split": [2, 5, 10],
    }
    t0 = time.time()
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X, y)
    t_grid = time.time() - t0

    # Random Search (Bergstra & Bengio efficiency: samples broad continuous distributions)
    param_dist = {
        "n_estimators": randint(40, 160),
        "max_depth": randint(3, 10),
        "min_samples_split": randint(2, 12),
    }
    t0 = time.time()
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=15, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
    )
    random_search.fit(X, y)
    t_rand = time.time() - t0

    print(f"Grid Search (27 fits * 3 folds):   Time = {t_grid:.2f}s | Best Score = {grid_search.best_score_:.4f}")
    print(f"Random Search (15 fits * 3 folds): Time = {t_rand:.2f}s | Best Score = {random_search.best_score_:.4f}")
    print("Bergstra & Bengio Theorem: Random Search explores many more distinct values per parameter for equal/lower compute.")


# =====================================================================
# 7. DIAGNOSING LEARNING CURVES
# =====================================================================

def demo_learning_curves():
    print("\n" + "=" * 60)
    print("DEMO 7: LEARNING CURVE DIAGNOSTICS (BIAS VS. VARIANCE)")
    print("=" * 60)

    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    model = LogisticRegression(random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy", random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    diag_df = pd.DataFrame({
        "Training_Samples": train_sizes,
        "Train_Accuracy": np.round(train_mean, 4),
        "Val_Accuracy": np.round(val_mean, 4),
        "Generalization_Gap": np.round(train_mean - val_mean, 4),
    })
    print(diag_df.to_string(index=False))
    print("\nDiagnostic Rules:")
    print(" - Large gap that persists with more data -> High Variance (Overfitting)")
    print(" - High training error & high val error plateauing close together -> High Bias (Underfitting)")


# =====================================================================
# 8. STATISTICAL SIGNIFICANCE TESTS: McNEMAR & PAIRED BOOTSTRAP
# =====================================================================

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[float, float]:
    """McNemar's test with continuity correction on paired binary classification errors."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency Table:
    # b: A correct, B wrong
    # c: B correct, A wrong
    b = int(np.sum(correct_a & (~correct_b)))
    c = int(np.sum((~correct_a) & correct_b))

    if b + c == 0:
        return 0.0, 1.0

    # Edwards' continuity correction
    chi2_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
    p_value = float(1.0 - chi2.cdf(chi2_stat, df=1))
    return chi2_stat, p_value


def paired_bootstrap_test_regression(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, n_bootstraps=2000
) -> tuple[float, float, float]:
    """Paired bootstrap test for difference in RMSE between two regressors."""
    np.random.seed(42)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        rmse_a = np.sqrt(np.mean((y_true[idx] - y_pred_a[idx]) ** 2))
        rmse_b = np.sqrt(np.mean((y_true[idx] - y_pred_b[idx]) ** 2))
        diffs.append(rmse_a - rmse_b)

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    mean_diff = float(np.mean(diffs))
    return mean_diff, ci_lower, ci_upper


def demo_statistical_significance():
    print("\n" + "=" * 60)
    print("DEMO 8: STATISTICAL SIGNIFICANCE FOR MODEL COMPARISON")
    print("=" * 60)

    # 1. McNemar's Test for Classifiers
    y_test = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0] * 20)
    # Model A: standard
    pred_a = y_test.copy()
    pred_a[:15] = 1 - pred_a[:15]  # inject 15 errors
    # Model B: improved
    pred_b = y_test.copy()
    pred_b[:5] = 1 - pred_b[:5]   # inject only 5 errors

    stat, p_val = mcnemar_test(y_test, pred_a, pred_b)
    print(f"McNemar's Test Chi2: {stat:.4f} | p-value: {p_val:.6f}")
    if p_val < 0.05:
        print("Result: Statistically significant difference between Model A and Model B (p < 0.05)!")
    else:
        print("Result: Difference is NOT statistically significant.")

    # 2. Paired Bootstrap for Regressors
    np.random.seed(42)
    y_true_reg = np.random.normal(50, 10, size=300)
    pred_reg_a = y_true_reg + np.random.normal(0, 4.0, size=300)
    pred_reg_b = y_true_reg + np.random.normal(0, 3.2, size=300)

    mean_diff, ci_l, ci_u = paired_bootstrap_test_regression(y_true_reg, pred_reg_a, pred_reg_b)
    print(f"\nPaired Bootstrap RMSE Difference: Mean delta = {mean_diff:.4f}")
    print(f"95% Bootstrap Confidence Interval: [{ci_l:.4f}, {ci_u:.4f}]")
    if ci_l > 0 or ci_u < 0:
        print("Result: 95% CI does not overlap 0 -> Significant performance divergence!")
    else:
        print("Result: 95% CI overlaps 0 -> Insufficient evidence of divergence.")


if __name__ == "__main__":
    demo_split_discipline()
    demo_nested_cross_validation()
    demo_imbalance_metrics_roc_vs_pr_auc()
    demo_regression_metrics_outlier_stress()
    demo_probability_calibration()
    demo_hyperparameter_search()
    demo_learning_curves()
    demo_statistical_significance()
    print("\n[SUCCESS] Module 5 execution finished successfully.")
