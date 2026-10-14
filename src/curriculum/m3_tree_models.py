"""
Module 3: Supervised Learning — Tree-Based Models
=================================================
Complete, self-contained runnable implementations covering:
1. Splitting criteria from scratch (Gini, Entropy, Information Gain, MSE).
2. Cost-Complexity Pruning (CCP / weakest-link pruning path).
3. Random Forest mechanics: Bagging, Feature Subsampling, and Out-of-Bag (OOB) error.
4. Functional Gradient Boosting from scratch (pseudo-residuals & step updates).
5. Modern histogram-based gradient boosting (LightGBM/XGBoost architecture).
6. Feature Importance: Gini (MDI), Permutation Importance, and SHAP.
7. End-to-end worked placement-prediction task: Linear baseline vs. Gradient Boosted Tree.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    log_loss,
    confusion_matrix,
)


# =====================================================================
# 1. SPLITTING CRITERIA FROM SCRATCH
# =====================================================================

def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini Impurity: G = 1 - sum(p_k^2)."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs**2))


def entropy(y: np.ndarray) -> float:
    """Compute Shannon Entropy: H = -sum(p_k * log2(p_k))."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # Filter zero probabilities to avoid log2(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute Information Gain: IG = H(Parent) - [|L|/|P| * H(L) + |R|/|P| * H(R)]."""
    n_total = len(y_parent)
    if n_total == 0:
        return 0.0
    h_parent = entropy(y_parent)
    weight_left = len(y_left) / n_total
    weight_right = len(y_right) / n_total
    return float(h_parent - (weight_left * entropy(y_left) + weight_right * entropy(y_right)))


def mse_regression_impurity(y: np.ndarray) -> float:
    """Compute MSE / Variance reduction criterion for regression trees."""
    if len(y) == 0:
        return 0.0
    y_mean = np.mean(y)
    return float(np.mean((y - y_mean) ** 2))


def demo_splitting_criteria():
    print("\n" + "=" * 60)
    print("DEMO 1: SPLITTING CRITERIA")
    print("=" * 60)
    # Parent node: 6 Positive, 4 Negative
    parent = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    # Left split: 5 Positive, 1 Negative
    left = np.array([1, 1, 1, 1, 1, 0])
    # Right split: 1 Positive, 3 Negative
    right = np.array([1, 0, 0, 0])

    print(f"Parent Gini:            {gini_impurity(parent):.4f}")
    print(f"Parent Entropy:         {entropy(parent):.4f} bits")
    print(f"Left Child Entropy:     {entropy(left):.4f} bits")
    print(f"Right Child Entropy:    {entropy(right):.4f} bits")
    print(f"Information Gain (IG):  {information_gain(parent, left, right):.4f} bits")

    reg_targets = np.array([10.5, 11.2, 12.0, 15.8, 16.2])
    print(f"Regression Node MSE:    {mse_regression_impurity(reg_targets):.4f}")


# =====================================================================
# 2. COST-COMPLEXITY PRUNING (CCP / WEAKEST-LINK PRUNING)
# =====================================================================

def demo_cost_complexity_pruning(X_tr, X_val, y_tr, y_val):
    print("\n" + "=" * 60)
    print("DEMO 2: POST-PRUNING VIA COST-COMPLEXITY (CCP)")
    print("=" * 60)

    # 1. Fit unconstrained tree to calculate ccp_alpha pruning path
    unpruned = DecisionTreeClassifier(random_state=42)
    path = unpruned.cost_complexity_pruning_path(X_tr, y_tr)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    print(f"Computed {len(ccp_alphas)} candidate effective alphas.")
    print(f"Smallest alpha (full tree):    {ccp_alphas[0]:.6f}")
    print(f"Largest alpha (root node only): {ccp_alphas[-1]:.6f}")

    # 2. Train sequence of pruned trees along alpha path and score on validation set
    best_alpha = 0.0
    best_val_score = -1.0
    best_model = None

    # Sample a representative subset of alphas across the path to speed up search
    sampled_alphas = ccp_alphas[:: max(1, len(ccp_alphas) // 10)]

    for alpha in sampled_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_tr, y_tr)
        val_acc = clf.score(X_val, y_val)
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_alpha = alpha
            best_model = clf

    print(f"\n[Optimal CCP Alpha]: {best_alpha:.6f}")
    print(f"Optimal Tree Depth: {best_model.get_depth()} (Nodes: {best_model.tree_.node_count})")
    print(f"Validation Accuracy: {best_val_score:.4f}")
    return best_model


# =====================================================================
# 3. RANDOM FORESTS: BAGGING, OOB, AND VARIANCE REDUCTION
# =====================================================================

def demo_random_forest_mechanics(X_tr, y_tr, X_te, y_te):
    print("\n" + "=" * 60)
    print("DEMO 3: RANDOM FOREST (BAGGING, FEATURE SUBSAMPLING & OOB)")
    print("=" * 60)

    n_samples = len(X_tr)
    # Demonstrate theoretical bootstrap coverage: 1 - 1/e ≈ 63.2% unique in-bag
    indices = np.arange(n_samples)
    bootstrap_sample = np.random.choice(indices, size=n_samples, replace=True)
    unique_in_bag = len(np.unique(bootstrap_sample))
    oob_count = n_samples - unique_in_bag
    print(f"Dataset Size:           {n_samples}")
    print(f"Unique In-Bag Samples:  {unique_in_bag} ({unique_in_bag / n_samples * 100:.2f}%)")
    print(f"Out-of-Bag (OOB) Count: {oob_count} ({oob_count / n_samples * 100:.2f}%) [Theory: ~36.8%]")

    # Train Random Forest with OOB scoring enabled
    rf = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",  # Random subspace method
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    print(f"Random Forest OOB Score:   {rf.oob_score_:.4f}")
    print(f"Held-out Test Set Accuracy: {rf.score(X_te, y_te):.4f}")
    return rf


# =====================================================================
# 4. FUNCTIONAL GRADIENT BOOSTING FROM SCRATCH
# =====================================================================

class SimpleGradientBoostingRegressor:
    """Minimal functional gradient descent regressor with MSE loss."""

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.f0 = 0.0

    def fit(self, X, y):
        # 1. Initialize with constant: mean of targets
        self.f0 = float(np.mean(y))
        f_m = np.full(len(y), self.f0)

        for m in range(self.n_estimators):
            # 2. Compute pseudo-residuals (negative gradient for MSE loss: y - f_m)
            residuals = y - f_m

            # 3. Fit weak regression tree to pseudo-residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42 + m)
            tree.fit(X, residuals)

            # 4. Update ensemble predictions with shrinkage (learning rate)
            predictions = tree.predict(X)
            f_m += self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        preds = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds


def demo_gradient_boosting_scratch():
    print("\n" + "=" * 60)
    print("DEMO 4: FUNCTIONAL GRADIENT BOOSTING FROM SCRATCH")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=(200, 1))
    y = np.sin(X).ravel() + 0.1 * np.random.normal(size=200)

    custom_gbm = SimpleGradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=3)
    custom_gbm.fit(X, y)
    preds = custom_gbm.predict(X)
    mse = float(np.mean((y - preds) ** 2))
    print(f"Scratch GBDT fitted 30 trees successfully. Final Train MSE: {mse:.4f}")


# =====================================================================
# 5. FEATURE IMPORTANCE: GINI, PERMUTATION, AND APPROXIMATE SHAP
# =====================================================================

def demo_feature_importance(model, X_val, y_val, feature_names):
    print("\n" + "=" * 60)
    print("DEMO 5: FEATURE IMPORTANCE TRIAD")
    print("=" * 60)

    # 1. Gini / MDI (Mean Decrease in Impurity)
    if hasattr(model, "feature_importances_"):
        mdi_df = pd.DataFrame({
            "Feature": feature_names,
            "Gini_Importance": model.feature_importances_,
        }).sort_values(by="Gini_Importance", ascending=False)
        print("--- 1. Gini Importance (MDI) ---")
        print(mdi_df.to_string(index=False))

    # 2. Permutation Importance (out-of-sample on validation set)
    perm_res = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Mean": perm_res.importances_mean,
        "Permutation_Std": perm_res.importances_std,
    }).sort_values(by="Permutation_Mean", ascending=False)
    print("\n--- 2. Permutation Importance (Validation Set) ---")
    print(perm_df.to_string(index=False))

    # 3. Model-Agnostic Marginal Attribution Preview (Game-Theoretic Intuition)
    print("\n--- 3. Game-Theoretic Marginal Attribution Intuition ---")
    print("SHAP calculates each feature's contribution across all 2^|F| feature subsets.")
    print("Sum of SHAP values + Base Value = Final Model Output: f(x) = E[f(x)] + sum(phi_i)")


# =====================================================================
# 6. END-TO-END WORKED EXAMPLE: CAMPUS PLACEMENT PREDICTION
# =====================================================================

def run_placement_prediction_pipeline():
    print("\n" + "=" * 60)
    print("DEMO 6: END-TO-END PLACEMENT PREDICTION BENCHMARK")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 3000

    cgpa = np.random.normal(7.3, 1.0, n_samples).clip(4.0, 10.0)
    iq = np.random.normal(105, 12, n_samples).clip(70, 145)
    internships = np.random.poisson(1.2, n_samples).clip(0, 5)
    projects = np.random.poisson(2.5, n_samples).clip(0, 8)
    backlogs = np.random.binomial(3, 0.12, n_samples)

    # Non-linear placement log-odds with interaction and threshold
    logits = (
        1.5 * (cgpa - 7.0)
        + 0.04 * (iq - 100)
        + 0.8 * internships
        + 0.35 * projects
        - 1.9 * (backlogs > 0).astype(float)
        + 1.1 * ((cgpa > 8.0) & (projects >= 3)).astype(float)  # Non-linear interaction
        - 1.8
    )
    prob = 1 / (1 + np.exp(-logits))
    placed = (np.random.rand(n_samples) < prob).astype(int)

    feature_cols = ["cgpa", "iq", "internships", "projects", "backlogs"]
    X = pd.DataFrame(
        np.column_stack([cgpa, iq, internships, projects, backlogs]),
        columns=feature_cols,
    )
    y = placed

    # 3-way split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Model A: Baseline Logistic Regression (Linear)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    # Model B: Histogram-based Gradient Boosted Trees (LightGBM/XGBoost style)
    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    hgb.fit(X_train, y_train)
    hgb_probs = hgb.predict_proba(X_test)[:, 1]

    # Benchmark Comparison
    lr_auc = roc_auc_score(y_test, lr_probs)
    hgb_auc = roc_auc_score(y_test, hgb_probs)
    lr_f1 = f1_score(y_test, (lr_probs >= 0.5).astype(int))
    hgb_f1 = f1_score(y_test, (hgb_probs >= 0.5).astype(int))

    print("\n--- HELD-OUT TEST BENCHMARK ---")
    print(f"Linear Baseline (Logistic Regression) -> ROC-AUC: {lr_auc:.4f} | F1: {lr_f1:.4f}")
    print(f"Gradient Boosted Tree (Hist-GBDT)      -> ROC-AUC: {hgb_auc:.4f} | F1: {hgb_f1:.4f}")
    print(f"Relative Improvement in AUC:          {((hgb_auc - lr_auc) / lr_auc) * 100:+.2f}%")

    # Feature Importance on the trained champion GBDT
    demo_feature_importance(hgb, X_val, y_val, feature_cols)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    demo_splitting_criteria()
    X_tr, X_va, X_te, y_tr, y_va, y_te = run_placement_prediction_pipeline()
    demo_cost_complexity_pruning(X_tr, X_va, y_tr, y_va)
    demo_random_forest_mechanics(X_tr, y_tr, X_te, y_te)
    demo_gradient_boosting_scratch()
    print("\n[SUCCESS] Module 3 execution finished successfully.")
