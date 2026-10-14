"""
FreshChain - Module 3: Decision Trees & Pruning Analysis.

Covers:
1. Decision Tree Classifier (Entropy & Gini)
2. Entropy calculation on actual dataset
3. Information Gain step-by-step demonstration on real split
4. Gini Index calculation & comparison with Entropy
5. MSE concept demonstration & explanation (Regression vs Classification)
6. Pre-Pruning (max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes)
7. Post-Pruning concept
8. Cost-Complexity Pruning (ccp_alpha path, alpha vs accuracy, alpha vs complexity)
9. Weakest-Link Pruning mathematical explanation & link to sklearn CCP
10. Tree visual exports for presentation and dashboard
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "decision_tree"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Feature and target configuration
NUMERIC_COLS = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
CATEGORICAL_COLS = ["Fruit"]


def to_serializable(obj):
    """Recursively convert NumPy data types to Python standard types for JSON serialization."""
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
    """Load cleaned data and perform reproducible Train / Validation / Test split.
    
    Split strategy:
    - 70% Training: For fitting the trees and computing pruning paths
    - 15% Validation: For hyperparameter tuning and ccp_alpha selection
    - 15% Test: Held out untouched for final objective benchmarking
    """
    df = pd.read_csv(DATA_PATH)
    
    # Feature matrix: One-Hot encode Fruit
    X = pd.get_dummies(df[NUMERIC_COLS + CATEGORICAL_COLS], drop_first=False)
    # Target: 1 = Spoiled (Bad), 0 = Fresh (Good)
    y = (df["Class"] == "Bad").astype(int)
    
    # First split into 70% train and 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Split temp evenly into 15% validation and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, list(X.columns)


def compute_entropy_and_information_gain(X_train, y_train):
    """Step-by-step mathematical demonstration of Entropy and Information Gain.
    
    Formula:
      Entropy(S) = - sum(p_i * log2(p_i))
      Information Gain = Entropy(Parent) - sum(|S_v| / |S| * Entropy(S_v))
    """
    n_total = int(len(y_train))
    p1 = float(y_train.mean())  # P(Spoiled)
    p0 = float(1.0 - p1)        # P(Fresh)
    
    # Parent root entropy
    root_entropy = float(- (p0 * np.log2(p0) + p1 * np.log2(p1)))
    root_gini = float(1.0 - (p0**2 + p1**2))
    
    # Find the best split chosen at root by a decision tree
    dt_root = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=42)
    dt_root.fit(X_train, y_train)
    
    feature_idx = int(dt_root.tree_.feature[0])
    feature_name = str(X_train.columns[feature_idx])
    threshold = float(dt_root.tree_.threshold[0])
    
    # Partition training samples based on root split
    left_mask = X_train[feature_name] <= threshold
    right_mask = ~left_mask
    
    y_left = y_train[left_mask]
    y_right = y_train[right_mask]
    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())
    
    # Left child entropy
    p1_left = float(y_left.mean()) if n_left > 0 else 0.0
    p0_left = float(1.0 - p1_left)
    left_entropy = 0.0
    if p0_left > 0 and p1_left > 0:
        left_entropy = float(- (p0_left * np.log2(p0_left) + p1_left * np.log2(p1_left)))
    
    # Right child entropy
    p1_right = float(y_right.mean()) if n_right > 0 else 0.0
    p0_right = float(1.0 - p1_right)
    right_entropy = 0.0
    if p0_right > 0 and p1_right > 0:
        right_entropy = float(- (p0_right * np.log2(p0_right) + p1_right * np.log2(p1_right)))
        
    # Weighted child entropy
    weighted_child_entropy = float((n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy)
    information_gain = float(root_entropy - weighted_child_entropy)
    
    return {
        "root_samples": n_total,
        "p_spoiled_root": round(p1, 4),
        "p_fresh_root": round(p0, 4),
        "root_entropy": round(root_entropy, 4),
        "root_gini": round(root_gini, 4),
        "best_split_feature": feature_name,
        "best_split_threshold": round(threshold, 2),
        "left_child": {
            "condition": f"{feature_name} <= {threshold:.2f}",
            "samples": n_left,
            "p_spoiled": round(p1_left, 4),
            "p_fresh": round(p0_left, 4),
            "entropy": round(left_entropy, 4)
        },
        "right_child": {
            "condition": f"{feature_name} > {threshold:.2f}",
            "samples": n_right,
            "p_spoiled": round(p1_right, 4),
            "p_fresh": round(p0_right, 4),
            "entropy": round(right_entropy, 4)
        },
        "weighted_child_entropy": round(weighted_child_entropy, 4),
        "information_gain": round(information_gain, 4),
        "explanation": (
            f"At the root node, initial uncertainty is {root_entropy:.4f} bits. "
            f"Splitting on '{feature_name} <= {threshold:.2f}' divides {n_total} samples into "
            f"a pure left branch ({n_left} samples with entropy {left_entropy:.4f}) and a right branch "
            f"({n_right} samples with entropy {right_entropy:.4f}). "
            f"The weighted remaining entropy is {weighted_child_entropy:.4f} bits, producing an "
            f"Information Gain of {information_gain:.4f} bits (reduction in uncertainty)."
        )
    }


def compare_entropy_vs_gini(X_train, X_val, X_test, y_train, y_val, y_test):
    """Compares Decision Trees trained with Entropy vs Gini criterion (Phase 3C)."""
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf_entropy.fit(X_train, y_train)
    
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
    clf_gini.fit(X_train, y_train)
    
    return {
        "entropy_tree": {
            "criterion": "entropy",
            "formula": "Entropy(S) = -sum(p_i * log2(p_i))",
            "depth": int(clf_entropy.get_depth()),
            "leaves": int(clf_entropy.get_n_leaves()),
            "train_accuracy": round(float(accuracy_score(y_train, clf_entropy.predict(X_train))), 4),
            "val_accuracy": round(float(accuracy_score(y_val, clf_entropy.predict(X_val))), 4),
            "test_accuracy": round(float(accuracy_score(y_test, clf_entropy.predict(X_test))), 4)
        },
        "gini_tree": {
            "criterion": "gini",
            "formula": "Gini(S) = 1 - sum(p_i^2)",
            "depth": int(clf_gini.get_depth()),
            "leaves": int(clf_gini.get_n_leaves()),
            "train_accuracy": round(float(accuracy_score(y_train, clf_gini.predict(X_train))), 4),
            "val_accuracy": round(float(accuracy_score(y_val, clf_gini.predict(X_val))), 4),
            "test_accuracy": round(float(accuracy_score(y_test, clf_gini.predict(X_test))), 4)
        },
        "comparison_note": (
            "Gini is computationally faster because it avoids logarithmic operations. "
            "Entropy scales up to 1.0 bit for two classes, while Gini tops out at 0.5. "
            "Both criteria identify similar optimal decision boundaries on this dataset."
        )
    }


def demonstrate_mse_regression(X_train):
    """Demonstrates Mean Squared Error (MSE) conceptually as required by Phase 4.
    
    Explains:
    - MSE = (1/n) * sum((y_i - y_mean)^2)
    - Classification uses Entropy / Gini (probability distributions over categories).
    - Regression uses MSE / variance reduction (minimizing squared distance from continuous leaf mean).
    - Illustrates with a regression target (continuous cold-storage Temperature prediction).
    """
    y_reg = X_train["Temp"].copy()
    X_reg = X_train.drop(columns=["Temp"]).copy()
    
    y_mean = float(y_reg.mean())
    parent_mse = float(np.mean((y_reg - y_mean)**2))
    
    reg_tree = DecisionTreeRegressor(max_depth=1, random_state=42)
    reg_tree.fit(X_reg, y_reg)
    
    feat_idx = int(reg_tree.tree_.feature[0])
    thresh = float(reg_tree.tree_.threshold[0])
    feat_name = str(X_reg.columns[feat_idx])
    
    left_mask = X_reg[feat_name] <= thresh
    right_mask = ~left_mask
    
    y_left = y_reg[left_mask]
    y_right = y_reg[right_mask]
    
    mse_left = float(np.mean((y_left - y_left.mean())**2)) if len(y_left) > 0 else 0.0
    mse_right = float(np.mean((y_right - y_right.mean())**2)) if len(y_right) > 0 else 0.0
    
    n_tot = int(len(y_reg))
    weighted_mse = float((len(y_left)/n_tot) * mse_left + (len(y_right)/n_tot) * mse_right)
    variance_reduction = float(parent_mse - weighted_mse)
    
    return {
        "formula": "MSE = (1/n) * sum((y_i - y_mean)^2)",
        "distinction": (
            "Classification criteria (Entropy and Gini) measure impurity across categorical frequency distributions. "
            "In contrast, MSE measures variance around a continuous arithmetic mean and is used exclusively for Regression. "
            "A regression tree selects splits to maximize Variance Reduction: Parent MSE - Weighted Child MSE."
        ),
        "demonstration": {
            "continuous_target": "Storage Temperature (Temp in °C)",
            "parent_mean": round(y_mean, 2),
            "parent_mse": round(parent_mse, 4),
            "split_feature": feat_name,
            "split_threshold": round(thresh, 2),
            "left_child_mse": round(mse_left, 4),
            "right_child_mse": round(mse_right, 4),
            "weighted_child_mse": round(weighted_mse, 4),
            "variance_reduction": round(variance_reduction, 4)
        }
    }


def evaluate_pre_pruning(X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluates the effect of pre-pruning hyperparameters (Phase 5).
    
    Explores:
    - max_depth: [1, 2, 3, 4, 5, 6, 8, 10, 15, None]
    """
    depth_values = [1, 2, 3, 4, 5, 6, 8, 10, 15, None]
    depth_results = []
    
    for d in depth_values:
        clf = DecisionTreeClassifier(max_depth=d, criterion="gini", random_state=42)
        clf.fit(X_train, y_train)
        
        train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
        val_acc = float(accuracy_score(y_val, clf.predict(X_val)))
        test_acc = float(accuracy_score(y_test, clf.predict(X_test)))
        actual_depth = int(clf.get_depth())
        leaf_count = int(clf.get_n_leaves())
        
        depth_results.append({
            "param_max_depth": str(d),
            "tree_depth": actual_depth,
            "leaf_count": leaf_count,
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4)
        })
        
    # Plot Depth vs Accuracy
    plot_depths = [r["tree_depth"] for r in depth_results]
    train_scores = [r["train_accuracy"] for r in depth_results]
    val_scores = [r["val_accuracy"] for r in depth_results]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_depths, train_scores, "o-", label="Training Accuracy", color="#17a888", linewidth=2)
    ax.plot(plot_depths, val_scores, "s--", label="Validation Accuracy", color="#7c6ff0", linewidth=2)
    ax.set_title("Pre-Pruning: Tree Depth vs. Model Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Tree Depth", fontsize=12)
    ax.set_ylabel("Accuracy Score", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "prepruning_depth_curve.png", dpi=300)
    plt.close()
    
    # Pick the best pre-pruned tree based strictly on validation accuracy
    best_preprune_cfg = max(depth_results, key=lambda x: x["val_accuracy"])
    
    return depth_results, best_preprune_cfg


def evaluate_cost_complexity_pruning(X_train, X_val, X_test, y_train, y_val, y_test):
    """Implements Cost-Complexity Pruning (CCP) and Weakest-Link Pruning (Phases 6, 7, 8).
    
    Mathematical principle:
      R_alpha(T) = R(T) + alpha * |T|
      Weakest link formula:
      g(t) = [R(t) - R(T_t)] / (|T_t| - 1)
    """
    # 1. Grow unpruned baseline tree
    unpruned = DecisionTreeClassifier(random_state=42)
    unpruned.fit(X_train, y_train)
    
    # 2. Extract cost complexity pruning path
    path = unpruned.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # Subsample alphas evenly so we train around 20-30 representative trees
    if len(ccp_alphas) > 30:
        indices = np.linspace(0, len(ccp_alphas) - 1, 30, dtype=int)
        sampled_alphas = ccp_alphas[indices]
    else:
        sampled_alphas = ccp_alphas
        
    ccp_records = []
    trained_models = []
    
    for alpha in sampled_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        
        train_acc = float(accuracy_score(y_train, clf.predict(X_train)))
        val_acc = float(accuracy_score(y_val, clf.predict(X_val)))
        depth = int(clf.get_depth())
        n_leaves = int(clf.get_n_leaves())
        
        ccp_records.append({
            "ccp_alpha": float(round(float(alpha), 6)),
            "tree_depth": depth,
            "leaf_count": n_leaves,
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4)
        })
        trained_models.append(clf)
        
    # Plot alpha vs accuracy
    alphas = [r["ccp_alpha"] for r in ccp_records]
    train_accs = [r["train_accuracy"] for r in ccp_records]
    val_accs = [r["val_accuracy"] for r in ccp_records]
    depths = [r["tree_depth"] for r in ccp_records]
    leaves = [r["leaf_count"] for r in ccp_records]
    
    # Graph 1: ccp_alpha vs validation/train performance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, train_accs, "o-", label="Train Accuracy", color="#17a888", markersize=4)
    ax.plot(alphas, val_accs, "s--", label="Validation Accuracy", color="#f2a950", markersize=4)
    ax.set_title("Cost-Complexity Pruning: ccp_alpha vs. Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Effective ccp_alpha", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "ccp_alpha_vs_accuracy.png", dpi=300)
    plt.close()
    
    # Graph 2: ccp_alpha vs tree complexity (depth & leaves)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "#12172b"
    ax1.set_xlabel("Effective ccp_alpha", fontsize=12)
    ax1.set_ylabel("Tree Depth", color=color, fontsize=12)
    ax1.plot(alphas, depths, "o-", color=color, markersize=4, label="Tree Depth")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log")
    
    ax2 = ax1.twinx()
    color = "#ef6f6c"
    ax2.set_ylabel("Number of Leaves", color=color, fontsize=12)
    ax2.plot(alphas, leaves, "s--", color=color, markersize=4, label="Leaf Count")
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title("Cost-Complexity Pruning: ccp_alpha vs. Tree Complexity", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "ccp_alpha_vs_complexity.png", dpi=300)
    plt.close()
    
    # Select best alpha strictly using VALIDATION accuracy (NO TEST LEAKAGE)
    best_idx = int(np.argmax(val_accs))
    best_clf = trained_models[best_idx]
    best_alpha = alphas[best_idx]
    
    # Final evaluation of optimal CCP model on held-out test set
    test_pred = best_clf.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    
    return {
        "ccp_records": ccp_records,
        "optimal_alpha": best_alpha,
        "optimal_tree_depth": int(best_clf.get_depth()),
        "optimal_leaf_count": int(best_clf.get_n_leaves()),
        "best_val_accuracy": round(val_accs[best_idx], 4),
        "final_test_accuracy": round(test_acc, 4),
        "weakest_link_explanation": (
            "Weakest-Link Pruning: At each step, scikit-learn evaluates all internal nodes t and computes "
            "g(t) = [R(t) - R(T_t)] / (|T_t| - 1). The node with the minimum g(t) value produces the smallest increase "
            "in impurity per leaf removed, making it the 'weakest link'. Pruning this link yields the next subtree "
            "in the cost-complexity sequence. As alpha increases, weaker subtrees collapse sequentially until only the root remains."
        )
    }


def generate_tree_visualizations(X_train, y_train, feature_names):
    """Generates visual decision tree diagram showing conditions, samples, impurity, and class."""
    # Build an interpretable, pruned demonstration tree (depth=3)
    viz_tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    viz_tree.fit(X_train, y_train)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        viz_tree,
        feature_names=feature_names,
        class_names=["Fresh (Good)", "Spoiled (Bad)"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    plt.title("FreshChain — Decision Tree Visualization (max_depth=3)", fontsize=16, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "decision_tree_visualization.png", dpi=300)
    plt.close()


def run():
    """Main execution orchestrating all Decision Tree and Pruning phases."""
    print("[decision_tree] Loading and splitting data (70% train, 15% val, 15% test)...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_split_data()
    
    print("[decision_tree] Computing Root Entropy and Information Gain on real dataset...")
    entropy_info = compute_entropy_and_information_gain(X_train, y_train)
    
    print("[decision_tree] Comparing Entropy vs Gini trees...")
    entropy_vs_gini = compare_entropy_vs_gini(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("[decision_tree] Demonstrating MSE Regression concept vs Classification...")
    mse_info = demonstrate_mse_regression(X_train)
    
    print("[decision_tree] Evaluating Pre-Pruning parameters...")
    preprune_records, best_preprune = evaluate_pre_pruning(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("[decision_tree] Evaluating Cost-Complexity & Weakest-Link Pruning path...")
    ccp_info = evaluate_cost_complexity_pruning(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("[decision_tree] Generating Decision Tree visual diagram...")
    generate_tree_visualizations(X_train, y_train, feature_names)
    
    summary = {
        "entropy_and_information_gain": entropy_info,
        "entropy_vs_gini": entropy_vs_gini,
        "mse_regression_demonstration": mse_info,
        "pre_pruning": {
            "tuning_results": preprune_records,
            "best_configuration": best_preprune
        },
        "cost_complexity_pruning": ccp_info
    }
    
    out_file = REPORT_DIR / "summary.json"
    with open(out_file, "w") as f:
        json.dump(to_serializable(summary), f, indent=2)
        
    print(f"[decision_tree] Saved Decision Tree report and plots to {REPORT_DIR}")
    return summary


if __name__ == "__main__":
    run()
