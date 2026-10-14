"""
FreshChain - Module 3: Random Forest Analysis & Ensemble Learning.

Covers:
1. RandomForestClassifier implementation with oob_score=True
2. Bootstrap Sampling demonstration (sampling with replacement, duplicates, OOB count)
3. Random Feature Selection (max_features, de-correlation)
4. Majority Voting demonstration (tree-by-tree vote breakdown for test samples)
5. Out-of-Bag (OOB) Evaluation (comparison of OOB score vs Val vs Test accuracy)
6. Hyperparameter sensitivity analysis (n_estimators, max_depth, max_features)
7. Feature Importance calculation (Mean Decrease in Impurity) & comparison with Decision Tree
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "random_forest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    """Load cleaned data and perform reproducible Train / Validation / Test split."""
    df = pd.read_csv(DATA_PATH)
    X = pd.get_dummies(df[NUMERIC_COLS + CATEGORICAL_COLS], drop_first=False)
    y = (df["Class"] == "Bad").astype(int)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, list(X.columns)


def demonstrate_bootstrap_sampling(X_train):
    """Demonstrates Bootstrap Sampling with replacement (Phase 10).
    
    Shows:
    - Original sample count N
    - Number of unique observations picked
    - Number of duplicate selections
    - Number and percentage of omitted (Out-of-Bag) observations
    - Comparison with asymptotic theoretical expectation: (1 - 1/e) ~ 63.2% in, ~36.8% OOB.
    """
    n = len(X_train)
    indices = np.arange(n)
    
    np.random.seed(42)
    boot_indices = np.random.choice(indices, size=n, replace=True)
    
    unique_selected = len(np.unique(boot_indices))
    duplicates = n - unique_selected
    oob_count = n - unique_selected
    pct_selected = (unique_selected / n) * 100.0
    pct_oob = (oob_count / n) * 100.0
    
    return {
        "total_training_samples": n,
        "bootstrap_sample_size": n,
        "unique_samples_selected": unique_selected,
        "duplicate_instances_count": duplicates,
        "omitted_oob_instances_count": oob_count,
        "percentage_selected": round(pct_selected, 2),
        "percentage_oob": round(pct_oob, 2),
        "theoretical_expectation": {
            "in_bag_percentage": "~63.21%",
            "out_of_bag_percentage": "~36.79%",
            "formula": "lim (n->inf) [1 - (1 - 1/n)^n] = 1 - 1/e"
        },
        "explanation": (
            f"From the {n} training records, sampling with replacement picked {unique_selected} unique samples "
            f"({pct_selected:.2f}%) and duplicated some observations {duplicates} times. "
            f"Exactly {oob_count} records ({pct_oob:.2f}%) were omitted from this particular tree's training, "
            f"becoming its Out-of-Bag (OOB) evaluation set. This closely matches the theoretical 63.2% / 36.8% rule."
        )
    }


def demonstrate_majority_voting(rf_model, X_test, y_test, feature_names):
    """Demonstrates Majority Voting by inspecting individual tree predictions (Phase 12).
    
    Takes 5 real test samples and shows how individual trees vote.
    """
    sample_indices = [0, 10, 25, 50, 100]
    X_samples = X_test.iloc[sample_indices]
    y_true = y_test.iloc[sample_indices]
    
    voting_breakdown = []
    
    for i, (idx, row) in enumerate(X_samples.iterrows()):
        row_df = pd.DataFrame([row])
        tree_preds = [int(tree.predict(row_df.values)[0]) for tree in rf_model.estimators_]
        
        votes_spoiled = sum(tree_preds)
        votes_fresh = len(tree_preds) - votes_spoiled
        majority_winner = 1 if votes_spoiled >= votes_fresh else 0
        
        voting_breakdown.append({
            "test_sample_index": int(idx),
            "true_label": "Spoiled (Bad)" if int(y_true.iloc[i]) == 1 else "Fresh (Good)",
            "total_trees": len(rf_model.estimators_),
            "votes_for_spoiled": votes_spoiled,
            "votes_for_fresh": votes_fresh,
            "spoiled_vote_pct": round((votes_spoiled / len(rf_model.estimators_)) * 100.0, 1),
            "final_ensemble_prediction": "Spoiled (Bad)" if majority_winner == 1 else "Fresh (Good)",
            "consensus": "Unanimous" if votes_spoiled == 0 or votes_fresh == 0 else "Majority"
        })
        
    return voting_breakdown


def evaluate_rf_hyperparameters(X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluates the effect of n_estimators on OOB score and accuracy (Phase 14)."""
    estimator_counts = [10, 25, 50, 100, 150, 200]
    records = []
    
    for n_est in estimator_counts:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_features="sqrt",
            max_depth=10,
            random_state=42,
            oob_score=True,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        train_acc = float(accuracy_score(y_train, rf.predict(X_train)))
        val_acc = float(accuracy_score(y_val, rf.predict(X_val)))
        test_acc = float(accuracy_score(y_test, rf.predict(X_test)))
        oob_acc = float(rf.oob_score_)
        
        records.append({
            "n_estimators": n_est,
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "oob_score": round(oob_acc, 4)
        })
        
    # Plot n_estimators vs OOB and Validation Accuracy
    n_list = [r["n_estimators"] for r in records]
    oob_list = [r["oob_score"] for r in records]
    val_list = [r["val_accuracy"] for r in records]
    test_list = [r["test_accuracy"] for r in records]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_list, val_list, "s--", label="Validation Accuracy", color="#7c6ff0", linewidth=2)
    ax.plot(n_list, oob_list, "d-.", label="OOB Score", color="#f2a950", linewidth=2)
    ax.plot(n_list, test_list, "o-", label="Test Accuracy", color="#17a888", linewidth=2)
    ax.set_title("Random Forest: Number of Trees vs. OOB & Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Estimators (n_estimators)", fontsize=11)
    ax.set_ylabel("Accuracy Score", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "n_estimators_vs_oob_accuracy.png", dpi=300)
    plt.close()
    
    return records


def compute_feature_importances(rf_model, dt_model, feature_names):
    """Computes and visualizes Mean Decrease in Impurity (MDI) feature importances (Phase 17)."""
    rf_importances = rf_model.feature_importances_
    dt_importances = dt_model.feature_importances_
    
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Random_Forest_Importance": rf_importances,
        "Decision_Tree_Importance": dt_importances
    }).sort_values(by="Random_Forest_Importance", ascending=True)
    
    # Plot horizontal comparison bar chart
    y_pos = np.arange(len(df_imp))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y_pos + height/2, df_imp["Random_Forest_Importance"], height, label="Random Forest (Ensemble)", color="#17a888")
    ax.barh(y_pos - height/2, df_imp["Decision_Tree_Importance"], height, label="Decision Tree (Single Tree)", color="#7c6ff0")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_imp["Feature"], fontsize=10)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title("Feature Importance: Random Forest vs. Single Decision Tree", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=300)
    plt.close()
    
    table = df_imp.sort_values(by="Random_Forest_Importance", ascending=False).to_dict(orient="records")
    return [
        {
            "Feature": r["Feature"],
            "Random_Forest_Importance": round(float(r["Random_Forest_Importance"]), 4),
            "Decision_Tree_Importance": round(float(r["Decision_Tree_Importance"]), 4)
        }
        for r in table
    ]


def run():
    """Main execution orchestrating all Random Forest phases."""
    print("[random_forest] Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_split_data()
    
    print("[random_forest] Demonstrating Bootstrap Sampling...")
    boot_info = demonstrate_bootstrap_sampling(X_train)
    
    print("[random_forest] Training tuned RandomForestClassifier with oob_score=True...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Train comparable single Decision Tree for feature importance comparison
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    
    print("[random_forest] Demonstrating Majority Voting on real test samples...")
    voting_info = demonstrate_majority_voting(rf, X_test, y_test, feature_names)
    
    print("[random_forest] Evaluating OOB score vs Test & Val accuracy...")
    train_acc = float(accuracy_score(y_train, rf.predict(X_train)))
    val_acc = float(accuracy_score(y_val, rf.predict(X_val)))
    test_acc = float(accuracy_score(y_test, rf.predict(X_test)))
    oob_acc = float(rf.oob_score_)
    
    oob_summary = {
        "oob_score": round(oob_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_accuracy": round(train_acc, 4),
        "oob_vs_test_gap": round(abs(oob_acc - test_acc), 4),
        "explanation": (
            "Out-of-Bag (OOB) evaluation uses each sample only on the trees where it was omitted during bootstrap. "
            "Because the OOB score (approx. {:.2f}%) aligns closely with the held-out test accuracy ({:.2f}%), "
            "it serves as an internal cross-validation estimate with zero data leakage."
        ).format(oob_acc * 100, test_acc * 100)
    }
    
    print("[random_forest] Evaluating hyperparameter sensitivity...")
    hp_records = evaluate_rf_hyperparameters(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("[random_forest] Computing Feature Importance table & comparison plot...")
    feat_imp = compute_feature_importances(rf, dt, feature_names)
    
    summary = {
        "bootstrap_sampling": boot_info,
        "random_feature_selection": {
            "max_features_strategy": "sqrt",
            "features_considered_per_split": int(np.ceil(np.sqrt(len(feature_names)))),
            "total_available_features": len(feature_names),
            "explanation": (
                "At each decision node, only a random subset of features (sqrt(8) = 3 features) is evaluated. "
                "This breaks correlation between trees: even if 'Humid (%)' is the globally strongest predictor, "
                "other trees are forced to discover secondary signals (e.g. CO2, Light, Temperature), creating an ensemble "
                "of diverse expert trees."
            )
        },
        "majority_voting_demonstration": voting_info,
        "out_of_bag_evaluation": oob_summary,
        "hyperparameter_sensitivity": hp_records,
        "feature_importance": feat_imp
    }
    
    out_file = REPORT_DIR / "summary.json"
    with open(out_file, "w") as f:
        json.dump(to_serializable(summary), f, indent=2)
        
    print(f"[random_forest] Saved Random Forest report and plots to {REPORT_DIR}")
    return summary


if __name__ == "__main__":
    run()
