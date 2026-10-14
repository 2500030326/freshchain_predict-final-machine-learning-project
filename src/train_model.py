"""
Final training and model selection pipeline for FreshChain.

Combines the choices justified in EDA, feature scaling, encoding, missing values,
and Module 3 tree models into one unified workflow:
  - Decision Tree (Entropy)
  - Decision Tree (Gini)
  - Pre-Pruned Decision Tree
  - Cost-Complexity Pruned Decision Tree
  - Random Forest (Bagged Ensemble with OOB Score)
  - Logistic Regression (Linear Baseline)

Selects the champion model based on generalization performance (ROC-AUC / F1-Score)
and serializes:
  - models/champion.joblib
  - models/registry.json

Run standalone with: python src/train_model.py
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_cleaning import run as clean_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
CATEGORICAL_COLS = ["Fruit"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])


def evaluate(pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }


def find_best_ccp_alpha(preprocessor, X_train, y_train, X_val, y_val) -> float:
    """Finds best ccp_alpha via validation set."""
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    base_tree = DecisionTreeClassifier(random_state=42)
    base_tree.fit(X_train_proc, y_train)
    path = base_tree.cost_complexity_pruning_path(X_train_proc, y_train)
    ccp_alphas = path.ccp_alphas
    
    if len(ccp_alphas) > 25:
        indices = np.linspace(0, len(ccp_alphas) - 1, 25, dtype=int)
        sampled_alphas = ccp_alphas[indices]
    else:
        sampled_alphas = ccp_alphas
        
    best_alpha = 0.0
    best_val_acc = -1.0
    for alpha in sampled_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_train_proc, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val_proc))
        if acc > best_val_acc:
            best_val_acc = acc
            best_alpha = float(alpha)
            
    return best_alpha


def run() -> dict:
    df = clean_run(save=True)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = (df["Class"] == "Good").astype(int)

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    preprocessor = build_preprocessor()
    optimal_alpha = find_best_ccp_alpha(preprocessor, X_train, y_train, X_val, y_val)

    candidates = {
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=100, max_features="sqrt", max_depth=10, random_state=42, oob_score=True, n_jobs=-1
            ))
        ]),
        "decision_tree_entropy": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(criterion="entropy", random_state=42))
        ]),
        "decision_tree_gini": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(criterion="gini", random_state=42))
        ]),
        "prepruned_decision_tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42))
        ]),
        "cost_complexity_pruned_tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(ccp_alpha=optimal_alpha, random_state=42))
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=2000))
        ]),
    }

    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(pipeline, X_test, y_test)
        if hasattr(pipeline.named_steps["classifier"], "oob_score_"):
            results[name]["oob_score"] = round(float(pipeline.named_steps["classifier"].oob_score_), 4)
        print(f"[train_model] {name}: {results[name]}")

    champion_name = max(results, key=lambda n: results[n]["roc_auc"])
    champion_pipeline = candidates[champion_name]
    joblib.dump(champion_pipeline, MODELS_DIR / "champion.joblib")

    registry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "champion_model": champion_name,
        "champion_artifact": "champion.joblib",
        "feature_columns": list(X.columns),
        "target_mapping": {"0": "Bad", "1": "Good"},
        "test_metrics_all_models": results,
    }
    with open(MODELS_DIR / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    print(f"[train_model] Champion: {champion_name} — saved to models/champion.joblib")
    return registry


if __name__ == "__main__":
    run()
