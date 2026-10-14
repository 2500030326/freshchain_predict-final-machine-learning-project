"""
M2 · Session 11-12 — Encoding Categorical Features.

Demonstrates the four encoders from the session deck on this project's
one categorical column, "Fruit" (4 categories: Orange, Pineapple, Banana,
Tomato — nominal, no natural order):
  1. One-Hot Encoding    — the right fit here (low cardinality, nominal)
  2. Ordinal Encoding     — shown as the PITFALL case: Fruit has no order,
                             so imposing one is the mistake the deck warns
                             against, demonstrated rather than recommended.
  3. Target Encoding      — mean Good-rate per fruit, with K-fold to avoid
                             the leakage the deck specifically calls out.
  4. Embedding-based      — preview only (needs a neural net); described,
                             not trained, exactly as the deck treats it.

Run standalone with: python src/encoding.py
Outputs: reports/encoding/summary.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold

from data_cleaning import run as clean_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "encoding"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def one_hot_demo(df: pd.DataFrame) -> dict:
    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(df[["Fruit"]].head(6))
    cols = ohe.get_feature_names_out(["Fruit"])
    table = pd.DataFrame(encoded, columns=cols)
    table.insert(0, "Fruit", df["Fruit"].head(6).values)
    return {"sample_table": table.to_dict(orient="records"), "output_width": len(cols)}


def ordinal_pitfall_demo(df: pd.DataFrame) -> dict:
    """Fruit has no natural rank. Encoding it 0-3 alphabetically would tell
    a linear model that "Orange minus Banana" is a meaningful quantity —
    it isn't. This mirrors the deck's Stream/CS-ECE-IT pitfall example.
    """
    wrong_order = sorted(df["Fruit"].unique())  # alphabetical — arbitrary, not a real rank
    enc = OrdinalEncoder(categories=[wrong_order])
    sample = df[["Fruit"]].head(6)
    encoded = enc.fit_transform(sample)
    return {
        "warning": "Fruit has no true order — this encoding is shown as the PITFALL, not a recommendation.",
        "arbitrary_order_used": wrong_order,
        "sample": [{"Fruit": f, "ordinal_code": int(c[0])} for f, c in zip(sample["Fruit"], encoded)],
    }


def target_encoding_demo(df: pd.DataFrame, n_folds: int = 5, smoothing: float = 10.0) -> dict:
    """Mean Good-rate per Fruit, with smoothing (rare-category protection)
    and K-fold (leakage protection) — the production-grade version from
    the deck, not the naive full-dataset mean that leaks the label.
    """
    y = (df["Class"] == "Good").astype(int)
    global_mean = y.mean()

    # Naive version, for comparison — computed on the full dataset (leaks)
    naive_means = df.groupby("Fruit").apply(lambda g: (g["Class"] == "Good").mean()).to_dict()

    # K-fold out-of-fold target encoding (leak-free)
    oof_encoded = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        y_fold = y.iloc[train_idx]
        stats = train_fold.groupby("Fruit").apply(lambda g: (g["Class"] == "Good").mean())
        counts = train_fold["Fruit"].value_counts()
        smoothed = {}
        for fruit in df["Fruit"].unique():
            n_c = counts.get(fruit, 0)
            mean_c = stats.get(fruit, global_mean)
            smoothed[fruit] = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)
        oof_encoded.iloc[val_idx] = df.iloc[val_idx]["Fruit"].map(smoothed)

    return {
        "global_good_rate": round(float(global_mean), 4),
        "naive_means_per_fruit": {k: round(float(v), 4) for k, v in naive_means.items()},
        "leak_free_kfold_smoothed_sample": oof_encoded.head(6).round(4).tolist(),
        "note": "Naive means are computed on the full dataset and leak the label; "
                "the K-fold + smoothing version is what the final pipeline would use in production.",
    }


def embedding_preview() -> dict:
    return {
        "status": "preview only — not trained in this project",
        "idea": "Replace each Fruit category with a small learned dense vector "
                "(e.g. 2-4 dimensions), trained jointly with a neural network "
                "instead of computed by hand. Useful at much higher cardinality "
                "than this dataset's 4 fruit categories.",
    }


def encoder_comparison_table() -> dict:
    return {
        "One-Hot": {"output_width": "1 col/category", "needs_label": "No", "handles_high_cardinality": "Poorly", "leakage_risk": "None"},
        "Ordinal": {"output_width": "1 column", "needs_label": "No", "handles_high_cardinality": "Well", "leakage_risk": "None"},
        "Target": {"output_width": "1 column", "needs_label": "Yes", "handles_high_cardinality": "Well", "leakage_risk": "High if done naively; low with K-fold + smoothing"},
        "Embedding": {"output_width": "k dims (learned)", "needs_label": "Yes (trained jointly)", "handles_high_cardinality": "Best", "leakage_risk": "None"},
    }


def run() -> dict:
    df = clean_run(save=False)
    report = {
        "one_hot": one_hot_demo(df),
        "ordinal_pitfall": ordinal_pitfall_demo(df),
        "target_encoding": target_encoding_demo(df),
        "embedding_preview": embedding_preview(),
        "encoder_comparison": encoder_comparison_table(),
        "chosen_encoder_for_production_pipeline": "One-Hot — Fruit is nominal with only 4 categories, "
                                                    "comfortably low cardinality, exactly the 'good fit' case from the deck.",
    }
    with open(REPORT_DIR / "summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[encoding] Done. summary.json saved to {REPORT_DIR}")
    return report


if __name__ == "__main__":
    run()
