"""
M1/M2 - Data cleaning.

Stage in the lifecycle: raw dataframe -> cleaned dataset

Handles the two real issues found by inspecting this dataset:
1. The label column has inconsistent casing ("Bad" vs "BAD") which
   sklearn would otherwise treat as three separate classes.
2. Missing-value handling per M2 syllabus (deletion / imputation), applied
   generically even though this particular file has none, so the pipeline
   doesn't silently break if a future data pull does have gaps.
"""
from pathlib import Path
import pandas as pd

from data_ingestion import load_raw

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Label normalisation ---
    # "BAD" and "Bad" are the same class; normalise casing before anything
    # downstream treats them as distinct.
    df["Class"] = df["Class"].astype(str).str.strip().str.title()
    valid_classes = {"Good", "Bad"}
    bad_rows = ~df["Class"].isin(valid_classes)
    if bad_rows.any():
        print(f"[data_cleaning] Dropping {bad_rows.sum()} rows with unrecognised Class values")
        df = df[~bad_rows]

    # --- Categorical normalisation ---
    df["Fruit"] = df["Fruit"].astype(str).str.strip().str.title()

    # --- Duplicate rows ---
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        print(f"[data_cleaning] Dropped {n_before - len(df)} exact duplicate rows")

    # --- Missing values (M2: deletion / imputation) ---
    numeric_cols = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        print(f"[data_cleaning] Imputing {n_missing} missing numeric values with column medians")
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    if df["Fruit"].isna().any() or df["Class"].isna().any():
        df = df.dropna(subset=["Fruit", "Class"])

    # --- Basic sanity ranges (simple outlier flag, not removed, just logged) ---
    if (df["Humid (%)"] > 100).any() or (df["Humid (%)"] < 0).any():
        print("[data_cleaning] Warning: Humid (%) values outside [0, 100] found")

    df = df.reset_index(drop=True)
    print(f"[data_cleaning] Clean dataset: {len(df)} rows")
    return df


def run(save: bool = True) -> pd.DataFrame:
    raw = load_raw()
    cleaned = clean(raw)
    if save:
        out_path = PROCESSED_DIR / "clean.csv"
        cleaned.to_csv(out_path, index=False)
        print(f"[data_cleaning] Saved cleaned dataset to {out_path}")
    return cleaned


if __name__ == "__main__":
    df = run()
    print(df["Class"].value_counts())
    print(df["Fruit"].value_counts())
