"""
M1 - Data ingestion.

Stage in the lifecycle:  raw data -> validated raw dataframe

This is intentionally the *only* place that touches the raw CSV. Every other
stage downstream reads from data/processed/, never from data/raw/, so the
raw source of truth is never silently mutated in place.
"""
from pathlib import Path
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"

EXPECTED_COLUMNS = ["Fruit", "Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)", "Class"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and run a light schema check.

    A schema check here is what catches an upstream data-source change
    (renamed column, dropped column, new column) before it silently
    breaks a training run three stages downstream.
    """
    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Schema check failed. Missing columns: {missing}")

    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        print(f"[data_ingestion] Note: unexpected extra columns found: {extra}")

    print(f"[data_ingestion] Loaded {len(df)} rows, {df.shape[1]} columns from {path}")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print(df.dtypes)
