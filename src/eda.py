"""
M1 · Session 5-8 — Data Loading and Exploratory Data Analysis.

Run this FIRST, before app.py:
    python src/eda.py

Covers, in order, exactly what the session deck walks through:
  1. Data loading with Pandas
  2. Understanding dataset structure (head, tail, shape, columns, dtypes, describe)
  3. Data quality check (missing values, duplicates)
  4. Target variable analysis (Class distribution)
  5. Univariate analysis (histograms for numeric, count plot for categorical)
  6. Bivariate analysis (numeric-vs-target, categorical-vs-target)
  7. Multivariate analysis (correlation heatmap)

All outputs are written to reports/eda/ — app.py reads them from there to
render the dashboard's EDA tab, so this script must be run at least once
before starting the dashboard.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
EDA_DIR = PROJECT_ROOT / "reports" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
CATEGORICAL_COLS = ["Fruit"]
TARGET_COL = "Class"


# ---------------------------------------------------------------------------
# 1. Data loading (Session 5)
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    print(f"[eda] Loaded {len(df)} rows, {df.shape[1]} columns from {RAW_PATH}")
    # Label casing cleanup ("Bad" vs "BAD") — the one real data-quality fix
    # needed for this dataset, done here so every EDA plot below reflects
    # the true two-class picture.
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.title()
    df = df[df[TARGET_COL].isin(["Good", "Bad"])].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Dataset structure (Session 5-6)
# ---------------------------------------------------------------------------
def dataset_structure(df: pd.DataFrame) -> dict:
    info = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.head(5).to_dict(orient="records"),
        "tail": df.tail(5).to_dict(orient="records"),
        "describe": df[NUMERIC_COLS].describe().round(3).to_dict(),
    }
    print(f"[eda] Shape: {info['shape']}")
    print(f"[eda] Columns: {info['column_names']}")
    return info


# ---------------------------------------------------------------------------
# 3. Data quality check (Session 6)
# ---------------------------------------------------------------------------
def data_quality(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    n_dupes = int(df.duplicated().sum())
    report = {
        "missing_values": {c: {"count": int(missing[c]), "pct": float(missing_pct[c])} for c in df.columns},
        "duplicate_rows": n_dupes,
        "duplicate_pct": round(n_dupes / len(df) * 100, 2),
    }
    print(f"[eda] Duplicate rows: {n_dupes} ({report['duplicate_pct']}%)")
    return report


# ---------------------------------------------------------------------------
# 4. Target variable analysis (Session 6)
# ---------------------------------------------------------------------------
def target_variable_analysis(df: pd.DataFrame) -> dict:
    counts = df[TARGET_COL].value_counts()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(counts.index, counts.values, color=["#17a888", "#ef6f6c"])
    ax.set_title("Target variable (Class) distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "target_distribution.png", dpi=120)
    plt.close(fig)
    return counts.to_dict()


# ---------------------------------------------------------------------------
# 5. Univariate analysis (Session 6, 10)
# ---------------------------------------------------------------------------
def univariate_analysis(df: pd.DataFrame):
    # Histograms for numeric features
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        ax.hist(df[col], bins=30, color="#7c6ff0", alpha=0.85)
        ax.set_title(col)
    fig.suptitle("Univariate analysis — numeric feature histograms")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "univariate_histograms.png", dpi=120)
    plt.close(fig)

    # Box plots for outlier detection
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.boxplot(y=df[col], ax=ax, color="#f2a950")
        ax.set_title(col)
    fig.suptitle("Univariate analysis — box plots (outlier detection)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "univariate_boxplots.png", dpi=120)
    plt.close(fig)

    # Count plot for the categorical feature
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Fruit", ax=ax, color="#17a888")
    ax.set_title("Univariate analysis — Fruit count plot")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "univariate_countplot.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Bivariate analysis (Session 6, 11)
# ---------------------------------------------------------------------------
def bivariate_analysis(df: pd.DataFrame):
    # Numeric feature vs target (box plots split by class)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        sns.boxplot(data=df, x=TARGET_COL, y=col, ax=ax,
                    palette={"Good": "#17a888", "Bad": "#ef6f6c"})
        ax.set_title(col)
    fig.suptitle("Bivariate analysis — numeric features vs Class")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "bivariate_numeric_vs_target.png", dpi=120)
    plt.close(fig)

    # Categorical feature vs target (count plot, hue=Class)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=df, x="Fruit", hue=TARGET_COL, ax=ax,
                  palette={"Good": "#17a888", "Bad": "#ef6f6c"})
    ax.set_title("Bivariate analysis — Fruit vs Class")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "bivariate_fruit_vs_target.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Multivariate analysis (Session 6, 12)
# ---------------------------------------------------------------------------
def multivariate_analysis(df: pd.DataFrame) -> dict:
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Multivariate analysis — correlation matrix")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "multivariate_correlation.png", dpi=120)
    plt.close(fig)

    # Pairwise scatter matrix coloured by class — the classic multivariate view
    plot_df = df[NUMERIC_COLS + [TARGET_COL]].copy()
    g = sns.pairplot(plot_df, hue=TARGET_COL, palette={"Good": "#17a888", "Bad": "#ef6f6c"},
                      plot_kws={"alpha": 0.4, "s": 12})
    g.fig.suptitle("Multivariate analysis — pairwise scatter matrix", y=1.02)
    g.savefig(EDA_DIR / "multivariate_pairplot.png", dpi=110)
    plt.close(g.fig)

    return corr.round(3).to_dict()


def run() -> dict:
    df = load_data()
    structure = dataset_structure(df)
    quality = data_quality(df)
    target = target_variable_analysis(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    corr = multivariate_analysis(df)

    report = {
        "structure": structure,
        "data_quality": quality,
        "target_distribution": target,
        "correlation": corr,
    }
    with open(EDA_DIR / "summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[eda] Done. All plots + summary.json saved to {EDA_DIR}")
    print("[eda] Now run: python app.py")
    return report


if __name__ == "__main__":
    run()
