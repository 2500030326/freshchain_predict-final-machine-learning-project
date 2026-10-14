"""
Module F: Probability Calibration, Brier Score, and Log Loss
Reference Decks: Probability_Calibration_Placement.pptx & Module_5_Session_M5-6_to_M5-8.pptx
Authoritative requirements:
- Probability calibration fundamentals (trustworthy probabilities, overconfidence)
- 10-student Placement Calibration Dataset (S1-S10)
- Over-confidence demonstration: 5 students with p=0.80, actual rate 60%
- Platt Scaling: p_cal = 1 / (1 + exp(-(a*f + b))), where f = ln(p/(1-p))
  -> Exact S7 worked example: f = ln(1.5) = 0.405, a=1.773, b=0.789 => p_cal = 0.819
  -> Exact 10-student Platt calibrated array: [0.043, 0.159, 0.329, 0.424, 0.607, 0.759, 0.819, 0.908, 0.963, 0.991]
- Isotonic Regression with PAVA (Pool Adjacent Violators Algorithm)
  -> Exact 10-student Isotonic array: [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
- Brier Score: BS = (1/N) * sum((p_i - y_i)^2)
  -> 5-student hand exercise: p=[0.8, 0.2, 0.6, 0.4, 0.7], y=[1, 0, 1, 0, 1] => BS = 0.098
  -> 10-student comparison: Raw = 0.157, Platt = 0.143, Isotonic = 0.100
- Alternative parameter convention (A=0.8, B=-0.2, logit=1.0)
- Scikit-learn CalibratedClassifierCV
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def get_calibration_10student_dataset() -> pd.DataFrame:
    """Exact 10-student dataset from Slide 4."""
    data = {
        "Student": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
        "CGPA": [6.2, 6.5, 7.0, 7.1, 7.5, 7.8, 8.0, 8.3, 8.7, 9.0],
        "Aptitude": [55, 60, 65, 70, 72, 75, 80, 85, 90, 95],
        "ActualY": [0, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        "RawP": [0.10, 0.20, 0.30, 0.35, 0.45, 0.55, 0.60, 0.70, 0.80, 0.90],
    }
    return pd.DataFrame(data)


def platt_scale_manual(p_raw: np.ndarray, a=1.773, b=0.789) -> np.ndarray:
    """Platt scaling sigmoid formulation matching Slide 6."""
    eps = 1e-12
    p_clipped = np.clip(p_raw, eps, 1 - eps)
    f = np.log(p_clipped / (1 - p_clipped))
    return 1.0 / (1.0 + np.exp(-(a * f + b)))


def pava_isotonic_manual(p_raw: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """PAVA (Pool Adjacent Violators Algorithm) monotonic step mapping from Slide 8."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return ir.fit_transform(p_raw, y_true)


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Brier score: (1/N) * sum((p - y)^2)."""
    return float(np.mean((y_probs - y_true) ** 2))


def run_module_f_walkthrough() -> dict:
    df = get_calibration_10student_dataset()
    y_true = df["ActualY"].values
    p_raw = df["RawP"].values

    # 1. Platt Scaling
    p_platt = platt_scale_manual(p_raw, a=1.773, b=0.789)
    df["PlattP"] = np.round(p_platt, 3)

    # 2. Isotonic PAVA
    p_iso = pava_isotonic_manual(p_raw, y_true)
    df["IsotonicP"] = np.round(p_iso, 3)

    # 3. Brier Scores on 10 students
    bs_raw = compute_brier_score(y_true, p_raw)
    bs_platt = compute_brier_score(y_true, p_platt)
    bs_iso = compute_brier_score(y_true, p_iso)

    # 4. S7 Worked Example Step-by-Step
    s7_raw = 0.60
    s7_f = np.log(0.60 / 0.40)  # ln(1.5) = 0.405465
    s7_cal = 1.0 / (1.0 + np.exp(-(1.773 * s7_f + 0.789)))

    # 5. Slide 10: 5-Student Exercise
    ex5_p = np.array([0.8, 0.2, 0.6, 0.4, 0.7])
    ex5_y = np.array([1, 0, 1, 0, 1])
    ex5_sq_diff = (ex5_p - ex5_y) ** 2  # [0.04, 0.04, 0.16, 0.16, 0.09]
    ex5_bs = float(np.mean(ex5_sq_diff))  # 0.49 / 5 = 0.098

    # 6. Alternative parameter convention (A=0.8, B=-0.2, logit=1.0)
    alt_logit = 1.0
    alt_prob = float(1.0 / (1.0 + np.exp(-(0.8 * alt_logit - 0.2))))

    return {
        "calibration_table": df.to_dict(orient="records"),
        "s7_walkthrough": {
            "student": "S7",
            "p_raw": s7_raw,
            "log_odds_f": round(float(s7_f), 4),
            "a": 1.773, "b": 0.789,
            "calculated_p_cal": round(float(s7_cal), 3),
            "slide_reported_p_cal": 0.819
        },
        "brier_comparison_10_students": {
            "raw_brier": round(bs_raw, 3),
            "platt_brier": round(bs_platt, 3),
            "isotonic_brier": round(bs_iso, 3),
            "slide_reported": {"raw": 0.157, "platt": 0.143, "isotonic": 0.100}
        },
        "exercise_5_students": {
            "predicted_p": ex5_p.tolist(),
            "actual_y": ex5_y.tolist(),
            "sq_errors": [round(float(x), 4) for x in ex5_sq_diff],
            "sum_sq_error": round(float(np.sum(ex5_sq_diff)), 4),
            "brier_score": round(ex5_bs, 4)
        },
        "alternative_convention": {
            "A": 0.8, "B": -0.2, "logit": 1.0,
            "calibrated_prob": round(alt_prob, 4)
        },
        "overconfidence_demo": {
            "five_students_pred": 0.80,
            "actual_placed": "3 out of 5 (60%)",
            "diagnosis": "Model over-confident by 20%. Calibration maps 0.80 -> 0.60."
        },
        "quiz": [
            {"q": "What calibrator is best for N = 400 validation samples?", "a": "Platt Scaling (Sigmoid) because 2 parameters prevent overfitting."},
            {"q": "How does PAVA maintain monotonicity in Isotonic Regression?", "a": "It sorts scores and pools adjacent violators by taking their weighted average whenever an inversion (higher score mapping to lower outcome) occurs."},
            {"q": "What is the Brier Score of a constant 0.50 baseline on a 50/50 dataset?", "a": "BS = 0.25 (since (0.5 - 1)^2 = (0.5 - 0)^2 = 0.25). Any model worse than 0.25 is worse than a coin flip!"}
        ]
    }
