"""
Module E: Precision-Recall Curves and Class Imbalance
Reference Deck: Module_5_Session_M5-5.pptx
Authoritative requirements:
- FPR denominator dilution flaw in ROC-AUC
- Dilution worked example: 100 Positives, 99,000 Negatives, 90 TP, 990 FP
  -> Recall = 90%, FPR = 1.0%, but Precision = 8.33% (False alarms 11:1!)
- Baseline PR-AUC = prevalence pi = P / (P + N)
- 10-Student Placement Benchmark with 20% placement imbalance (Slide 8-9)
  -> Trajectory table for tau in {>=0.90, >=0.80, >=0.70, >=0.50, >=0.20}
  -> Optimal cutoff tau* = 0.90 (Peak F1 = 1.000) vs Default 0.50 trap (Precision = 33.3%)
- Non-monotonic sawtooth behavior and Davis-Goadrich harmonic interpolation
- Slide 11 Hand calculation practice problem: AP = 0.854 vs baseline 0.50
- Concept check quiz
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, classification_report


def get_imbalanced_placement_data() -> pd.DataFrame:
    """10-Student benchmark with 20% positive prevalence from Slide 8."""
    data = {
        "StudentID": ["S03", "S07", "S01", "S09", "S05", "S08", "S06", "S02", "S10", "S04"],
        "CGPA": [9.1, 8.8, 8.5, 8.2, 8.0, 7.5, 7.0, 7.2, 6.9, 6.8],
        "Coding": [92, 89, 85, 80, 72, 68, 61, 65, 59, 58],
        "TrueClass": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 Placed (20%), 8 Not Placed (80%)
        "PredProb": [0.95, 0.91, 0.88, 0.82, 0.72, 0.60, 0.40, 0.25, 0.20, 0.15],
        "RankOrder": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    return pd.DataFrame(data)


def compute_pr_trajectory(y_true: np.ndarray, y_probs: np.ndarray, thresholds: list[float]) -> list[dict]:
    """Evaluates precision, recall, and F1 at specific thresholds matching Slide 9."""
    res = []
    for tau in thresholds:
        y_pred = (y_probs >= tau).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        res.append({
            "threshold": tau,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Precision": round(prec * 100, 1),
            "Recall": round(rec * 100, 1),
            "F1": round(f1 * 100, 1)
        })
    return res


def solve_slide_11_practice_problem() -> dict:
    """Exact hand problem from Slide 11."""
    # Placed: 0.90, 0.80, 0.60, 0.40 (4 positive)
    # Not placed: 0.70, 0.50, 0.30, 0.20 (4 negative)
    y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    y_scores = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20])

    baseline = float(np.mean(y_true))  # 4/8 = 0.50

    # Cutoff 0.55
    pred_55 = (y_scores >= 0.55).astype(int)
    tp_55 = int(np.sum((y_true == 1) & (pred_55 == 1)))
    fp_55 = int(np.sum((y_true == 0) & (pred_55 == 1)))
    fn_55 = int(np.sum((y_true == 1) & (pred_55 == 0)))
    p_55 = tp_55 / (tp_55 + fp_55)
    r_55 = tp_55 / (tp_55 + fn_55)

    # Cutoff 0.35
    pred_35 = (y_scores >= 0.35).astype(int)
    tp_35 = int(np.sum((y_true == 1) & (pred_35 == 1)))
    fp_35 = int(np.sum((y_true == 0) & (pred_35 == 1)))
    fn_35 = int(np.sum((y_true == 1) & (pred_35 == 0)))
    p_35 = tp_35 / (tp_35 + fp_35)
    r_35 = tp_35 / (tp_35 + fn_35)

    # Rank hits: 0.90 (1/1), 0.80 (2/2), 0.60 (3/4), 0.40 (4/6)
    precisions_at_hits = [1/1, 2/2, 3/4, 4/6]
    ap = float(np.mean(precisions_at_hits))
    gain = ap - baseline

    return {
        "baseline_prevalence": baseline,
        "cutoff_0_55": {"TP": tp_55, "FP": fp_55, "FN": fn_55, "precision": round(p_55, 3), "recall": round(r_55, 3)},
        "cutoff_0_35": {"TP": tp_35, "FP": fp_35, "FN": fn_35, "precision": round(p_35, 3), "recall": round(r_35, 3)},
        "average_precision": round(ap, 3),
        "gain_over_baseline": round(gain, 3)
    }


def run_module_e_walkthrough() -> dict:
    df = get_imbalanced_placement_data()
    y_true = df["TrueClass"].values
    y_probs = df["PredProb"].values

    prevalence = float(np.mean(y_true))  # 0.20
    test_thresholds = [0.90, 0.80, 0.70, 0.50, 0.20]
    trajectory_table = compute_pr_trajectory(y_true, y_probs, test_thresholds)

    # Scikit-learn PR-AUC and AP
    precision_arr, recall_arr, thresh_arr = precision_recall_curve(y_true, y_probs)
    pr_auc = float(auc(recall_arr, precision_arr))
    ap_score = float(average_precision_score(y_true, y_probs))

    # Optimal F1 threshold
    f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)
    opt_idx = int(np.argmax(f1_scores))
    opt_tau = float(thresh_arr[opt_idx])

    # 99k Dilution Demonstration
    dilution_demo = {
        "scenario": "100 Positives, 99,000 Negatives, 90 TP, 990 FP",
        "recall": 90.0,
        "fpr": round(990 / 99000 * 100, 2),  # 1.0%
        "precision": round(90 / (90 + 990) * 100, 2),  # 8.33%
        "takeaway": "ROC curve looks stellar at 1% FPR, yet 91.7% of all flagged alerts are false alarms! PR-AUC exposes this instantly."
    }

    practice_sol = solve_slide_11_practice_problem()

    return {
        "dataset": df.to_dict(orient="records"),
        "prevalence": prevalence,
        "pr_auc": round(pr_auc, 4),
        "average_precision": round(ap_score, 4),
        "optimal_threshold": round(opt_tau, 2),
        "trajectory_table": trajectory_table,
        "dilution_demo": dilution_demo,
        "practice_problem_solution": practice_sol,
        "quiz": [
            {"q": "What is the baseline PR-AUC score for a random classifier on a dataset with 5% positive class prevalence?", "a": "0.05 (5.0%). In PR space, random guessing equals the positive class prevalence pi."},
            {"q": "Why can ROC-AUC remain deceptively high when a model produces 1,000 false alarms on a rare positive class?", "a": "Because the False Positive Rate denominator (N_neg) is massive, making FPR tiny. Precision-Recall AUC exposes this instantly because Precision = TP / (TP + FP)."},
            {"q": "When optimizing an imbalanced prediction system where False Negatives are twice as costly as False Positives, which metric should be maximized?", "a": "The F2-score (beta = 2.0), which places twice as much weight on Recall as Precision."}
        ]
    }
