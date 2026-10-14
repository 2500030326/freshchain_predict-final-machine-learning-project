"""
Module D: Confusion Matrix and ROC-AUC
Reference Deck: Module_5_Session_M5-4.pptx
Authoritative requirements:
- Full 2x2 confusion matrix (Actual rows, Predicted cols)
- Expected loss formula: Total Loss = C_FP * FP + C_FN * FN
- Threshold sweeping & trajectory: tau in {>=0.90, >=0.72, >=0.50, >=0.30}
- Youden's J statistic: J(tau) = TPR(tau) - FPR(tau)
- Distance to top-left corner: d(tau) = sqrt((1 - TPR)^2 + FPR^2)
- Pairwise ranking interpretation: AUC = P(p_pos > p_neg) = U / (N_pos * N_neg)
- Exact 10-Student Placement Benchmark (Slide 9-12):
  - Positives: S03(0.95), S07(0.91), S01(0.88), S09(0.82), S05(0.72)
  - Negatives: S08(0.60), S06(0.40), S02(0.25), S10(0.20), S04(0.15)
  - Optimal cutoff tau* = 0.72 -> J = 1.0000, AUC = 1.0000 (25/25 satisfied pairs)
- Interactive threshold & cost evaluators
- Concept check quiz
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def get_ranked_placement_dataset() -> pd.DataFrame:
    """Exact 10-student dataset sorted by predicted probability from Slide 9."""
    data = {
        "StudentID": ["S03", "S07", "S01", "S09", "S05", "S08", "S06", "S02", "S10", "S04"],
        "CGPA": [9.1, 8.8, 8.5, 8.2, 8.0, 7.5, 7.0, 7.2, 6.9, 6.8],
        "CodingScore": [92, 89, 85, 80, 72, 68, 61, 65, 59, 58],
        "TrueClass": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 5 Positives, 5 Negatives
        "PredictedProb": [0.95, 0.91, 0.88, 0.82, 0.72, 0.60, 0.40, 0.25, 0.20, 0.15],
        "ScoreRank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    return pd.DataFrame(data)


def compute_threshold_metrics(y_true: np.ndarray, y_probs: np.ndarray, tau: float, c_fp=1.0, c_fn=5.0) -> dict:
    """Computes full confusion matrix and metrics for any decision threshold tau."""
    y_pred = (y_probs >= tau).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    accuracy = (tp + tn) / len(y_true)
    youden_j = tpr - fpr
    dist_corner = np.sqrt((1.0 - tpr) ** 2 + fpr ** 2)
    expected_loss = c_fp * fp + c_fn * fn

    return {
        "threshold": round(tau, 3),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "TPR": round(float(tpr), 4),
        "FPR": round(float(fpr), 4),
        "Precision": round(float(precision), 4),
        "Accuracy": round(float(accuracy), 4),
        "Youden_J": round(float(youden_j), 4),
        "DistToCorner": round(float(dist_corner), 4),
        "ExpectedLoss": round(float(expected_loss), 2)
    }


def compute_pairwise_ranking_auc(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, int, int]:
    """Pairwise Ranking evaluation: P(p_pos > p_neg) = U / (N_pos * N_neg)."""
    pos_scores = y_probs[y_true == 1]
    neg_scores = y_probs[y_true == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    total_pairs = n_pos * n_neg

    satisfied_pairs = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                satisfied_pairs += 1
            elif p == n:
                satisfied_pairs += 0.5

    auc = satisfied_pairs / total_pairs if total_pairs > 0 else 0.0
    return float(auc), int(satisfied_pairs), int(total_pairs)


def run_module_d_walkthrough() -> dict:
    df = get_ranked_placement_dataset()
    y_true = df["TrueClass"].values
    y_probs = df["PredictedProb"].values

    # 1. Slide 10 Operational Threshold Trajectory Table
    test_thresholds = [0.90, 0.72, 0.50, 0.30]
    trajectory_table = [compute_threshold_metrics(y_true, y_probs, tau) for tau in test_thresholds]

    # 2. Slide 11 ROC Trajectory Coordinates
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, y_probs)
    roc_points = []
    for f, t, th in zip(fpr_arr, tpr_arr, thresh_arr):
        roc_points.append({"FPR": round(float(f), 4), "TPR": round(float(t), 4), "threshold": round(float(th), 4)})

    # 3. Pairwise Ranking AUC
    auc_val, satisfied, total_pairs = compute_pairwise_ranking_auc(y_true, y_probs)

    # 4. Optimal Youden J
    opt_entry = min(trajectory_table, key=lambda x: x["DistToCorner"])

    return {
        "dataset": df.to_dict(orient="records"),
        "trajectory_table": trajectory_table,
        "roc_curve_coordinates": roc_points,
        "pairwise_ranking": {
            "n_pos": int(np.sum(y_true == 1)),
            "n_neg": int(np.sum(y_true == 0)),
            "total_pairs": total_pairs,
            "satisfied_pairs": satisfied,
            "auc": round(auc_val, 4),
            "explanation": "All 5 positive students have predicted probabilities strictly greater than all 5 negative students! S05 (lowest pos, p=0.72) > S08 (highest neg, p=0.60)."
        },
        "optimal_threshold": {
            "tau_opt": 0.72,
            "youden_j": 1.0000,
            "policy": "tau* = 0.72 achieves perfect separation (100% precision & recall) on this benchmark."
        },
        "quiz": [
            {"q": "What does a ROC-AUC score of 0.85 mathematically signify?", "a": "85% probability that a randomly chosen positive instance receives a higher score than a random negative instance."},
            {"q": "How does raising the decision threshold tau affect Precision and Recall?", "a": "Precision increases (fewer FP), while Recall decreases (more FN)."},
            {"q": "What point on the ROC curve maximizes Youden's J-Statistic?", "a": "The threshold tau* that maximizes vertical distance J = TPR - FPR above the diagonal."}
        ]
    }
