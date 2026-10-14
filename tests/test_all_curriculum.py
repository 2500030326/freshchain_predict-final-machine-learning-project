"""
Comprehensive Unit Test Suite for All 10 Authoritative Curriculum Modules (A through J)
Validates all 176 slides across the 10 PowerPoint decks in FreshPredict.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from curriculum.module_a_split_leakage import (
    run_module_a_walkthrough,
    get_example_placement_dataset,
    compute_empirical_risk
)
from curriculum.module_b_nested_cv import (
    run_module_b_walkthrough,
    compute_nested_fits_count,
    generate_placement_benchmark
)
from curriculum.module_c_metrics import (
    run_module_c_walkthrough,
    compute_f_beta,
    compute_huber_loss,
    get_case_study_data
)
from curriculum.module_d_roc_auc import (
    run_module_d_walkthrough,
    compute_threshold_metrics,
    get_ranked_placement_dataset
)
from curriculum.module_e_pr_curves import (
    run_module_e_walkthrough,
    solve_slide_11_practice_problem
)
from curriculum.module_f_calibration import (
    run_module_f_walkthrough,
    platt_scale_manual,
    compute_brier_score
)
from curriculum.module_g_hyperparam_search import (
    run_module_g_walkthrough,
    solve_search_budget_practice_problem,
    compute_expected_improvement
)
from curriculum.module_h_cluster_regression import (
    run_module_h_walkthrough,
    get_5house_dataset,
    predict_house_price
)
from curriculum.module_i_kmeans import (
    run_module_i_walkthrough,
    solve_slide_24_silhouette_hand_example
)
from curriculum.module_j_hierarchical import (
    run_module_j_walkthrough,
    solve_ward_merge_sequence
)
from curriculum.curriculum_engine import (
    get_module_report,
    handle_interactive_query
)


class TestModuleASplitLeakage(unittest.TestCase):
    """Module A: M5-1 Split Discipline, Empirical Risk, and Data Leakage"""

    def test_module_a_execution(self):
        result = run_module_a_walkthrough()
        self.assertIn("dataset", result)
        self.assertEqual(len(result["dataset"]), 10)
        self.assertEqual(len(result["dev_students"]), 8)
        self.assertEqual(len(result["test_students"]), 2)
        self.assertEqual(len(result["cv_scores_4fold"]), 4)
        self.assertGreaterEqual(result["test_accuracy"], 0.5)

    def test_empirical_risk(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])
        # 1 error out of 5 -> 0.20
        risk = compute_empirical_risk(y_true, y_pred, loss_type="0-1")
        self.assertAlmostEqual(risk, 0.20, places=3)


class TestModuleBNestedCV(unittest.TestCase):
    """Module B: M5-2 Cross-Validation Variants and Nested CV"""

    def test_nested_fits_formula(self):
        # N_fits = K_outer * (K_inner * |Lambda| + 1)
        # K_outer=5, K_inner=3, |Lambda|=10 -> 5 * (3 * 10 + 1) = 5 * 31 = 155
        fits = compute_nested_fits_count(k_outer=5, k_inner=3, num_grid_params=10)
        self.assertEqual(fits, 155)

        fits_small = compute_nested_fits_count(k_outer=5, k_inner=3, num_grid_params=4)
        self.assertEqual(fits_small, 65)

    def test_stratified_allocation(self):
        X, y = generate_placement_benchmark(n_samples=1000)
        self.assertEqual(len(X), 1000)
        self.assertEqual(int(y.sum()), 670)
        self.assertEqual(int((y == 0).sum()), 330)

    def test_module_b_walkthrough(self):
        result = run_module_b_walkthrough()
        self.assertEqual(result["n_fits_calculation"]["total_fits"], 65)
        self.assertIn("nested_f1_scores", result)
        self.assertEqual(len(result["nested_f1_scores"]), 5)
        self.assertAlmostEqual(result["slide_reported_benchmark"]["reported_mean_f1"], 0.9430, places=3)


class TestModuleCMetrics(unittest.TestCase):
    """Module C: M5-3 Classification and Regression Metrics"""

    def test_ten_student_salary_regression(self):
        df = get_case_study_data()
        self.assertEqual(len(df), 10)

        # Sum of Absolute Errors = 65, Sum of Squared Errors = 2113
        abs_errors = df["AbsError"]
        sq_errors = df["SqError"]
        self.assertAlmostEqual(abs_errors.sum(), 65.0, places=2)
        self.assertAlmostEqual(sq_errors.sum(), 2113.0, places=2)
        self.assertAlmostEqual(abs_errors.mean(), 6.5, places=2)
        self.assertAlmostEqual(np.sqrt(sq_errors.mean()), 14.536, places=2)

    def test_f_beta_computation(self):
        p = 0.8333
        r = 1.0
        f1 = compute_f_beta(p, r, beta=1.0)
        expected_f1 = 2 * (p * r) / (p + r)
        self.assertAlmostEqual(f1, expected_f1, places=3)

        f2 = compute_f_beta(p, r, beta=2.0)
        self.assertGreater(f2, f1)

        f05 = compute_f_beta(p, r, beta=0.5)
        self.assertLess(f05, f1)

    def test_huber_loss(self):
        delta = 1.345
        e_small = np.array([1.0])
        self.assertAlmostEqual(compute_huber_loss(e_small, delta)[0], 0.5 * (1.0 ** 2), places=4)

        e_large = np.array([4.0])
        expected_large = delta * (4.0 - 0.5 * delta)
        self.assertAlmostEqual(compute_huber_loss(e_large, delta)[0], expected_large, places=4)


class TestModuleDROCAUC(unittest.TestCase):
    """Module D: M5-4 Confusion Matrix and ROC-AUC"""

    def test_ranked_10_students(self):
        df = get_ranked_placement_dataset()
        self.assertEqual(len(df), 10)
        self.assertEqual(df["TrueClass"].sum(), 5)

        # At optimal tau* = 0.72, perfect separation
        res = compute_threshold_metrics(df["TrueClass"].values, df["PredictedProb"].values, tau=0.72)
        self.assertEqual(res["TP"], 5)
        self.assertEqual(res["FP"], 0)
        self.assertEqual(res["TN"], 5)
        self.assertEqual(res["FN"], 0)
        self.assertAlmostEqual(res["TPR"], 1.0, places=4)
        self.assertAlmostEqual(res["FPR"], 0.0, places=4)
        self.assertAlmostEqual(res["Youden_J"], 1.0, places=4)
        self.assertAlmostEqual(res["DistToCorner"], 0.0, places=4)

    def test_module_d_auc(self):
        res = run_module_d_walkthrough()
        self.assertEqual(res["pairwise_ranking"]["auc"], 1.0)
        self.assertEqual(res["pairwise_ranking"]["satisfied_pairs"], 25)
        self.assertEqual(res["pairwise_ranking"]["total_pairs"], 25)


class TestModuleEPRCurves(unittest.TestCase):
    """Module E: M5-5 Precision-Recall Curves & Class Imbalance"""

    def test_fpr_dilution(self):
        res = run_module_e_walkthrough()
        dilution = res["dilution_demo"]
        self.assertEqual(dilution["fpr"], 1.0)
        self.assertAlmostEqual(dilution["precision"], 8.33, places=2)

    def test_hand_pr_slide11(self):
        pr = solve_slide_11_practice_problem()
        self.assertAlmostEqual(pr["average_precision"], 0.854, places=2)
        self.assertAlmostEqual(pr["gain_over_baseline"], 0.354, places=2)


class TestModuleFCalibration(unittest.TestCase):
    """Module F: M5-6 & Placement Deck Probability Calibration"""

    def test_platt_scaling_slide7(self):
        p_raw = np.array([0.60])
        p_cal = platt_scale_manual(p_raw, a=1.773, b=0.789)[0]
        self.assertAlmostEqual(p_cal, 0.819, places=2)

    def test_brier_score_exercise(self):
        y_true = np.array([1, 0, 1, 0, 1])
        p_pred = np.array([0.8, 0.2, 0.6, 0.4, 0.7])
        bs = compute_brier_score(y_true, p_pred)
        self.assertAlmostEqual(bs, 0.098, places=2)

    def test_calibration_comparison_10_students(self):
        res = run_module_f_walkthrough()
        comp = res["brier_comparison_10_students"]
        self.assertAlmostEqual(comp["raw_brier"], 0.157, places=2)
        self.assertAlmostEqual(comp["platt_brier"], 0.143, places=2)
        self.assertAlmostEqual(comp["isotonic_brier"], 0.100, places=2)


class TestModuleGHyperparamSearch(unittest.TestCase):
    """Module G: M5-7 Hyperparameter Search Strategies"""

    def test_random_search_budget_problem(self):
        sol = solve_search_budget_practice_problem(n_lr=4, n_depths=3, k_folds=5, top_pct=0.05, confidence=0.95)
        self.assertEqual(sol["grid_model_fits"], 60)
        self.assertAlmostEqual(sol["random_60_p_hit"], 0.9536, places=3)
        self.assertEqual(sol["samples_for_95_pct_confidence"], 59)

    def test_expected_improvement(self):
        mu = np.array([0.85, 0.70])
        sigma = np.array([0.05, 0.05])
        ei = compute_expected_improvement(mu, sigma, f_best=0.80)
        self.assertGreater(ei[0], ei[1])


class TestModuleHClusterRegression(unittest.TestCase):
    """Module H: Cluster-Augmented Regression"""

    def test_slide_five_houses(self):
        df = get_5house_dataset()
        self.assertEqual(len(df), 5)

        res = run_module_h_walkthrough()
        self.assertAlmostEqual(res["step2_baseline_model"]["r2_score"], 0.039, places=2)
        self.assertAlmostEqual(res["step3_augmented_model"]["r2_score"], 1.0, places=2)

    def test_price_prediction_formula(self):
        pred_a = predict_house_price(size_sqft=1000, cluster="A")
        pred_b = predict_house_price(size_sqft=1000, cluster="B")
        self.assertAlmostEqual(pred_a["augmented_predicted_price_k"], 150.0, places=2)
        self.assertAlmostEqual(pred_b["augmented_predicted_price_k"], 300.0, places=2)


class TestModuleIKMeans(unittest.TestCase):
    """Module I: M4-1 K-Means Clustering"""

    def test_slide24_hand_silhouette(self):
        sil = solve_slide_24_silhouette_hand_example()
        self.assertAlmostEqual(sil["mean_silhouette_s_bar"], 0.805, places=2)
        self.assertEqual(len(sil["points_table"]), 5)

    def test_module_i_walkthrough(self):
        res = run_module_i_walkthrough()
        self.assertIn("elbow_and_silhouette_trace", res)
        self.assertEqual(len(res["elbow_and_silhouette_trace"]), 4)


class TestModuleJHierarchical(unittest.TestCase):
    """Module J: M4-2 Hierarchical Clustering"""

    def test_slide16_ward_linkage(self):
        res = solve_ward_merge_sequence()
        merges = res["merge_sequence"]
        self.assertEqual(len(merges), 4)

        # Merge 1: {A, B} height = 1.41
        self.assertAlmostEqual(merges[0]["height"], 1.41, places=2)
        # Merge 2: {C, D} height = 1.41
        self.assertAlmostEqual(merges[1]["height"], 1.41, places=2)
        # Merge 3: {CD, E} height = 5.72
        self.assertAlmostEqual(merges[2]["height"], 5.72, places=2)
        # Merge 4: {AB, CDE} height = 11.32
        self.assertAlmostEqual(merges[3]["height"], 11.32, places=2)


class TestCurriculumEngineAndEndpoints(unittest.TestCase):
    """Engine integration and interactive queries"""

    def test_get_all_module_reports(self):
        for mid in [f"module_{c}" for c in "abcdefghij"]:
            report = get_module_report(mid)
            self.assertIsNotNone(report)
            self.assertIsInstance(report, dict)
            self.assertGreater(len(report), 0)

    def test_interactive_queries(self):
        res_d = handle_interactive_query("module_d", {"threshold": 0.50, "cost_fp": 2.0, "cost_fn": 10.0})
        self.assertIn("ExpectedLoss", res_d)

        res_b = handle_interactive_query("module_b", {"k_outer": 5, "k_inner": 3, "grid_size": 10})
        self.assertEqual(res_b["total_model_fits"], 155)

        res_c = handle_interactive_query("module_c", {"precision": 0.8, "recall": 0.9, "beta": 1.0, "delta": 1.345, "residual": 3.0})
        self.assertIn("f_beta", res_c)
        self.assertIn("huber_loss", res_c)

        res_f = handle_interactive_query("module_f", {"raw_probability": 0.60, "param_a": 1.773, "param_b": 0.789})
        self.assertAlmostEqual(res_f["calibrated_probability"], 0.819, places=2)

        res_g = handle_interactive_query("module_g", {"n_lr": 4, "n_depths": 3, "k_folds": 5, "confidence": 0.95})
        self.assertEqual(res_g["grid_model_fits"], 60)

        res_h = handle_interactive_query("module_h", {"size": 1200, "cluster": "B"})
        self.assertIn("augmented_predicted_price_k", res_h)


if __name__ == "__main__":
    unittest.main()
