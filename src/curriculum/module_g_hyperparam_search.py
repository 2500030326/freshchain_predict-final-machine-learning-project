"""
Module G: Hyperparameter Search and Optimization
Reference Deck: Module_5_Session_M5-6_to_M5-8.pptx
Authoritative requirements:
- Grid Search (exhaustive lattice evaluation)
- Random Search (Bergstra-Bengio continuous sampling efficiency)
- Bayesian Optimization (Gaussian Process surrogate & Expected Improvement acquisition):
    EI(theta) = (mu(theta) - f_best) * Phi(Z) + sigma(theta) * phi(Z)
- Slide 10 Search Budget Practice Problem:
    (a) 4 learning rates, 3 tree depths, 5-fold CV -> 60 fits
    (b) 60 random configs in top 5%: P(hit) = 1 - 0.95^60 = 0.954 (95.4%)
    (c) 95% confidence sample count: n >= ln(0.05)/ln(0.95) = 58.4 -> n = 59
- Scikit-learn GridSearchCV, RandomizedSearchCV & GP Regressor
- Self-assessment quiz
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import randint, norm


def solve_search_budget_practice_problem(
    n_lr=4, n_depths=3, k_folds=5, top_pct=0.05, confidence=0.95
) -> dict:
    """Exact problem from Slide 10."""
    n_configs = n_lr * n_depths
    grid_fits = n_configs * k_folds

    # 60 configurations probability
    n_samples_60 = 60
    p_miss_60 = (1.0 - top_pct) ** n_samples_60
    p_hit_60 = 1.0 - p_miss_60

    # Samples needed for confidence
    # 1 - (1 - top_pct)^n >= confidence => (1 - top_pct)^n <= 1 - confidence
    # n * ln(1 - top_pct) <= ln(1 - confidence) => n >= ln(1 - confidence) / ln(1 - top_pct)
    n_needed = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - top_pct))

    return {
        "grid_configurations": n_configs,
        "grid_model_fits": grid_fits,
        "random_60_p_miss": round(p_miss_60, 4),
        "random_60_p_hit": round(p_hit_60, 4),
        "samples_for_95_pct_confidence": n_needed,
        "formula_derivation": "n >= ln(0.05) / ln(0.95) = -2.9957 / -0.05129 = 58.40 => 59 samples"
    }


def compute_expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    """Expected Improvement formula: (mu - f_best) * Phi(Z) + sigma * phi(Z)."""
    with np.errstate(divide='warn'):
        z = np.zeros_like(mu)
        mask = sigma > 1e-9
        z[mask] = (mu[mask] - f_best) / sigma[mask]
        ei = np.zeros_like(mu)
        ei[mask] = (mu[mask] - f_best) * norm.cdf(z[mask]) + sigma[mask] * norm.pdf(z[mask])
    return ei


def run_module_g_walkthrough() -> dict:
    # 1. Slide 10 Search Budget calculation
    budget_sol = solve_search_budget_practice_problem()

    # 2. Gaussian Process Surrogate demonstration
    np.random.seed(42)
    sampled_depths = np.array([3, 5, 10, 15, 20]).reshape(-1, 1)
    val_scores = np.array([0.72, 0.85, 0.94, 0.91, 0.88])  # peak around 10
    f_best = float(np.max(val_scores))

    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), random_state=42)
    gp.fit(sampled_depths, val_scores)

    untested_depths = np.linspace(1, 25, 25).reshape(-1, 1)
    mu, sigma = gp.predict(untested_depths, return_std=True)
    ei = compute_expected_improvement(mu, sigma, f_best)

    gp_results = []
    for d, m, s, e in zip(untested_depths.ravel(), mu, sigma, ei):
        gp_results.append({
            "depth": int(d),
            "predicted_mean_acc": round(float(m), 4),
            "uncertainty_sigma": round(float(s), 4),
            "expected_improvement": round(float(e), 5)
        })

    next_best_depth = int(untested_depths[np.argmax(ei)][0])

    return {
        "practice_problem_solution": budget_sol,
        "bayesian_optimization_gp": {
            "sampled_depths": [3, 5, 10, 15, 20],
            "validation_scores": [0.72, 0.85, 0.94, 0.91, 0.88],
            "f_best": f_best,
            "next_recommended_depth_to_evaluate": next_best_depth,
            "gp_predictions_table": gp_results[:10]  # top 10 preview
        },
        "comparison_matrix": {
            "Grid Search": {"Budget": "Exhaustive Lattice", "Dimension Scaling": "Poor O(k^d)", "Pros": "Guaranteed coverage on small grids"},
            "Random Search": {"Budget": "User fixed n_iter", "Dimension Scaling": "High (Bergstra-Bengio)", "Pros": "Tests more unique values per parameter"},
            "Bayesian Optimization": {"Budget": "Sequential GP / TPE", "Dimension Scaling": "Sample efficient", "Pros": "Balances exploration and exploitation"}
        },
        "quiz": [
            {"q": "What is the main advantage of Random Search over Grid Search?", "a": "Bergstra-Bengio Theorem proves Random Search tests more unique values along important dimensions for the same compute budget."},
            {"q": "How does the Expected Improvement (EI) formula balance exploration vs exploitation?", "a": "The (mu - f_best)*Phi(Z) term rewards high predicted scores (exploitation), while sigma*phi(Z) rewards high uncertainty (exploration)."},
            {"q": "How many samples give at least 95% confidence of landing in top 5%?", "a": "n >= ln(0.05)/ln(0.95) = 58.4 -> Exactly 59 samples."}
        ]
    }
