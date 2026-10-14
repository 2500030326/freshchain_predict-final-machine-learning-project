"""
Curriculum Engine: Unified Master Dispatcher for Modules A through J
Integrates all 10 authoritative PowerPoint decks into the FreshPredict backend.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from .module_a_split_leakage import run_module_a_walkthrough
    from .module_b_nested_cv import run_module_b_walkthrough, compute_nested_fits_count
    from .module_c_metrics import run_module_c_walkthrough, compute_f_beta, compute_huber_loss
    from .module_d_roc_auc import run_module_d_walkthrough, compute_threshold_metrics, get_ranked_placement_dataset
    from .module_e_pr_curves import run_module_e_walkthrough
    from .module_f_calibration import run_module_f_walkthrough, platt_scale_manual
    from .module_g_hyperparam_search import run_module_g_walkthrough, solve_search_budget_practice_problem
    from .module_h_cluster_regression import run_module_h_walkthrough, predict_house_price
    from .module_i_kmeans import run_module_i_walkthrough
    from .module_j_hierarchical import run_module_j_walkthrough
except (ImportError, ValueError):
    try:
        from curriculum.module_a_split_leakage import run_module_a_walkthrough
        from curriculum.module_b_nested_cv import run_module_b_walkthrough, compute_nested_fits_count
        from curriculum.module_c_metrics import run_module_c_walkthrough, compute_f_beta, compute_huber_loss
        from curriculum.module_d_roc_auc import run_module_d_walkthrough, compute_threshold_metrics, get_ranked_placement_dataset
        from curriculum.module_e_pr_curves import run_module_e_walkthrough
        from curriculum.module_f_calibration import run_module_f_walkthrough, platt_scale_manual
        from curriculum.module_g_hyperparam_search import run_module_g_walkthrough, solve_search_budget_practice_problem
        from curriculum.module_h_cluster_regression import run_module_h_walkthrough, predict_house_price
        from curriculum.module_i_kmeans import run_module_i_walkthrough
        from curriculum.module_j_hierarchical import run_module_j_walkthrough
    except ImportError:
        from src.curriculum.module_a_split_leakage import run_module_a_walkthrough
        from src.curriculum.module_b_nested_cv import run_module_b_walkthrough, compute_nested_fits_count
        from src.curriculum.module_c_metrics import run_module_c_walkthrough, compute_f_beta, compute_huber_loss
        from src.curriculum.module_d_roc_auc import run_module_d_walkthrough, compute_threshold_metrics, get_ranked_placement_dataset
        from src.curriculum.module_e_pr_curves import run_module_e_walkthrough
        from src.curriculum.module_f_calibration import run_module_f_walkthrough, platt_scale_manual
        from src.curriculum.module_g_hyperparam_search import run_module_g_walkthrough, solve_search_budget_practice_problem
        from src.curriculum.module_h_cluster_regression import run_module_h_walkthrough, predict_house_price
        from src.curriculum.module_i_kmeans import run_module_i_walkthrough
        from src.curriculum.module_j_hierarchical import run_module_j_walkthrough

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "reports" / "curriculum"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RUNNERS = {
    "module_a": run_module_a_walkthrough,
    "module_b": run_module_b_walkthrough,
    "module_c": run_module_c_walkthrough,
    "module_d": run_module_d_walkthrough,
    "module_e": run_module_e_walkthrough,
    "module_f": run_module_f_walkthrough,
    "module_g": run_module_g_walkthrough,
    "module_h": run_module_h_walkthrough,
    "module_i": run_module_i_walkthrough,
    "module_j": run_module_j_walkthrough,
}


def get_module_report(module_id: str) -> dict:
    """Retrieves or computes report for given module ID."""
    clean_id = module_id.lower().replace("-", "_")
    cache_path = CACHE_DIR / f"{clean_id}.json"

    # Try loading from cache
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Compute live
    if clean_id in RUNNERS:
        data = RUNNERS[clean_id]()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    return {}


def regenerate_all_reports() -> dict:
    """Computes and caches reports for all 10 modules."""
    results = {}
    for mod_id, runner in RUNNERS.items():
        data = runner()
        cache_path = CACHE_DIR / f"{mod_id}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        results[mod_id] = "generated"
    return results


def handle_interactive_query(module_id: str, payload: dict) -> dict:
    """Handles real-time parameter sweeps and what-if calculations."""
    mid = module_id.lower().replace("-", "_")

    if mid == "module_d":
        # Interactive threshold change
        tau = float(payload.get("threshold", 0.50))
        c_fp = float(payload.get("cost_fp", 1.0))
        c_fn = float(payload.get("cost_fn", 5.0))
        df = get_ranked_placement_dataset()
        res = compute_threshold_metrics(df["TrueClass"].values, df["PredictedProb"].values, tau, c_fp, c_fn)
        return res

    elif mid == "module_h":
        # Interactive house price prediction
        size = float(payload.get("size", 1100.0))
        cluster = str(payload.get("cluster", "A")).upper()
        return predict_house_price(size, cluster)

    elif mid == "module_c":
        # Interactive F-beta & Huber loss
        prec = float(payload.get("precision", 0.8333))
        rec = float(payload.get("recall", 1.0))
        beta = float(payload.get("beta", 1.0))
        delta = float(payload.get("delta", 1.345))
        err = float(payload.get("residual", 4.0))
        f_val = compute_f_beta(prec, rec, beta)
        huber_val = float(compute_huber_loss(np.array([err]), delta)[0])
        return {
            "beta": beta,
            "f_beta": round(f_val, 4),
            "delta": delta,
            "residual": err,
            "huber_loss": round(huber_val, 4),
            "squared_error_loss": round(0.5 * (err ** 2), 4),
            "absolute_error_loss": round(abs(err), 4)
        }

    elif mid == "module_b":
        # Interactive N_fits formula
        k_out = int(payload.get("k_outer", 5))
        k_in = int(payload.get("k_inner", 3))
        grid_sz = int(payload.get("grid_size", 4))
        n_fits = compute_nested_fits_count(k_out, k_in, grid_sz)
        return {"k_outer": k_out, "k_inner": k_in, "grid_size": grid_sz, "total_model_fits": n_fits}

    elif mid == "module_g":
        # Interactive search budget
        n_lr = int(payload.get("n_lr", 4))
        n_depths = int(payload.get("n_depths", 3))
        k_folds = int(payload.get("k_folds", 5))
        conf = float(payload.get("confidence", 0.95))
        return solve_search_budget_practice_problem(n_lr, n_depths, k_folds, confidence=conf)

    elif mid == "module_f":
        # Interactive Platt scaling for custom raw probability
        p_raw = float(payload.get("raw_probability", 0.60))
        a = float(payload.get("param_a", 1.773))
        b = float(payload.get("param_b", 0.789))
        p_cal = float(platt_scale_manual(np.array([p_raw]), a=a, b=b)[0])
        return {"raw_probability": p_raw, "a": a, "b": b, "calibrated_probability": round(p_cal, 4)}

    return {"error": f"Interactive queries not configured for {module_id}"}
