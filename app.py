"""
Flask serving app + real-time dashboard.

Run AFTER src/eda.py has been run at least once:
    python src/eda.py     (only needed once, or after data changes)
    python app.py

Then open http://127.0.0.1:5000 in your browser.

On first run, if no trained model exists yet, this script trains one
automatically (calls src/train_model.py) so there's no separate step to
remember beyond eda.py.
"""
from pathlib import Path
import sys
import json
import time
from collections import deque

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, render_template

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.csv"
REGISTRY_PATH = MODELS_DIR / "registry.json"

sys.path.insert(0, str(SRC_DIR))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

_state = {"pipeline": None, "registry": None, "recent_predictions": deque(maxlen=200)}


def ensure_trained():
    artifact_ok = False
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                registry = json.load(f)
            artifact_path = MODELS_DIR / Path(registry["champion_artifact"]).name
            artifact_ok = artifact_path.exists()
        except Exception:
            artifact_ok = False

    if artifact_ok:
        print("[app] Found existing trained model.")
        return

    print("[app] No valid trained model found — training now...")
    import train_model
    train_model.run()
    print("[app] Training complete.")


def load_model():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    # Resolve the artifact path relative to this machine's models/ folder,
    # never trust an absolute path baked into registry.json — it may have
    # been generated on a different machine (this is what broke it before).
    artifact_path = MODELS_DIR / Path(registry["champion_artifact"]).name
    pipeline = joblib.load(artifact_path)
    _state["pipeline"] = pipeline
    _state["registry"] = registry
    print(f"[app] Loaded champion model: {registry['champion_model']}")


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Static report files (plots) — served so <img> tags in the dashboard work
# ---------------------------------------------------------------------------
@app.route("/reports/<path:subpath>")
def serve_reports(subpath):
    return send_from_directory(REPORTS_DIR, subpath)


# ---------------------------------------------------------------------------
# API: health / model info
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _state["pipeline"] is not None,
                     "timestamp": time.time()})


@app.route("/api/model_info")
def model_info():
    if _state["registry"] is None:
        return jsonify({"error": "Model not loaded"}), 503
    r = _state["registry"]
    return jsonify({
        "champion_model": r["champion_model"],
        "created_at": r["created_at"],
        "test_metrics_all_models": r["test_metrics_all_models"],
    })


# ---------------------------------------------------------------------------
# API: dataset stats + preview
# ---------------------------------------------------------------------------
@app.route("/api/stats")
def dataset_stats():
    if not DATA_PATH.exists():
        return jsonify({"error": "Processed dataset not found"}), 503
    df = pd.read_csv(DATA_PATH)
    good_rate = float((df["Class"] == "Good").mean())
    return jsonify({
        "total_records": int(len(df)),
        "good_rate": round(good_rate * 100, 2),
        "avg_temp": round(float(df["Temp"].mean()), 2),
        "avg_humidity": round(float(df["Humid (%)"].mean()), 2),
        "avg_light": round(float(df["Light (Fux)"].mean()), 2),
        "avg_co2": round(float(df["CO2 (pmm)"].mean()), 2),
        "class_distribution": df["Class"].value_counts().to_dict(),
        "fruit_distribution": df["Fruit"].value_counts().to_dict(),
    })


@app.route("/api/dataset_preview")
def dataset_preview():
    limit = min(max(int(request.args.get("limit", 15)), 1), 100)
    df = pd.read_csv(DATA_PATH)
    return jsonify({"columns": list(df.columns), "rows": df.head(limit).to_dict(orient="records")})


# ---------------------------------------------------------------------------
# API: session-topic reports (EDA / scaling / encoding / missing values)
# ---------------------------------------------------------------------------
def _load_report(name: str):
    path = REPORTS_DIR / name / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@app.route("/api/eda")
def api_eda():
    report = _load_report("eda")
    if report is None:
        return jsonify({"error": "No EDA report found. Run `python src/eda.py` first."}), 404
    images = sorted(p.name for p in (REPORTS_DIR / "eda").glob("*.png"))
    return jsonify({"summary": report, "images": images})


@app.route("/api/scaling")
def api_scaling():
    report = _load_report("scaling")
    if report is None:
        return jsonify({"error": "No scaling report found. Run `python src/scaling.py` first."}), 404
    return jsonify(report)


@app.route("/api/encoding")
def api_encoding():
    report = _load_report("encoding")
    if report is None:
        return jsonify({"error": "No encoding report found. Run `python src/encoding.py` first."}), 404
    return jsonify(report)


@app.route("/api/missing_values")
def api_missing_values():
    report = _load_report("missing_values")
    if report is None:
        return jsonify({"error": "No missing-values report found. Run `python src/missing_values.py` first."}), 404
    return jsonify(report)


@app.route("/api/decision_tree")
def api_decision_tree():
    report = _load_report("decision_tree")
    if report is None:
        return jsonify({"error": "No decision-tree report found. Run `python src/decision_tree.py` first."}), 404
    return jsonify(report)


@app.route("/api/random_forest")
def api_random_forest():
    report = _load_report("random_forest")
    if report is None:
        return jsonify({"error": "No random-forest report found. Run `python src/random_forest.py` first."}), 404
    return jsonify(report)


@app.route("/api/model_comparison")
def api_model_comparison():
    report = _load_report("model_comparison")
    if report is None:
        return jsonify({"error": "No model-comparison report found. Run `python src/model_comparison.py` first."}), 404
    return jsonify(report)


# ---------------------------------------------------------------------------
# API: Curriculum Modules (Modules A through J + M3-M6)
# ---------------------------------------------------------------------------
@app.route("/api/curriculum/<module_id>")
def api_curriculum(module_id):
    from curriculum.curriculum_engine import get_module_report
    from curriculum.curriculum_service import get_curriculum_report

    # Try Module A-J engine first, then fallback to M3-M6 service
    data = get_module_report(module_id)
    if not data:
        data = get_curriculum_report(module_id)
    if not data:
        return jsonify({"error": f"Module {module_id} not found"}), 404
    return jsonify(data)


@app.route("/api/curriculum/interactive/<module_id>", methods=["GET", "POST"])
def api_curriculum_interactive(module_id):
    from curriculum.curriculum_engine import handle_interactive_query
    payload = request.get_json(silent=True) or request.args.to_dict()
    res = handle_interactive_query(module_id, payload)
    return jsonify(res)


@app.route("/api/curriculum/run_all", methods=["POST"])
def api_curriculum_run_all():
    from curriculum.curriculum_engine import regenerate_all_reports
    from curriculum.curriculum_service import generate_all_curriculum_reports

    res_engine = regenerate_all_reports()
    res_service = generate_all_curriculum_reports()
    all_mods = list(res_engine.keys()) + list(res_service.keys())
    return jsonify({
        "status": "ok",
        "message": "All curriculum modules regenerated.",
        "modules": all_mods
    })



# ---------------------------------------------------------------------------
# API: live prediction
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if _state["pipeline"] is None:
        return jsonify({"error": "Model not loaded"}), 503
    payload = request.get_json(force=True)
    try:
        row = pd.DataFrame([{
            "Temp": float(payload["Temp"]),
            "Humid (%)": float(payload["Humid"]),
            "Light (Fux)": float(payload["Light"]),
            "CO2 (pmm)": float(payload["CO2"]),
            "Fruit": payload["Fruit"],
        }])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid payload: {e}"}), 422

    proba_good = float(_state["pipeline"].predict_proba(row)[0, 1])
    label = "Good" if proba_good >= 0.5 else "Bad"

    record = {"timestamp": time.time(), "prediction": label, "probability_good": round(proba_good, 4),
               **{k: payload[k] for k in ["Fruit", "Temp", "Humid", "Light", "CO2"]}}
    _state["recent_predictions"].append(record)

    return jsonify({
        "prediction": label,
        "probability_good": round(proba_good, 4),
        "model_version": _state["registry"]["champion_model"],
    })


# ---------------------------------------------------------------------------
# API: real-time monitoring feed (recent predictions, for the live dashboard)
# ---------------------------------------------------------------------------
@app.route("/api/recent_predictions")
def recent_predictions():
    items = list(_state["recent_predictions"])[-20:][::-1]  # most recent first
    good_count = sum(1 for i in _state["recent_predictions"] if i["prediction"] == "Good")
    total = len(_state["recent_predictions"])
    return jsonify({
        "items": items,
        "total_logged": total,
        "good_rate_pct": round(good_count / total * 100, 1) if total else None,
    })


if __name__ == "__main__":
    ensure_trained()
    load_model()
    print("\n[app] Starting FreshPredict dashboard at http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
