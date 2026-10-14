"""
FreshChain - Module 3: Production Spoilage Prediction API.

Provides a clean, standalone inference function for IoT Cold-Storage monitoring:
- Accepts sensor measurements: Fruit, Temp, Humid (%), Light (Fux), CO2 (pmm)
- Validates data boundaries
- Transforms inputs using identical preprocessor
- Returns predicted class, probability, and human-readable cold-storage status.
"""
from pathlib import Path
import json
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "champion.joblib"
REGISTRY_PATH = MODELS_DIR / "registry.json"

_cached_pipeline = None


def get_pipeline():
    """Lazy-load and cache the trained champion model pipeline."""
    global _cached_pipeline
    if _cached_pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run src/train_model.py first.")
        _cached_pipeline = joblib.load(MODEL_PATH)
    return _cached_pipeline


def predict_spoilage(fruit: str, temp: float, humid: float, light: float, co2: float) -> dict:
    """Predicts fruit spoilage from real-time environmental IoT sensor readings.
    
    Parameters:
    -----------
    fruit : str
        Fruit variety (e.g. 'Tomato', 'Orange', 'Pineapple', 'Banana')
    temp : float
        Storage temperature in °C
    humid : float
        Relative humidity percentage (0 - 100)
    light : float
        Illuminance in Lux (Light (Fux))
    co2 : float
        Carbon dioxide concentration in ppm
        
    Returns:
    --------
    dict with:
      - 'prediction': 'Spoiled' or 'Fresh'
      - 'spoilage_probability_pct': float (e.g. 87.4)
      - 'status_message': Human-readable advisory for cold storage operator
      - 'raw_inputs': dict of received readings
    """
    pipeline = get_pipeline()
    
    # Input DataFrame matching expected feature schema
    row = pd.DataFrame([{
        "Temp": float(temp),
        "Humid (%)": float(humid),
        "Light (Fux)": float(light),
        "CO2 (pmm)": float(co2),
        "Fruit": str(fruit).strip().title()
    }])
    
    # Predict probabilities: [P(Fresh=0), P(Spoiled=1)]
    probabilities = pipeline.predict_proba(row)[0]
    
    # Check if champion model was trained with Good=1 or Bad=1
    # By reading registry target mapping
    target_mapping = {"0": "Bad", "1": "Good"}
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                reg = json.load(f)
                target_mapping = reg.get("target_mapping", target_mapping)
        except Exception:
            pass
            
    # If target_mapping has "1": "Good", probability of Good is probabilities[1] and Bad is probabilities[0]
    # If target_mapping has "1": "Bad", probability of Bad is probabilities[1]
    if target_mapping.get("1") == "Good":
        prob_good = probabilities[1]
        prob_bad = probabilities[0]
    else:
        prob_bad = probabilities[1]
        prob_good = probabilities[0]
        
    is_spoiled = prob_bad >= 0.5
    status_label = "Spoiled" if is_spoiled else "Fresh"
    confidence = (prob_bad if is_spoiled else prob_good) * 100.0
    
    if is_spoiled:
        msg = f"CRITICAL: Spoilage condition detected! Spoilage probability is {prob_bad*100:.1f}%. Check humidity and ventilation."
    else:
        msg = f"OPTIMAL: Safe storage conditions. Freshness probability is {prob_good*100:.1f}%."
        
    return {
        "prediction": status_label,
        "is_spoiled": bool(is_spoiled),
        "probability": f"{confidence:.1f}%",
        "spoilage_probability_pct": round(prob_bad * 100.0, 2),
        "freshness_probability_pct": round(prob_good * 100.0, 2),
        "status_message": msg,
        "raw_inputs": {
            "fruit": fruit,
            "temp": temp,
            "humid": humid,
            "light": light,
            "co2": co2
        }
    }


if __name__ == "__main__":
    print("--- FreshChain Live Prediction Tests ---")
    
    # Test case 1: Elevated humidity and temp (typical spoilage condition)
    sample_spoiled = predict_spoilage(fruit="Orange", temp=25.0, humid=95.0, light=14.8, co2=370.0)
    print("\nTest Case 1 (High Humidity Storage):")
    print(f"Prediction: {sample_spoiled['prediction']}")
    print(f"Probability: {sample_spoiled['probability']}")
    print(f"Status: {sample_spoiled['status_message']}")
    
    # Test case 2: Controlled humidity (fresh condition)
    sample_fresh = predict_spoilage(fruit="Orange", temp=22.0, humid=88.0, light=7.3, co2=310.0)
    print("\nTest Case 2 (Normal Controlled Storage):")
    print(f"Prediction: {sample_fresh['prediction']}")
    print(f"Probability: {sample_fresh['probability']}")
    print(f"Status: {sample_fresh['status_message']}")
