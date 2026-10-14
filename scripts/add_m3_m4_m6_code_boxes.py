"""
Add code boxes for M3, M4, and M6 into templates/index.html.
"""

from pathlib import Path

INDEX_PATH = Path("templates/index.html")
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    html = f.read()

def make_code_box(title, code_str):
    return f"""<div class="code-box">
  <div class="code-box-header"><span>💻 {title}</span><button class="copy-code-btn" onclick="copyCode(this)">Copy Code</button></div>
  <pre>{code_str.strip()}</pre>
</div>"""

code_m3 = make_code_box("Python Implementation · Tree Models, GBDT, & SHAP Importance (M3)", """from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import numpy as np

# 1. Cost-Complexity Pruning (CCP) Path
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
# Select optimal alpha minimizing cross-validation error

# 2. Modern Histogram Gradient Boosting (LightGBM/XGBoost style)
gbdt = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.5,
    random_state=42
)
gbdt.fit(X_train, y_train)

# 3. Feature Importance Triad: Permutation Importance
perm = permutation_importance(gbdt, X_val, y_val, n_repeats=10, random_state=42)
for feat, imp in zip(feature_names, perm.importances_mean):
    print(f"{feat:15s}: {imp:+.4f}")""")

code_m4 = make_code_box("Python Implementation · PCA Pitfall Fix, DBSCAN & Anomaly Detection (M4)", """from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline

# 1. PCA Standardization Protection: NEVER run PCA on raw unscaled features!
# Without StandardScaler, high-magnitude features (e.g. Light ~ 8000 lux) dominate ~100% variance.
safe_pca = make_pipeline(StandardScaler(), PCA(n_components=2))
X_pca = safe_pca.fit_transform(X)

# 2. DBSCAN for Non-Convex Clusters & Automatic Noise Outlier Isolation
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
n_noise = np.sum(labels == -1)

# 3. Isolation Forest Anomaly Detection
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(X)  # -1 for anomaly, +1 for normal""")

code_m6 = make_code_box("Python Implementation · Production MLOps: Drift Monitoring & FastAPI Serving (M6)", """from scipy.stats import ks_2samp
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort

# 1. Production Data Drift Detection via Kolmogorov-Smirnov Test
def detect_feature_drift(baseline_data, production_data, alpha=0.01):
    stat, p_val = ks_2samp(baseline_data, production_data)
    drift_detected = p_val < alpha
    return {"ks_statistic": round(float(stat), 4), "p_value": round(float(p_val), 4), "drift": drift_detected}

# 2. Ultra Low-Latency Model Serving with ONNX Runtime & FastAPI
app = FastAPI(title="Produce Freshness Prediction API")
session = ort.InferenceSession("models/champion_model.onnx")

class ProducePayload(BaseModel):
    Temp: float
    Humid: float
    Light: float
    CO2: float

@app.post("/predict")
def serve_prediction(item: ProducePayload):
    inputs = np.array([[item.Temp, item.Humid, item.Light, item.CO2]], dtype=np.float32)
    ort_inputs = {session.get_inputs()[0].name: inputs}
    proba = session.run(None, ort_inputs)[1][0][1] # P(Good)
    return {"label": "Good" if proba >= 0.5 else "Bad", "probability": round(float(proba), 4)}""")

if 'id="m3RfBagging"' in html:
    pos = html.find('id="m3RfBagging"')
    end = html.find('</div></div>', pos) + 12
    html = html[:end] + "\n" + code_m3 + html[end:]
    print("Injected M3 code box.")

if 'id="m4DownstreamDetail"' in html:
    pos = html.find('id="m4DownstreamDetail"')
    end = html.find('</div></div>', pos) + 12
    html = html[:end] + "\n" + code_m4 + html[end:]
    print("Injected M4 code box.")

if 'id="m6MlopsDetail"' in html:
    pos = html.find('id="m6MlopsDetail"')
    end = html.find('</div>', pos) + 6
    html = html[:end] + "\n" + code_m6 + html[end:]
    print("Injected M6 code box.")

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Updated templates/index.html with M3, M4, M6 code boxes.")
