"""
Inject syntax-styled, copyable Python code boxes into templates/index.html
for all curriculum modules.
"""

from pathlib import Path

INDEX_PATH = Path("templates/index.html")
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add CSS for code box
code_css = """
  /* Code Box Component */
  .code-box { background: #0d1730; border: 1px solid #1f2d52; border-radius: 8px; margin-top: 14px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
  .code-box-header { display: flex; justify-content: space-between; align-items: center; background: #152244; padding: 7px 14px; font-size: 11px; color: #8da2d4; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; border-bottom: 1px solid #1f2d52; }
  .code-box pre { margin: 0; padding: 12px 14px; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 12px; color: #93c5fd; overflow-x: auto; line-height: 1.5; }
  .copy-code-btn { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 2px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; transition: all 0.2s; font-family: sans-serif; }
  .copy-code-btn:hover { background: var(--accent-teal); border-color: var(--accent-teal); color: #fff; }
"""

if ".code-box {" not in html:
    html = html.replace("</style>", code_css + "\n</style>")
    print("Added code-box CSS.")

# 2. Add copyCode JS function
copy_js = """
function copyCode(btn) {
  const pre = btn.closest('.code-box').querySelector('pre');
  navigator.clipboard.writeText(pre.innerText).then(() => {
    const orig = btn.innerText;
    btn.innerText = 'Copied!';
    setTimeout(() => { btn.innerText = orig; }, 1500);
  });
}
"""

if "function copyCode(" not in html:
    html = html.replace("</script>\n</body>", copy_js + "\n</script>\n</body>")
    print("Added copyCode JS function.")

def make_code_box(title, code_str):
    return f"""<div class="code-box">
  <div class="code-box-header"><span>💻 {title}</span><button class="copy-code-btn" onclick="copyCode(this)">Copy Code</button></div>
  <pre>{code_str.strip()}</pre>
</div>"""

# Module A Code
code_mod_a = make_code_box("Python Implementation · Leakage-Free Pipeline & 4-Fold CV (M5-1)", """from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. Isolate Vault Test Set (S09, S10) BEFORE any scaling or parameter fitting
X_dev, y_dev = X.loc[dev_indices], y.loc[dev_indices]
X_test, y_test = X.loc[test_indices], y.loc[test_indices]

# 2. Encapsulate StandardScaler inside Pipeline to prevent fold leakage
pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))

# 3. 4-Fold Stratified CV on Dev Set (S01-S08)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_dev, y_dev, cv=skf, scoring='accuracy')
# Mean CV Accuracy: 1.0000 (100% on placement benchmark)

# 4. Final fit on entire Dev partition, strictly evaluate on vault-locked test set
pipeline.fit(X_dev, y_dev)
test_accuracy = pipeline.score(X_test, y_test)  # 1.0000""")

# Module B Code
code_mod_b = make_code_box("Python Implementation · Nested Stratified Cross-Validation (M5-2)", """from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Double-loop architecture eliminating hyperparameter selection bias
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Inner Loop: Tunes hyperparameter C on training folds only
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid={'model__C': [0.01, 0.1, 1.0, 10.0]},
    cv=inner_cv,
    scoring='f1'
)

# Outer Loop: Unbiased generalization estimation on held-out test folds
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_f1_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1')

# Total model fits: K_out * (K_in * |Lambda| + 1) = 5 * (3 * 4 + 1) = 65 fits
print(f"Unbiased Nested F1: {nested_f1_scores.mean():.4f} (+/- {nested_f1_scores.std()*2:.4f})")""")

# Module C Code
code_mod_c = make_code_box("Python Implementation · F-Beta Harmonic Mean & Huber Loss (M5-3)", """import numpy as np

def compute_f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    \"\"\"Generalized F-beta: (1 + beta^2) * (P * R) / (beta^2 * P + R)\"\"\"
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return float((1 + b2) * (precision * recall) / (b2 * precision + recall))

def compute_huber_loss(residuals: np.ndarray, delta: float = 1.345) -> np.ndarray:
    \"\"\"Huber Hybrid Loss: quadratic for |e| <= delta, linear for |e| > delta\"\"\"
    abs_e = np.abs(residuals)
    return np.where(abs_e <= delta, 0.5 * (residuals ** 2), delta * (abs_e - 0.5 * delta))

# Slide 9-11 Case Study: S08 outlier error e = $45k
residuals = np.array([3.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 45.0, 2.0, 0.0])
huber = compute_huber_loss(residuals, delta=1.345)
mae = np.mean(np.abs(residuals))    # $6.50k
rmse = np.sqrt(np.mean(residuals**2)) # $14.54k""")

# Module D Code
code_mod_d = make_code_box("Python Implementation · ROC Threshold Trajectory & Pairwise AUC (M5-4)", """from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np

def evaluate_threshold(y_true, y_probs, tau=0.72, c_fp=1.0, c_fn=5.0):
    y_pred = (y_probs >= tau).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    youden_j = tpr - fpr
    dist_corner = np.sqrt((1.0 - tpr)**2 + fpr**2)
    expected_loss = c_fp * fp + c_fn * fn
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "TPR": tpr, "FPR": fpr, "Youden_J": youden_j, "Loss": expected_loss}

# Optimal Cutoff tau* = 0.72 achieves Youden's J = 1.0000 (Perfect separation)
# Pairwise Ranking AUC: U / (N_pos * N_neg) = 25 / (5 * 5) = 1.0000""")

# Module E Code
code_mod_e = make_code_box("Python Implementation · Precision-Recall Trajectory & Average Precision (M5-5)", """from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

# 100k Dilution Demo: 100 Positives, 99,000 Negatives
# 90 TP, 990 FP -> Recall = 90.0%, FPR = 1.0% (looks great), but Precision = 8.33% (91.7% false alarms!)

# Placement Benchmark under 20% Imbalance:
precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
ap_score = average_precision_score(y_true, y_probs)

# Slide 11 Hand Calculation: Rank hits at [0.90, 0.80, 0.60, 0.40]
# Precisions at hits: [1/1, 2/2, 3/4, 4/6] -> AP = mean([1.0, 1.0, 0.75, 0.667]) = 0.854
# Massive gain over random baseline prevalence pi = 0.50 (+0.354 gain)""")

# Module F Code
code_mod_f = make_code_box("Python Implementation · Platt Scaling Sigmoid & Isotonic PAVA (M5-6)", """import numpy as np
from sklearn.isotonic import IsotonicRegression

# 1. Platt Scaling: p_cal = 1 / (1 + exp(-(a*f + b))), where f = ln(p/(1-p))
def platt_scale_manual(p_raw, a=1.773, b=0.789):
    eps = 1e-12
    p_clipped = np.clip(p_raw, eps, 1 - eps)
    f = np.log(p_clipped / (1 - p_clipped)) # Log-odds logit
    return 1.0 / (1.0 + np.exp(-(a * f + b)))

# S7 Worked Example: p = 0.60 -> f = ln(1.5) = 0.405 -> p_cal = 0.819
s7_cal = platt_scale_manual(np.array([0.60]), a=1.773, b=0.789)[0]

# 2. PAVA Isotonic Regression (Non-parametric step function)
ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
p_iso = ir.fit_transform(p_raw, y_true)

# 3. Brier Score Loss: (1/N) * sum((p - y)^2)
# 10-Student Placement Benchmark: Raw (0.157) -> Platt (0.143) -> Isotonic (0.100)""")

# Module G Code
code_mod_g = make_code_box("Python Implementation · Search Budget Solver & GP Expected Improvement (M5-7)", """import math
import numpy as np
from scipy.stats import norm

# Slide 10 Search Budget Theorem (Bergstra-Bengio):
# P(at least one in top q) = 1 - (1 - q)^n
# For q = 0.05 and n = 60: P = 1 - (0.95)^60 = 0.9539 (95.4%)
# Minimum budget for 95% confidence: n >= ln(0.05) / ln(0.95) = 58.4 -> n = 59

def compute_expected_improvement(mu, sigma, f_best):
    \"\"\"Expected Improvement: EI(theta) = (mu - f_best)*Phi(Z) + sigma*phi(Z)\"\"\"
    z = (mu - f_best) / (sigma + 1e-9)
    ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)""")

# Module H Code
code_mod_h = make_code_box("Python Implementation · Cluster-Augmented Regression (Location OLS Fix)", """import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Exact 5-House Dataset:
# House 1: L=1, S=1000, P=150 (Cluster A)  |  House 3: L=8, S=1000, P=300 (Cluster B)
# House 2: L=2, S=1200, P=180 (Cluster A)  |  House 4: L=9, S=1200, P=340 (Cluster B)
#                                          |  House 5: L=8.5, S=1100, P=320 (Cluster B)

# Step 1: K-Means on Location -> Centroids at 1.5 and 8.5
km = KMeans(n_clusters=2, random_state=42).fit(df[['Location_L']])
df['Cluster_B'] = (km.labels_ == 1).astype(float)

# Step 2: Baseline OLS (Size only): P_hat = 65.5 + 0.175*S -> R^2 = 0.039 (Residual bias!)
# Step 3: Augmented OLS with Interaction: P_hat = 0.15*S + 100*C + 0.05*(S*C) -> R^2 = 1.000!
X_aug = np.column_stack([df['Size_S'], df['Cluster_B'], df['Size_S'] * df['Cluster_B']])
model_aug = LinearRegression().fit(X_aug, df['Price_P'])""")

# Module I Code
code_mod_i = make_code_box("Python Implementation · Lloyd's K-Means & Slide 24 Hand Silhouette", """import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Slide 24 Hand Silhouette Calculation:
# Points: A(1,1), B(2,1), C(1,2) in C1; D(5,5), E(6,5) in C2
# For A: a(A) = mean(dist(A,B), dist(A,C)) = (1 + 1)/2 = 1.000
#        b(A) = mean(dist(A,D), dist(A,E)) = (5.657 + 6.403)/2 = 6.030
#        s(A) = (b(A) - a(A)) / max(a(A), b(A)) = (6.030 - 1.000) / 6.030 = 0.834
# Mean s_bar = 0.801 -> Strong cluster structure (0.71 to 1.00 range)

# Lloyd's alternating steps:
# 1. Assignment: c(i) = argmin_k ||x_i - mu_k||^2
# 2. Update:     mu_k = (1 / |C_k|) * sum_{i in C_k} x_i""")

# Module J Code
code_mod_j = make_code_box("Python Implementation · Hierarchical Ward Linkage & Optimal Cut (M4-2)", """import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 5 Points: A=2, B=4, C=10, D=12, E=18
X = np.array([2.0, 4.0, 10.0, 12.0, 18.0]).reshape(-1, 1)

# Ward linkage variance increase: Delta_J = (|A|*|B| / (|A| + |B|)) * ||mu_A - mu_B||^2
# Height: h = sqrt(Delta_J)
# Merge Sequence:
# Step 1: (A,B) -> h = 1.41
# Step 2: (C,D) -> h = 1.41
# Step 3: ({CD}, E) -> h = 5.72
# Step 4: ({AB}, {CDE}) -> h = 11.32 (Root)
# Optimal Cut: Largest gap is 11.32 - 5.72 = 5.60 -> Cut at h = 8.0 yields K = 2 clusters!
Z = linkage(X, method='ward')
clusters = fcluster(Z, t=2, criterion='maxclust')""")

# Inject into sections
insertions = [
    ('id="modF_tableDetail"', code_mod_f),
    ('id="modG_gpDetail"', code_mod_g),
    ('id="modH_tableDetail"', code_mod_h),
    ('id="modI_vivaDetail"', code_mod_i),
    ('id="modJ_vivaDetail"', code_mod_j),
]

for target, box in insertions:
    # Find closing </div> of this element
    pos = html.find(target)
    if pos != -1:
        end_pos = html.find('</div>', pos)
        if end_pos != -1:
            end_pos += 6
            html = html[:end_pos] + "\n" + box + html[end_pos:]
            print(f"Injected code box for {target}")
    else:
        print(f"Warning: target {target} not found")

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Dashboard templates/index.html successfully updated with embedded code boxes.")
