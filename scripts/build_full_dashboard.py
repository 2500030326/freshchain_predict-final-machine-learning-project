"""
Script to build the comprehensive, production-grade templates/index.html
incorporating Modules A through J with interactive simulators and full slide data.
"""

from pathlib import Path

INDEX_PATH = Path("templates/index.html")

# Read existing index.html to preserve all existing styles and sections
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the sidebar navigation
old_sidebar_curriculum = """  <div class="nav-section-label">Module 3 · Tree Models</div>
  <div class="nav-item" data-target="m3_curriculum"><span class="nav-icon">🌳</span>M3: Trees, GBDT & SHAP</div>
  <div class="nav-item" data-target="decision_tree"><span class="nav-icon">🌿</span>Decision Tree & Pruning</div>
  <div class="nav-item" data-target="random_forest"><span class="nav-icon">🌲</span>Random Forest & OOB</div>
  <div class="nav-item" data-target="model_comparison"><span class="nav-icon">🏆</span>Model Comparison</div>

  <div class="nav-section-label">Module 4 · Unsupervised</div>
  <div class="nav-item" data-target="m4_curriculum"><span class="nav-icon">🔮</span>M4: Clustering, PCA & Anomalies</div>

  <div class="nav-section-label">Module 5 · Evaluation</div>
  <div class="nav-item" data-target="m5_curriculum"><span class="nav-icon">🎯</span>M5: Split, CV & Calibration</div>

  <div class="nav-section-label">Module 6 · ML Engineering</div>
  <div class="nav-item" data-target="m6_curriculum"><span class="nav-icon">🚀</span>M6: Production, Skew & Drift</div>"""

new_sidebar_curriculum = """  <div class="nav-section-label">M5 Sessions · Evaluation & Optimization</div>
  <div class="nav-item" data-target="module_a"><span class="nav-icon">🔒</span>M5-1: Split & Data Leakage</div>
  <div class="nav-item" data-target="module_b"><span class="nav-icon">🔁</span>M5-2: Stratified & Nested CV</div>
  <div class="nav-item" data-target="module_c"><span class="nav-icon">📏</span>M5-3: Metrics & Huber Loss</div>
  <div class="nav-item" data-target="module_d"><span class="nav-icon">🎯</span>M5-4: ROC-AUC & Youden's J</div>
  <div class="nav-item" data-target="module_e"><span class="nav-icon">📉</span>M5-5: PR Curves & Imbalance</div>
  <div class="nav-item" data-target="module_f"><span class="nav-icon">🎲</span>M5-6: Calibration & Brier Score</div>
  <div class="nav-item" data-target="module_g"><span class="nav-icon">🔍</span>M5-7: Hyperparameter Search</div>
  <div class="nav-item" data-target="m5_curriculum"><span class="nav-icon">📊</span>M5: Full Evaluation Overview</div>

  <div class="nav-section-label">M4 Sessions · Unsupervised Learning</div>
  <div class="nav-item" data-target="module_i"><span class="nav-icon">📍</span>M4-1: K-Means & Lloyd's Alg</div>
  <div class="nav-item" data-target="module_j"><span class="nav-icon">🌲</span>M4-2: Hierarchical Clustering</div>
  <div class="nav-item" data-target="module_h"><span class="nav-icon">🏠</span>M4-3: Cluster Regression</div>
  <div class="nav-item" data-target="m4_curriculum"><span class="nav-icon">🔮</span>M4: Full Clustering Overview</div>

  <div class="nav-section-label">Tree Models & Engineering</div>
  <div class="nav-item" data-target="m3_curriculum"><span class="nav-icon">🌳</span>M3: Trees, GBDT & SHAP</div>
  <div class="nav-item" data-target="decision_tree"><span class="nav-icon">🌿</span>Decision Tree & Pruning</div>
  <div class="nav-item" data-target="random_forest"><span class="nav-icon">🌲</span>Random Forest & OOB</div>
  <div class="nav-item" data-target="model_comparison"><span class="nav-icon">🏆</span>Model Comparison</div>
  <div class="nav-item" data-target="m6_curriculum"><span class="nav-icon">🚀</span>M6: Production, Skew & Drift</div>"""

if old_sidebar_curriculum in content:
    content = content.replace(old_sidebar_curriculum, new_sidebar_curriculum)
    print("Updated sidebar navigation.")
else:
    print("Warning: old sidebar block not matched exactly.")

# 2. Prepare HTML sections for Modules A through J
modules_sections_html = """
  <!-- ===================================================================== -->
  <!-- MODULE A: TRAIN / VAL / TEST SPLIT & LEAKAGE (M5-1)                    -->
  <!-- ===================================================================== -->
  <section id="module_a">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module A (M5-1) · Train / Validation / Test Split Discipline & Data Leakage</h2>
        <p class="section-sub">Source: <code>module_5_session_M5_S1 (L59-60).pptx</code> (23 Slides) · Minimizing Empirical Risk vs. True Risk, 3-Way Partitioning, Data Leakage Taxonomy, 4-Fold CV Execution, and Vault-Locked Final Evaluation.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M5-1</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Split Discipline</div><div class="stat-value" style="color:var(--accent-teal);">80% / 20%</div><div class="stat-foot">Dev Set (8) / Vault Test (2)</div></div>
      <div class="card"><div class="stat-label">Cross-Validation</div><div class="stat-value">4-Fold</div><div class="stat-foot">Stratified K-Fold on Dev</div></div>
      <div class="card"><div class="stat-label">Development CV Accuracy</div><div class="stat-value" id="modA_cvAcc">100%</div><div class="stat-foot">Mean: 1.0000 (±0.0000)</div></div>
      <div class="card"><div class="stat-label">Vault-Locked Test Acc</div><div class="stat-value" id="modA_testAcc">100%</div><div class="stat-foot">Evaluated strictly once</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. 4-Fold Stratified Cross-Validation Execution Matrix</h3>
        <p style="font-size:12px; color:var(--text-muted);">Allocation matrix of development students S01–S08 across 4 validation iterations (Slide 19):</p>
        <div id="modA_matrixDetail"><div class="loading">Loading matrix…</div></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. Vault-Locked Test Set Evaluation (S09 & S10)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Unbiased test predictions evaluated only after pipeline finalization (Slide 16):</p>
        <div id="modA_testDetail"><div class="loading">Loading test results…</div></div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">3. Data Leakage Taxonomy & Golden Engineering Rules</h3>
        <p style="font-size:12px; color:var(--text-muted);">Eliminating contamination across all operational pipeline stages (Slide 9):</p>
        <table>
          <thead><tr><th>Leakage Class</th><th>Mechanism</th><th>Production Remedy</th></tr></thead>
          <tbody>
            <tr><td><b>Preprocessing Leakage</b></td><td>Fitting scalers/imputers on full dataset prior to splitting</td><td>Encapsulate inside Scikit-Learn <code>Pipeline</code></td></tr>
            <tr><td><b>Temporal Leakage</b></td><td>Using future features to predict historical states</td><td>Strict point-in-time time-travel joins</td></tr>
            <tr><td><b>Target Contamination</b></td><td>Including features that encode ground-truth target</td><td>Automated feature dependency audits</td></tr>
          </tbody>
        </table>
        <div class="warn-box" style="margin-top:12px;">🔒 <b>Golden Rule:</b> The test set must remain completely isolated from every modeling decision, hyperparameter tuning iteration, and feature transformation fit!</div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">4. Self-Assessment Concept Check Quiz (Slide 23)</h3>
        <div id="modA_quizDetail"><div class="loading">Loading quiz…</div></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE B: STRATIFIED & NESTED CROSS-VALIDATION (M5-2)                  -->
  <!-- ===================================================================== -->
  <section id="module_b">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module B (M5-2) · Stratified & Nested Cross-Validation</h2>
        <p class="section-sub">Source: <code>module_5_session_M5_S2 (L61-62).pptx</code> (18 Slides) · Class Prior Preservation, Hyperparameter Selection Bias, Nested CV Double-Loop Scheme, and Model Fit Formula.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M5-2</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Model Fits Formula</div><div class="stat-value" style="font-size:20px; color:var(--accent-teal);">N_fits</div><div class="stat-foot">K_out × (K_in × |Λ| + 1)</div></div>
      <div class="card"><div class="stat-label">Benchmark Grid Fits</div><div class="stat-value" id="modB_totalFits">155 fits</div><div class="stat-foot">K_out=5, K_in=3, |Λ|=10</div></div>
      <div class="card"><div class="stat-label">Unbiased Nested F1</div><div class="stat-value" id="modB_nestedF1">0.9430</div><div class="stat-foot">±0.0308 (Slide 11 benchmark)</div></div>
      <div class="card"><div class="stat-label">Optimistic Inner Score</div><div class="stat-value" style="color:var(--accent-orange);">0.9600</div><div class="stat-foot">Selection bias: +0.0170</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. Interactive Nested CV Model-Fit Calculator</h3>
        <p style="font-size:12px; color:var(--text-muted);">Dynamically compute required model trainings based on outer folds, inner folds, and hyperparameter lattice:</p>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:12px;">
          <div><label style="font-size:11px; font-weight:600;">K_outer Folds</label><input type="number" id="modB_in_kout" value="5" min="2" max="10" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">K_inner Folds</label><input type="number" id="modB_in_kin" value="3" min="2" max="10" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">Grid Options |Λ|</label><input type="number" id="modB_in_grid" value="10" min="1" max="100" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
        </div>
        <button onclick="calculateModBFits()" class="predict-btn" style="padding:9px; font-size:12px; margin-top:0;">Recalculate Model Fits</button>
        <div id="modB_fitsResult" style="margin-top:12px; padding:12px; background:#e9fbf6; border-radius:8px; font-weight:600;">5 × (3 × 10 + 1) = 155 total model training runs.</div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. Stratified K-Fold Class Prior Allocation (Slide 10)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Exact preservation of 67% Placed / 33% Not Placed across all 5 evaluation folds (N=1,000):</p>
        <div id="modB_proportionsTable"><div class="loading">Loading fold allocation…</div></div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">3. Inner Loop Grid Search vs. Outer Loop Trace</h3>
        <p style="font-size:12px; color:var(--text-muted);">Trace of hyperparameter selection over C in {0.01, 0.1, 1.0, 10.0} (Slide 11):</p>
        <div id="modB_traceDetail"><div class="loading">Loading trace…</div></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">4. Self-Assessment Concept Checks (Slide 18)</h3>
        <div id="modB_quizDetail"><div class="loading">Loading quiz…</div></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE C: CLASSIFICATION & REGRESSION METRICS (M5-3)                  -->
  <!-- ===================================================================== -->
  <section id="module_c">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module C (M5-3) · Classification & Regression Metrics</h2>
        <p class="section-sub">Source: <code>Module_5_Session_M5-3.pptx</code> (15 Slides) · Accuracy Paradox, Precision/Recall/F1/F-beta, Outlier Sensitivity (RMSE vs MAE), Huber Loss (δ=1.345), and 10-Student Case Study.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M5-3</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Case Study Accuracy</div><div class="stat-value" style="color:var(--accent-teal);">90.0%</div><div class="stat-foot">9/10 correct (S08 False Alarm)</div></div>
      <div class="card"><div class="stat-label">Precision & Recall</div><div class="stat-value">83.3% / 100%</div><div class="stat-foot">F1: 90.91% · F0.5: 86.2%</div></div>
      <div class="card"><div class="stat-label">MAE (Robust L1)</div><div class="stat-value">$6.50k</div><div class="stat-foot">SAE: $65.0k (10 students)</div></div>
      <div class="card"><div class="stat-label">RMSE (Outlier L2)</div><div class="stat-value" style="color:var(--accent-orange);">$14.54k</div><div class="stat-foot">2.24× MAE (Dominated by S08)</div></div>
    </div>

    <div class="card" style="margin-bottom:16px;">
      <h3 style="margin-top:0;">1. Exact 10-Student Placement & Salary Package Benchmark (Slide 9 & 11)</h3>
      <p style="font-size:12px; color:var(--text-muted);">Complete student performance and outlier salary error record ($45k anomaly on S08):</p>
      <div id="modC_tableDetail" style="overflow-x:auto;"><div class="loading">Loading table…</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">2. Interactive Huber Loss & F-beta Simulator</h3>
        <p style="font-size:12px; color:var(--text-muted);">Test the smooth transition between quadratic MSE and linear MAE at δ = 1.345:</p>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
          <div><label style="font-size:11px; font-weight:600;">Residual Error e ($k)</label><input type="number" id="modC_in_err" value="4.0" step="0.5" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">Huber Threshold δ</label><input type="number" id="modC_in_delta" value="1.345" step="0.05" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
        </div>
        <button onclick="calculateModCHuber()" class="predict-btn" style="padding:9px; font-size:12px; margin-top:0;">Evaluate Loss Functions</button>
        <div id="modC_huberResult" style="margin-top:12px; padding:12px; background:#f4f6fb; border-radius:8px;">
          <div class="metric-row"><span>Huber Loss L_δ(e)</span><span><b>4.4813</b></span></div>
          <div class="metric-row"><span>MSE Loss (0.5·e²)</span><span>8.0000 (Exploding penalty)</span></div>
          <div class="metric-row"><span>MAE Loss (|e|)</span><span>4.0000 (Constant slope)</span></div>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">3. Self-Assessment Concept Checks (Slide 14)</h3>
        <div id="modC_quizDetail"><div class="loading">Loading quiz…</div></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE D: CONFUSION MATRIX & ROC-AUC (M5-4)                           -->
  <!-- ===================================================================== -->
  <section id="module_d">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module D (M5-4) · Confusion Matrix, Decision Thresholds & ROC-AUC</h2>
        <p class="section-sub">Source: <code>Module_5_Session_M5-4.pptx</code> (15 Slides) · Full 2×2 Matrix, Expected Loss C_FP·FP + C_FN·FN, Threshold Trajectory (≥0.90, ≥0.72, ≥0.50, ≥0.30), Youden's J Optimization, and Pairwise Probabilistic Ranking.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M5-4</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Optimal Threshold (Youden J)</div><div class="stat-value" style="color:var(--accent-teal);">τ* = 0.72</div><div class="stat-foot">Max J = 1.0000 (Slide 10)</div></div>
      <div class="card"><div class="stat-label">Pairwise Ranking AUC</div><div class="stat-value">1.0000</div><div class="stat-foot">25/25 satisfied pairs (Slide 11)</div></div>
      <div class="card"><div class="stat-label">Default Cutoff (τ=0.50)</div><div class="stat-value">90.0%</div><div class="stat-foot">Precision 83.3% (S08 False Alarm)</div></div>
      <div class="card"><div class="stat-label">Conservative (τ=0.90)</div><div class="stat-value" style="color:var(--accent-orange);">70.0%</div><div class="stat-foot">Zero FP, misses 60% of placed</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. Interactive Decision Threshold Slider & Asymmetric Cost Evaluator</h3>
        <p style="font-size:12px; color:var(--text-muted);">Adjust decision threshold τ and error costs to recalculate the 2×2 matrix in real-time:</p>
        <div style="margin-bottom:12px;">
          <label style="font-size:12px; font-weight:600;">Threshold τ: <span id="modD_tauLabel" style="color:var(--accent-teal); font-weight:800;">0.72</span></label>
          <input type="range" id="modD_in_tau" min="0.10" max="0.95" step="0.01" value="0.72" oninput="updateModDInteractive()" style="width:100%; margin-top:6px;">
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
          <div><label style="font-size:11px; font-weight:600;">Cost of False Alarm (C_FP)</label><input type="number" id="modD_in_cfp" value="1.0" step="0.5" oninput="updateModDInteractive()" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">Cost of Miss (C_FN)</label><input type="number" id="modD_in_cfn" value="5.0" step="0.5" oninput="updateModDInteractive()" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
        </div>
        <div id="modD_interactiveCard" style="background:#f4f6fb; padding:12px; border-radius:8px;">
          <div class="metric-row"><span>Confusion Matrix</span><span id="modD_dynCm">TP: 5 | FP: 0 | TN: 5 | FN: 0</span></div>
          <div class="metric-row"><span>Sensitivity (TPR) / Specificity (TNR)</span><span id="modD_dynTpr">1.0000 / 1.0000</span></div>
          <div class="metric-row"><span>Youden's J-Statistic (TPR − FPR)</span><span id="modD_dynYouden"><b>1.0000 (Maximum!)</b></span></div>
          <div class="metric-row"><span>Total Expected Loss</span><span id="modD_dynLoss"><b>$0.00</b></span></div>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. Operational Threshold Trajectory Benchmark (Slide 10)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Complete policy progression across conservative, optimal, default, and permissive cutoffs:</p>
        <div id="modD_trajectoryTable"><div class="loading">Loading trajectory…</div></div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">3. Pairwise Probabilistic Ranking Breakdown (Slide 11)</h3>
        <p style="font-size:12px; color:var(--text-muted);">AUC = P(p̂(X_pos) > p̂(X_neg)) = U / (N_pos × N_neg):</p>
        <div style="background:#e9fbf6; padding:12px; border-radius:8px; font-size:13px; line-height:1.5;">
          <b>Positives (5):</b> S03(0.95), S07(0.91), S01(0.88), S09(0.82), S05(0.72)<br>
          <b>Negatives (5):</b> S08(0.60), S06(0.40), S02(0.25), S10(0.20), S04(0.15)<br><br>
          All 5 positive students have scores strictly greater than all 5 negative students! S05 (lowest pos, p=0.72) > S08 (highest neg, p=0.60).<br>
          <b>Satisfied Pairs:</b> 25 / 25 = 1.00 → <b>ROC-AUC = 1.0000</b>.
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">4. Self-Assessment Concept Checks (Slide 14)</h3>
        <div id="modD_quizDetail"><div class="loading">Loading quiz…</div></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE E: PR CURVES & CLASS IMBALANCE (M5-5)                          -->
  <!-- ===================================================================== -->
  <section id="module_e">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module E (M5-5) · Precision-Recall Curves & Class Imbalance</h2>
        <p class="section-sub">Source: <code>Module_5_Session_M5-5.pptx</code> (15 Slides) · FPR Denominator Dilution Flaw, 100k Sample Dilution Proof, Baseline PR-AUC = π, 10-Student 20% Imbalance Trajectory, and Slide 11 Hand Problem.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M5-5</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Random PR Baseline</div><div class="stat-value" style="color:var(--accent-teal);">π = 20.0%</div><div class="stat-foot">Equal to positive prevalence</div></div>
      <div class="card"><div class="stat-label">Optimal Threshold (Peak F1)</div><div class="stat-value">τ* = 0.90</div><div class="stat-foot">F1: 100% · Precision: 100%</div></div>
      <div class="card"><div class="stat-label">Default τ=0.50 Precision</div><div class="stat-value" style="color:var(--accent-red);">33.3%</div><div class="stat-foot">False alarm rate: 66.7%!</div></div>
      <div class="card"><div class="stat-label">Slide 11 Average Precision</div><div class="stat-value">0.854</div><div class="stat-foot">+0.354 gain over baseline (0.50)</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. The 100,000-Sample FPR Dilution Proof (Slide 3)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Why ROC-AUC is dangerously deceptive when positive prevalence π < 5%:</p>
        <div style="background:#fdeeee; border:1px solid #f6c9c8; padding:14px; border-radius:8px; font-size:13px; line-height:1.5;">
          <b>Setup:</b> 100 Positives, 99,000 Negatives (π = 0.10%).<br>
          <b>Model Hits:</b> 90 True Positives, 990 False Positives.<br>
          • <b>Recall:</b> 90 / 100 = <b>90.0%</b><br>
          • <b>FPR:</b> 990 / 99,000 = <b>1.0%</b> (ROC curve looks world-class!)<br>
          • <b>Precision:</b> 90 / (90 + 990) = <b>8.33%</b> (11 false alarms for every 1 real hit!)<br><br>
          <b>Conclusion:</b> The massive pool of 99,000 true negatives dilutes the FPR denominator, masking high false alarm volume. <b>PR-AUC directly reflects operational reality.</b>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. 10-Student 20% Placement Imbalance Trajectory (Slide 9)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Precision, Recall & F1 trajectory across thresholds (Slide 9):</p>
        <div id="modE_trajectoryTable"><div class="loading">Loading trajectory…</div></div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">3. Practice Problem: Precision-Recall by Hand (Slide 11)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Step-by-step hand calculation with Average Precision (AP) formulation:</p>
        <div id="modE_practiceDetail"><div class="loading">Loading practice solution…</div></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">4. Self-Assessment Concept Checks (Slide 15)</h3>
        <div id="modE_quizDetail"><div class="loading">Loading quiz…</div></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE F: CALIBRATION & BRIER SCORE (M5-6)                            -->
  <!-- ===================================================================== -->
  <section id="module_f">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module F · Probability Calibration, Platt Scaling & Brier Score</h2>
        <p class="section-sub">Source: <code>Probability_Calibration_Placement.pptx</code> (13 Slides) & <code>M5-6_to_M5-8</code> · Trustworthy Probabilities, Platt Sigmoid Scaling, Isotonic Regression PAVA, and 10-Student Brier Score Reduction.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: Calibration</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Student S7 Raw Score</div><div class="stat-value">0.600</div><div class="stat-foot">Log-odds f = ln(1.5) = 0.405</div></div>
      <div class="card"><div class="stat-label">S7 Platt Calibrated</div><div class="stat-value" style="color:var(--accent-teal);">0.819</div><div class="stat-foot">a=1.773, b=0.789 (Slide 6)</div></div>
      <div class="card"><div class="stat-label">Raw Model Brier Score</div><div class="stat-value">0.157</div><div class="stat-foot">10-student illustrative baseline</div></div>
      <div class="card"><div class="stat-label">Calibrated Brier Score</div><div class="stat-value" style="color:var(--accent-teal);">0.143 / 0.100</div><div class="stat-foot">Platt (0.143) · Isotonic (0.100)</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. Interactive Platt Scaling Sigmoid Simulator</h3>
        <p style="font-size:12px; color:var(--text-muted);">Transform any raw probability score through learned parameters a and b:</p>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:12px;">
          <div><label style="font-size:11px; font-weight:600;">Raw Prob p_raw</label><input type="number" id="modF_in_praw" value="0.60" step="0.05" min="0.01" max="0.99" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">Parameter a</label><input type="number" id="modF_in_a" value="1.773" step="0.1" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">Parameter b</label><input type="number" id="modF_in_b" value="0.789" step="0.1" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
        </div>
        <button onclick="calculateModFPlatt()" class="predict-btn" style="padding:9px; font-size:12px; margin-top:0;">Calibrate Probability</button>
        <div id="modF_plattResult" style="margin-top:12px; padding:12px; background:#e9fbf6; border-radius:8px;">
          Log-odds f = ln(0.60 / 0.40) ≈ <b>0.405</b> → Calibrated: <b>0.819</b> (Matches Slide 6!)
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. 5-Student Brier Score Hand Exercise (Slide 10)</h3>
        <p style="font-size:12px; color:var(--text-muted);">BS = (1/N) × Σ (pᵢ − yᵢ)² · Penalizing confident wrong assertions:</p>
        <table>
          <thead><tr><th>Student</th><th>Predicted p</th><th>Actual y</th><th>Squared Error (p − y)²</th></tr></thead>
          <tbody>
            <tr><td>S1</td><td>0.80</td><td>1</td><td>0.04</td></tr>
            <tr><td>S2</td><td>0.20</td><td>0</td><td>0.04</td></tr>
            <tr><td>S3</td><td>0.60</td><td>1</td><td>0.16</td></tr>
            <tr><td>S4</td><td>0.40</td><td>0</td><td>0.16</td></tr>
            <tr><td>S5</td><td>0.70</td><td>1</td><td>0.09</td></tr>
          </tbody>
        </table>
        <div style="margin-top:10px; font-weight:700;">Sum of Squared Errors = 0.49 → Brier Score = 0.49 / 5 = <span style="color:var(--accent-teal);">0.098</span></div>
      </div>
    </div>

    <div class="card" style="margin-bottom:16px;">
      <h3 style="margin-top:0;">3. 10-Student Placement Calibration Benchmark (Slide 4, 7, 9)</h3>
      <p style="font-size:12px; color:var(--text-muted);">Comparing Raw probabilities against Platt Scaling and PAVA Isotonic Regression:</p>
      <div id="modF_tableDetail" style="overflow-x:auto;"><div class="loading">Loading calibration table…</div></div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE G: HYPERPARAMETER SEARCH (M5-7)                                -->
  <!-- ===================================================================== -->
  <section id="module_g">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module G (M5-7) · Hyperparameter Search & Optimization</h2>
        <p class="section-sub">Source: <code>Module_5_Session_M5-6_to_M5-8.pptx</code> (14 Slides) · Grid Search Lattice, Bergstra-Bengio Random Sampling Theorem, Bayesian Optimization (GP Surrogate & Expected Improvement), and Search Budget Problem.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M5-7</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Grid Configurations</div><div class="stat-value">12 lattice</div><div class="stat-foot">4 learning rates × 3 depths</div></div>
      <div class="card"><div class="stat-label">Grid Total Fits</div><div class="stat-value">60 fits</div><div class="stat-foot">12 configs × 5-fold CV</div></div>
      <div class="card"><div class="stat-label">Random Search Top 5%</div><div class="stat-value" style="color:var(--accent-teal);">95.4% hit</div><div class="stat-foot">P(hit) = 1 − 0.95^60 (Slide 10)</div></div>
      <div class="card"><div class="stat-label">95% Conf Sample Budget</div><div class="stat-value">n = 59</div><div class="stat-foot">n ≥ ln(0.05) / ln(0.95)</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. Interactive Search Budget & Confidence Solver (Slide 10)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Calculate exact grid fits and statistical sample count for arbitrary confidence levels:</p>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
          <div><label style="font-size:11px; font-weight:600;">Learning Rates</label><input type="number" id="modG_in_lr" value="4" min="1" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">Tree Depths</label><input type="number" id="modG_in_depth" value="3" min="1" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
        </div>
        <button onclick="calculateModGBudget()" class="predict-btn" style="padding:9px; font-size:12px; margin-top:0;">Solve Search Budget</button>
        <div id="modG_budgetResult" style="margin-top:12px; padding:12px; background:#f4f6fb; border-radius:8px;">
          Grid requires <b>60 model fits</b>. Random search of 60 draws achieves <b>95.4% chance</b> of landing in top 5%. Exact 95% confidence requires <b>n = 59 samples</b>.
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. Bayesian Optimization GP Surrogate Model (Slide 9)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Gaussian Process regressor (Matérn kernel ν=2.5) with Expected Improvement (EI):</p>
        <div id="modG_gpDetail"><div class="loading">Loading GP surrogate trace…</div></div>
      </div>
    </div>

    <div class="card" style="margin-bottom:16px;">
      <h3 style="margin-top:0;">3. Master Comparative Matrix: Hyperparameter Search Paradigms</h3>
      <table>
        <thead><tr><th>Method</th><th>Search Strategy</th><th>Sample Efficiency</th><th>Best Production Use Case</th></tr></thead>
        <tbody>
          <tr><td><b>Grid Search</b></td><td>Exhaustive combinatorial lattice</td><td>Low O(k^d)</td><td>Small parameter spaces (d ≤ 2)</td></tr>
          <tr><td><b>Random Search</b></td><td>Uniform continuous sampling</td><td>High (Bergstra-Bengio)</td><td>General tuning default (d ≥ 3)</td></tr>
          <tr><td><b>Bayesian Opt</b></td><td>GP / TPE surrogate + Expected Improvement</td><td>Maximum</td><td>Expensive training (Deep Learning / XGBoost)</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE H: CLUSTER-AUGMENTED REGRESSION                                 -->
  <!-- ===================================================================== -->
  <section id="module_h">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module H · Cluster-Augmented Regression (5-House Model)</h2>
        <p class="section-sub">Source: <code>Cluster_Augmented_Regression (1).pptx</code> (10 Slides) · Demonstrating why an omitted Location/Cluster variable takes a regression from R² = 0.04 to a perfect R² = 1.00 fit.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: Cluster Regression</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Baseline Size-Only R²</div><div class="stat-value" style="color:var(--accent-red);">0.0400</div><div class="stat-foot">P̂ = 65.5 + 0.175·S (Slide 4)</div></div>
      <div class="card"><div class="stat-label">Augmented Model R²</div><div class="stat-value" style="color:var(--accent-teal);">1.0000</div><div class="stat-foot">Every residual = $0.00 (Slide 9)</div></div>
      <div class="card"><div class="stat-label">Cluster A Centroid</div><div class="stat-value">1.50</div><div class="stat-foot">Houses 1, 2 (Location L=1, 2)</div></div>
      <div class="card"><div class="stat-label">Cluster B Centroid</div><div class="stat-value">8.50</div><div class="stat-foot">Houses 3, 4, 5 (Location L=8, 9, 8.5)</div></div>
    </div>

    <div class="card" style="margin-bottom:16px;">
      <h3 style="margin-top:0;">1. Complete 5-House Dataset & Residual Comparison (Slide 2 & 9)</h3>
      <p style="font-size:12px; color:var(--text-muted);">Comparing baseline model predictions against cluster-augmented regression:</p>
      <div id="modH_tableDetail" style="overflow-x:auto;"><div class="loading">Loading 5-house table…</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">2. Interactive House Price Predictor</h3>
        <p style="font-size:12px; color:var(--text-muted);">Predict price using baseline OLS vs. Cluster-augmented interaction model:</p>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
          <div><label style="font-size:11px; font-weight:600;">Size S (sqft)</label><input type="number" id="modH_in_size" value="1100" step="50" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);"></div>
          <div><label style="font-size:11px; font-weight:600;">Cluster Assignment</label>
            <select id="modH_in_cluster" style="width:100%; padding:6px; border-radius:6px; border:1px solid var(--border);">
              <option value="A">Cluster A (Low Price Corridor)</option>
              <option value="B" selected>Cluster B (Prime Corridor)</option>
            </select>
          </div>
        </div>
        <button onclick="predictModHPrice()" class="predict-btn" style="padding:9px; font-size:12px; margin-top:0;">Calculate Prices</button>
        <div id="modH_predResult" style="margin-top:12px; padding:12px; background:#e9fbf6; border-radius:8px;">
          Baseline: <b>$258.0k</b> | Augmented: <b>$320.0k</b> (Cluster Premium: <b>+$62.0k</b>)
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">3. Mathematical Derivations & Takeaways</h3>
        <div style="font-size:13px; line-height:1.5;">
          <b>Baseline OLS Equation (Slide 4):</b><br>
          <code>P̂ = 65.5 + 0.175·S</code> (R² ≈ 0.04)<br>
          <i>Diagnosis:</i> Systematic negative residuals on Cluster A and positive on Cluster B proves omitted-variable bias.<br><br>
          <b>Augmented Model Equation (Slide 9):</b><br>
          <code>P̂ = 0.15·S + 100·C + 0.05·(S×C)</code> (R² = 1.0)<br>
          <i>Cluster A (C=0):</i> P = 0.15·S ($150/sqft)<br>
          <i>Cluster B (C=1):</i> P = 100 + 0.20·S ($100k fixed premium + $200/sqft)
        </div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE I: K-MEANS CLUSTERING (M4-1)                                    -->
  <!-- ===================================================================== -->
  <section id="module_i">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module I (M4-1) · K-Means Clustering & Lloyd's Algorithm</h2>
        <p class="section-sub">Source: <code>M4-Session1-KMeans_clustering-new.pptx</code> (32 Slides) · WCSS Inertia Objective, Lloyd's 2-Step Iterative Solver, K-Means++ Seeding, Elbow Method, and Slide 24 Hand Silhouette Calculation.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M4-1</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Slide 24 Mean Silhouette</div><div class="stat-value" style="color:var(--accent-teal);">s̄ = 0.801</div><div class="stat-foot">Strong structure (0.71–1.00)</div></div>
      <div class="card"><div class="stat-label">Point A Silhouette s(A)</div><div class="stat-value">0.834</div><div class="stat-foot">a(A)=1.000, b(A)=6.030</div></div>
      <div class="card"><div class="stat-label">K-Means++ Bound</div><div class="stat-value">O(log K)</div><div class="stat-foot">Expected error vs optimal</div></div>
      <div class="card"><div class="stat-label">Lloyd's Convergence</div><div class="stat-value">Finite</div><div class="stat-foot">Monotonic WCSS reduction</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. Slide 24 Hand-Calculated Silhouette Table</h3>
        <p style="font-size:12px; color:var(--text-muted);">Hand calculation for C₁={A(1,1), B(2,1), C(1,2)} and C₂={D(5,5), E(6,5)}:</p>
        <div id="modI_handTable"><div class="loading">Loading silhouette table…</div></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. 9-Point Outer Loop Elbow & Marginal Gain Trace (Slide 18)</h3>
        <p style="font-size:12px; color:var(--text-muted);">WCSS Inertia curve and marginal gain reduction across K=2 through 5:</p>
        <div id="modI_elbowTable"><div class="loading">Loading elbow trace…</div></div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">3. Comparison: Three Techniques, One Pipeline (Slide 27)</h3>
        <table>
          <thead><tr><th>Technique</th><th>Core Purpose</th><th>Metric</th><th>Computational Cost</th></tr></thead>
          <tbody>
            <tr><td><b>Lloyd's Alg</b></td><td>Optimize assignments for fixed K</td><td>WCSS J</td><td>O(n·K·d) per iteration</td></tr>
            <tr><td><b>Elbow Method</b></td><td>Select optimal K via bend</td><td>WCSS vs K curve</td><td>Run K_max times</td></tr>
            <tr><td><b>Silhouette Score</b></td><td>Validate cluster cohesion & separation</td><td>s(i) ∈ [−1, +1]</td><td>O(n²) full pairwise</td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">4. Quick-Fire Viva Questions (Slide 31)</h3>
        <div id="modI_vivaDetail"><div class="loading">Loading viva questions…</div></div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- MODULE J: HIERARCHICAL CLUSTERING (M4-2)                              -->
  <!-- ===================================================================== -->
  <section id="module_j">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
      <div>
        <h2 class="section-title">Module J (M4-2) · Hierarchical Agglomerative Clustering</h2>
        <p class="section-sub">Source: <code>M4-Session2-Hierarchical_Clustering-new.pptx</code> (21 Slides) · Bottom-Up HAC, 4 Linkage Methods, Lance-Williams Updates, 5-Point Distance Matrix, Ward Merge Sequence, and Retail Store Customer Segmentation.</p>
      </div>
      <span class="pill good" style="font-size:13px; padding:6px 12px;">Authoritative Deck: M4-2</span>
    </div>

    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card"><div class="stat-label">Running Points</div><div class="stat-value">5 Points</div><div class="stat-foot">A=2, B=4, C=10, D=12, E=18</div></div>
      <div class="card"><div class="stat-label">Optimal Ward Cut</div><div class="stat-value" style="color:var(--accent-teal);">K = 2</div><div class="stat-foot">Largest gap: 5.72 → 11.32 (gap 5.60)</div></div>
      <div class="card"><div class="stat-label">Ward First Merges</div><div class="stat-value">h = 1.41</div><div class="stat-foot">(A,B) and (C,D) at h=1.41</div></div>
      <div class="card"><div class="stat-label">Retail Case Study</div><div class="stat-value">2 Segments</div><div class="stat-foot">P1-P3 (Casual) vs P4-P6 (VIP)</div></div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">1. Pairwise Distance Matrix on 5 Points (Slide 6)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Exact distance matrix D[i,j] = |x_i − x_j| for A=2, B=4, C=10, D=12, E=18:</p>
        <div id="modJ_distMatrix"><div class="loading">Loading distance matrix…</div></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">2. Ward Linkage Merge Sequence & Optimal Cut (Slide 10 & 13)</h3>
        <p style="font-size:12px; color:var(--text-muted);">Step-by-step merge heights and cut partitions:</p>
        <div id="modJ_wardMerges"><div class="loading">Loading Ward sequence…</div></div>
      </div>
    </div>

    <div class="grid grid-2" style="margin-bottom:16px;">
      <div class="card">
        <h3 style="margin-top:0;">3. Retail Customer Segmentation Case Study (Slide 19)</h3>
        <p style="font-size:12px; color:var(--text-muted);">6 customers: P1(10,2), P2(12,3), P3(11,2), P4(60,15), P5(62,14), P6(58,16):</p>
        <div id="modJ_retailDetail"><div class="loading">Loading retail case study…</div></div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">4. Viva-Voce Questions (Slide 20)</h3>
        <div id="modJ_vivaDetail"><div class="loading">Loading viva questions…</div></div>
      </div>
    </div>
  </section>
"""

# Insert the sections right before </div><!-- main -->
main_close_tag = "</div>\n\n<script>"
if main_close_tag in content:
    content = content.replace(main_close_tag, modules_sections_html + "\n</div>\n\n<script>")
    print("Inserted all 10 module sections into index.html.")
else:
    print("Warning: could not locate main_close_tag.")

# 3. Add JS functions for Modules A through J
js_code = """
// =====================================================================
// MODULES A THROUGH J LOADER & INTERACTIVE CONTROLLERS
// =====================================================================

async function loadModuleA() {
  try {
    const res = await fetch(API + "/api/curriculum/module_a").then(r => r.json());
    document.getElementById("modA_cvAcc").textContent = Math.round(res.mean_cv_accuracy * 100) + "%";
    document.getElementById("modA_testAcc").textContent = Math.round(res.test_accuracy * 100) + "%";

    let mHtml = `<table><thead><tr><th>Student</th><th>Status</th><th>Fold 1</th><th>Fold 2</th><th>Fold 3</th><th>Fold 4</th></tr></thead><tbody>`;
    const fMap = [
      ["S01", "Placed", "TRAIN", "VAL", "TRAIN", "TRAIN"],
      ["S02", "Not Placed", "TRAIN", "VAL", "TRAIN", "TRAIN"],
      ["S03", "Placed", "TRAIN", "TRAIN", "TRAIN", "VAL"],
      ["S04", "Not Placed", "VAL", "TRAIN", "TRAIN", "TRAIN"],
      ["S05", "Placed", "TRAIN", "TRAIN", "TRAIN", "VAL"],
      ["S06", "Not Placed", "VAL", "TRAIN", "TRAIN", "TRAIN"],
      ["S07", "Placed", "TRAIN", "TRAIN", "VAL", "TRAIN"],
      ["S08", "Not Placed", "TRAIN", "TRAIN", "VAL", "TRAIN"],
    ];
    fMap.forEach(r => {
      mHtml += `<tr><td><b>${r[0]}</b></td><td><span class="pill ${r[1]==='Placed'?'good':'bad'}">${r[1]}</span></td>
        <td>${r[2]==='VAL'?'<b>VAL</b>':'Train'}</td><td>${r[3]==='VAL'?'<b>VAL</b>':'Train'}</td>
        <td>${r[4]==='VAL'?'<b>VAL</b>':'Train'}</td><td>${r[5]==='VAL'?'<b>VAL</b>':'Train'}</td></tr>`;
    });
    mHtml += `</tbody></table>`;
    document.getElementById("modA_matrixDetail").innerHTML = mHtml;

    let tHtml = `<table><thead><tr><th>Student</th><th>Actual Status</th><th>Predicted Probability</th><th>Classification</th></tr></thead><tbody>`;
    res.test_predictions.forEach(p => {
      tHtml += `<tr><td><b>${p.student}</b></td><td><span class="pill ${p.actual===1?'good':'bad'}">${p.actual===1?'Placed':'Not Placed'}</span></td>
        <td><b>${p.prob}</b></td><td><span class="pill good">CORRECT (100%)</span></td></tr>`;
    });
    tHtml += `</tbody></table>`;
    tHtml += `<div style="margin-top:10px; font-size:12px; color:var(--text-muted);"><b>Learned Weights:</b> CGPA: +0.842 · Coding: +0.651 · Aptitude: +0.518 · Internship: +0.380 · Projects: +0.290</div>`;
    document.getElementById("modA_testDetail").innerHTML = tHtml;

    let qHtml = "";
    res.quiz.forEach((q, i) => {
      qHtml += `<div style="margin-bottom:10px; padding:10px; background:#f4f6fb; border-radius:8px;">
        <div style="font-weight:700; font-size:13px;">Q${i+1}: ${q.q}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;"><b>Answer:</b> ${q.a}</div>
      </div>`;
    });
    document.getElementById("modA_quizDetail").innerHTML = qHtml;
  } catch (e) { console.error(e); }
}

async function loadModuleB() {
  try {
    const res = await fetch(API + "/api/curriculum/module_b").then(r => r.json());
    document.getElementById("modB_totalFits").textContent = res.n_fits_calculation.total_fits + " fits";
    document.getElementById("modB_nestedF1").textContent = res.slide_reported_benchmark.reported_mean_f1.toFixed(4);

    let pHtml = `<table><thead><tr><th>Fold</th><th>Validation Size</th><th>Placed Count</th><th>Not Placed</th><th>Class Ratio</th></tr></thead><tbody>`;
    res.fold_allocation_table.forEach(f => {
      pHtml += `<tr><td><b>Fold ${f.fold}</b></td><td>${f.total_val}</td><td>${f.placed_count} (${f.placed_pct}%)</td><td>${f.not_placed_count} (${f.not_placed_pct}%)</td><td><span class="pill good">PRESERVED</span></td></tr>`;
    });
    pHtml += `</tbody></table>`;
    document.getElementById("modB_proportionsTable").innerHTML = pHtml;

    let trHtml = `
      <div class="metric-row"><span>Outer Folds F1 Scores</span><span>${res.slide_reported_benchmark.outer_scores.join(" · ")}</span></div>
      <div class="metric-row"><span>Unbiased Nested F1 Mean</span><span><b>${res.slide_reported_benchmark.reported_mean_f1} (±${res.slide_reported_benchmark.reported_margin})</b></span></div>
      <div class="metric-row"><span>Inner Grid Search Optimization</span><span>Tested C ∈ {0.01, 0.1, 1.0, 10.0} → Selected C* = <b>1.00</b></span></div>
      <div class="metric-row" style="background:#e9fbf6; padding:8px; border-radius:6px;"><span>Selection Bias Eliminated</span><span>Outer test sets were strictly held out during inner tuning.</span></div>`;
    document.getElementById("modB_traceDetail").innerHTML = trHtml;

    let qHtml = "";
    res.quiz.forEach((q, i) => {
      qHtml += `<div style="margin-bottom:10px; padding:10px; background:#f4f6fb; border-radius:8px;">
        <div style="font-weight:700; font-size:13px;">Q${i+1}: ${q.q}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;"><b>Answer:</b> ${q.a}</div>
      </div>`;
    });
    document.getElementById("modB_quizDetail").innerHTML = qHtml;
  } catch (e) { console.error(e); }
}

async function calculateModBFits() {
  const kout = parseInt(document.getElementById("modB_in_kout").value) || 5;
  const kin = parseInt(document.getElementById("modB_in_kin").value) || 3;
  const grid = parseInt(document.getElementById("modB_in_grid").value) || 10;
  const res = await fetch(API + `/api/curriculum/interactive/module_b?k_outer=${kout}&k_inner=${kin}&grid_size=${grid}`).then(r => r.json());
  document.getElementById("modB_fitsResult").innerHTML = `${kout} × (${kin} × ${grid} + 1) = <b>${res.total_model_fits} total model fits</b>.`;
}

async function loadModuleC() {
  try {
    const res = await fetch(API + "/api/curriculum/module_c").then(r => r.json());
    let tHtml = `<table><thead><tr><th>Student</th><th>CGPA</th><th>Coding</th><th>True Placed</th><th>Pred Prob</th><th>Pred Placed</th><th>True Salary</th><th>Pred Salary</th><th>Abs Error</th></tr></thead><tbody>`;
    res.case_study_table.forEach(s => {
      const isOutlier = s.Student === "S08";
      tHtml += `<tr style="${isOutlier ? 'background:#fff3e0;' : ''}">
        <td><b>${s.Student}</b></td><td>${s.CGPA}</td><td>${s.Coding}</td>
        <td><span class="pill ${s.TruePlaced===1?'good':'bad'}">${s.TruePlaced===1?'Placed':'Not'}</span></td>
        <td>${s.PredProb}</td>
        <td><span class="pill ${s.PredPlaced===1?'good':'bad'}">${s.PredPlaced===1?'Placed':'Not'}</span></td>
        <td>$${s.TrueSalary}k</td><td>$${s.PredSalary}k</td>
        <td><b>$${s.AbsError}k</b> ${isOutlier?'⚠️ OUTLIER':''}</td>
      </tr>`;
    });
    tHtml += `</tbody></table>`;
    document.getElementById("modC_tableDetail").innerHTML = tHtml;

    let qHtml = "";
    res.quiz.forEach((q, i) => {
      qHtml += `<div style="margin-bottom:10px; padding:10px; background:#f4f6fb; border-radius:8px;">
        <div style="font-weight:700; font-size:13px;">Q${i+1}: ${q.q}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;"><b>Answer:</b> ${q.a}</div>
      </div>`;
    });
    document.getElementById("modC_quizDetail").innerHTML = qHtml;
  } catch (e) { console.error(e); }
}

async function calculateModCHuber() {
  const err = parseFloat(document.getElementById("modC_in_err").value) || 4.0;
  const delta = parseFloat(document.getElementById("modC_in_delta").value) || 1.345;
  const res = await fetch(API + `/api/curriculum/interactive/module_c?residual=${err}&delta=${delta}`).then(r => r.json());
  document.getElementById("modC_huberResult").innerHTML = `
    <div class="metric-row"><span>Huber Loss L_δ(e)</span><span><b>${res.huber_loss}</b></span></div>
    <div class="metric-row"><span>MSE Quadratic Loss (0.5·e²)</span><span>${res.squared_error_loss}</span></div>
    <div class="metric-row"><span>MAE Linear Loss (|e|)</span><span>${res.absolute_error_loss}</span></div>`;
}

async function loadModuleD() {
  try {
    const res = await fetch(API + "/api/curriculum/module_d").then(r => r.json());
    let tHtml = `<table><thead><tr><th>Threshold τ</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th><th>TPR</th><th>FPR</th><th>Precision</th><th>Accuracy</th><th>Policy Note</th></tr></thead><tbody>`;
    const notes = {
      0.9: "Conservative. Zero false alarms, misses 60% of placed.",
      0.72: "Optimal Youden's J (J=1.00). Perfect separation!",
      0.5: "Default cutoff. Perfect recall, but S08 false alarm.",
      0.3: "Permissive. Captures all placed, FP rises to 2."
    };
    res.trajectory_table.forEach(r => {
      const isOpt = r.threshold === 0.72;
      tHtml += `<tr style="${isOpt ? 'background:#e9fbf6;' : ''}">
        <td><b>≥ ${r.threshold}</b> ${isOpt?'★':''}</td><td>${r.TP}</td><td>${r.FP}</td><td>${r.TN}</td><td>${r.FN}</td>
        <td>${(r.TPR*100).toFixed(0)}%</td><td>${(r.FPR*100).toFixed(0)}%</td><td>${(r.Precision*100).toFixed(1)}%</td><td>${(r.Accuracy*100).toFixed(0)}%</td>
        <td>${notes[r.threshold] || ''}</td>
      </tr>`;
    });
    tHtml += `</tbody></table>`;
    document.getElementById("modD_trajectoryTable").innerHTML = tHtml;

    let qHtml = "";
    res.quiz.forEach((q, i) => {
      qHtml += `<div style="margin-bottom:10px; padding:10px; background:#f4f6fb; border-radius:8px;">
        <div style="font-weight:700; font-size:13px;">Q${i+1}: ${q.q}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;"><b>Answer:</b> ${q.a}</div>
      </div>`;
    });
    document.getElementById("modD_quizDetail").innerHTML = qHtml;
  } catch (e) { console.error(e); }
}

async function updateModDInteractive() {
  const tau = parseFloat(document.getElementById("modD_in_tau").value);
  const cfp = parseFloat(document.getElementById("modD_in_cfp").value) || 1.0;
  const cfn = parseFloat(document.getElementById("modD_in_cfn").value) || 5.0;
  document.getElementById("modD_tauLabel").textContent = tau.toFixed(2);
  const res = await fetch(API + `/api/curriculum/interactive/module_d?threshold=${tau}&cost_fp=${cfp}&cost_fn=${cfn}`).then(r => r.json());
  document.getElementById("modD_dynCm").textContent = `TP: ${res.TP} | FP: ${res.FP} | TN: ${res.TN} | FN: ${res.FN}`;
  document.getElementById("modD_dynTpr").textContent = `${res.TPR} / ${(1-res.FPR).toFixed(4)}`;
  document.getElementById("modD_dynYouden").innerHTML = `<b>${res.Youden_J.toFixed(4)}</b> (Dist to corner: ${res.DistToCorner.toFixed(4)})`;
  document.getElementById("modD_dynLoss").innerHTML = `<b>$${res.ExpectedLoss.toFixed(2)}</b>`;
}

async function loadModuleE() {
  try {
    const res = await fetch(API + "/api/curriculum/module_e").then(r => r.json());
    let tHtml = `<table><thead><tr><th>Cutoff τ</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Imbalance Impact</th></tr></thead><tbody>`;
    const notes = {
      0.9: "Optimal Cutoff (τ*). Perfect Precision & Recall (Zero FP).",
      0.8: "Precision drops to 50% as S01 and S09 false alarms enter.",
      0.7: "Precision degrades further due to S05 false alarm.",
      0.5: "Default Cutoff Trap! 66.7% false alarm rate in positive predictions!",
      0.2: "Approaching random baseline prevalence (π = 20.0%)."
    };
    res.trajectory_table.forEach(r => {
      const isOpt = r.threshold === 0.90;
      tHtml += `<tr style="${isOpt ? 'background:#e9fbf6;' : ''}">
        <td><b>≥ ${r.threshold.toFixed(2)}</b> ${isOpt?'★':''}</td><td>${r.TP}</td><td>${r.FP}</td><td>${r.TN}</td><td>${r.FN}</td>
        <td><b>${r.Precision}%</b></td><td>${r.Recall}%</td><td>${r.F1}%</td>
        <td>${notes[r.threshold] || ''}</td>
      </tr>`;
    });
    tHtml += `</tbody></table>`;
    document.getElementById("modE_trajectoryTable").innerHTML = tHtml;

    const sol = res.practice_problem_solution;
    document.getElementById("modE_practiceDetail").innerHTML = `
      <div class="metric-row"><span>Random Baseline (π)</span><span>4 / 8 = <b>${sol.baseline_prevalence}</b></span></div>
      <div class="metric-row"><span>Cutoff 0.55</span><span>TP=${sol.cutoff_0_55.TP}, FP=${sol.cutoff_0_55.FP} → Precision: <b>${sol.cutoff_0_55.precision}</b>, Recall: <b>${sol.cutoff_0_55.recall}</b></span></div>
      <div class="metric-row"><span>Cutoff 0.35</span><span>TP=${sol.cutoff_0_35.TP}, FP=${sol.cutoff_0_35.FP} → Precision: <b>${sol.cutoff_0_35.precision}</b>, Recall: <b>${sol.cutoff_0_35.recall}</b></span></div>
      <div class="metric-row" style="background:#e9fbf6; padding:8px; border-radius:6px;"><span>Average Precision (AP)</span><span>(1.0 + 1.0 + 0.75 + 0.667)/4 = <b>${sol.average_precision}</b> (Gain: +${sol.gain_over_baseline})</span></div>`;

    let qHtml = "";
    res.quiz.forEach((q, i) => {
      qHtml += `<div style="margin-bottom:10px; padding:10px; background:#f4f6fb; border-radius:8px;">
        <div style="font-weight:700; font-size:13px;">Q${i+1}: ${q.q}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;"><b>Answer:</b> ${q.a}</div>
      </div>`;
    });
    document.getElementById("modE_quizDetail").innerHTML = qHtml;
  } catch (e) { console.error(e); }
}

async function loadModuleF() {
  try {
    const res = await fetch(API + "/api/curriculum/module_f").then(r => r.json());
    let tHtml = `<table><thead><tr><th>Student</th><th>CGPA</th><th>Aptitude</th><th>Actual y</th><th>Raw p</th><th>Platt p_cal</th><th>Isotonic p</th></tr></thead><tbody>`;
    res.calibration_table.forEach(s => {
      tHtml += `<tr><td><b>${s.Student}</b></td><td>${s.CGPA}</td><td>${s.Aptitude}</td>
        <td><span class="pill ${s.ActualY===1?'good':'bad'}">${s.ActualY===1?'Placed':'Not'}</span></td>
        <td>${s.RawP}</td><td><b>${s.PlattP}</b></td><td><b>${s.IsotonicP}</b></td></tr>`;
    });
    tHtml += `</tbody></table>`;
    tHtml += `<div style="margin-top:10px; font-size:12px; color:var(--text-muted);"><b>Brier Scores:</b> Raw = 0.157 → Platt = 0.143 → Isotonic = 0.100 (Lower is better!)</div>`;
    document.getElementById("modF_tableDetail").innerHTML = tHtml;
  } catch (e) { console.error(e); }
}

async function calculateModFPlatt() {
  const p = parseFloat(document.getElementById("modF_in_praw").value) || 0.60;
  const a = parseFloat(document.getElementById("modF_in_a").value) || 1.773;
  const b = parseFloat(document.getElementById("modF_in_b").value) || 0.789;
  const res = await fetch(API + `/api/curriculum/interactive/module_f?raw_probability=${p}&param_a=${a}&param_b=${b}`).then(r => r.json());
  document.getElementById("modF_plattResult").innerHTML = `Raw: <b>${res.raw_probability}</b> → Log-odds f = ln(p/(1-p)) → Calibrated: <b>${res.calibrated_probability}</b> (Platt Sigmoid)`;
}

async function loadModuleG() {
  try {
    const res = await fetch(API + "/api/curriculum/module_g").then(r => r.json());
    let gpHtml = `<table><thead><tr><th>Tree Depth</th><th>Predicted Mean Acc</th><th>Uncertainty σ</th><th>Expected Improvement</th></tr></thead><tbody>`;
    res.bayesian_optimization_gp.gp_predictions_table.forEach(g => {
      gpHtml += `<tr><td>Depth ${g.depth}</td><td>${g.predicted_mean_acc}</td><td>±${g.uncertainty_sigma}</td><td><b>${g.expected_improvement}</b></td></tr>`;
    });
    gpHtml += `</tbody></table>`;
    gpHtml += `<div style="margin-top:10px; font-weight:700; color:var(--accent-teal);">Next Recommended Candidate: Depth = ${res.bayesian_optimization_gp.next_recommended_depth_to_evaluate}</div>`;
    document.getElementById("modG_gpDetail").innerHTML = gpHtml;
  } catch (e) { console.error(e); }
}

async function calculateModGBudget() {
  const lr = parseInt(document.getElementById("modG_in_lr").value) || 4;
  const d = parseInt(document.getElementById("modG_in_depth").value) || 3;
  const res = await fetch(API + `/api/curriculum/interactive/module_g?n_lr=${lr}&n_depths=${d}`).then(r => r.json());
  document.getElementById("modG_budgetResult").innerHTML = `
    Grid requires <b>${res.grid_model_fits} model fits</b> (${res.grid_configurations} configs × 5 folds).<br>
    Random search of 60 draws achieves <b>${(res.random_60_p_hit*100).toFixed(1)}% chance</b> of landing in top 5%.<br>
    Exact 95% confidence requires <b>n = ${res.samples_for_95_pct_confidence} samples</b>.`;
}

async function loadModuleH() {
  try {
    const res = await fetch(API + "/api/curriculum/module_h").then(r => r.json());
    let tHtml = `<table><thead><tr><th>House</th><th>Location L</th><th>Size S (sqft)</th><th>Actual Price</th><th>Cluster</th><th>Baseline Pred</th><th>Baseline Residual</th><th>Augmented Pred</th><th>Augmented Residual</th></tr></thead><tbody>`;
    res.comparison_table.forEach(h => {
      tHtml += `<tr>
        <td><b>House ${h.House}</b></td><td>${h.Location_L}</td><td>${h.Size_S}</td><td>$${h.Actual_Price}k</td>
        <td><span class="pill ${h.Cluster==='A'?'good':'bad'}">Cluster ${h.Cluster}</span></td>
        <td>$${h.Baseline_Pred}k</td><td style="color:${h.Baseline_Residual<0?'var(--accent-red)':'var(--accent-teal)'};"><b>${h.Baseline_Residual}k</b></td>
        <td>$${h.Augmented_Pred}k</td><td><b>$0.0k (0.0)</b></td>
      </tr>`;
    });
    tHtml += `</tbody></table>`;
    document.getElementById("modH_tableDetail").innerHTML = tHtml;
  } catch (e) { console.error(e); }
}

async function predictModHPrice() {
  const s = parseFloat(document.getElementById("modH_in_size").value) || 1100;
  const c = document.getElementById("modH_in_cluster").value;
  const res = await fetch(API + `/api/curriculum/interactive/module_h?size=${s}&cluster=${c}`).then(r => r.json());
  document.getElementById("modH_predResult").innerHTML = `
    Baseline (Size only): <b>$${res.baseline_predicted_price_k}k</b> | Augmented (Size + Cluster): <b>$${res.augmented_predicted_price_k}k</b><br>
    Cluster Premium / Delta: <b>+$${res.difference_k}k</b>`;
}

async function loadModuleI() {
  try {
    const res = await fetch(API + "/api/curriculum/module_i").then(r => r.json());
    let hHtml = `<table><thead><tr><th>Point</th><th>Coords</th><th>Cluster</th><th>Cohesion a(i)</th><th>Separation b(i)</th><th>Silhouette s(i)</th></tr></thead><tbody>`;
    res.slide24_hand_calculation.points_table.forEach(p => {
      hHtml += `<tr><td><b>${p.Point}</b></td><td>${p.Coordinates}</td><td>${p.Cluster}</td><td>${p.a_i}</td><td>${p.b_i}</td><td><b>${p.s_i}</b></td></tr>`;
    });
    hHtml += `</tbody></table>`;
    hHtml += `<div style="margin-top:10px; font-weight:700;">Mean Silhouette s̄ = <span style="color:var(--accent-teal);">${res.slide24_hand_calculation.mean_silhouette_s_bar}</span> (${res.slide24_hand_calculation.interpretation})</div>`;
    document.getElementById("modI_handTable").innerHTML = hHtml;

    let eHtml = `<table><thead><tr><th>K</th><th>WCSS (Inertia)</th><th>Marginal Gain</th><th>Silhouette Score</th></tr></thead><tbody>`;
    res.elbow_and_silhouette_trace.forEach(k => {
      eHtml += `<tr><td><b>K = ${k.K}</b></td><td>${k.WCSS_Inertia}</td><td>−${k.Marginal_Gain_Reduction}</td><td><b>${k.Silhouette_Score}</b></td></tr>`;
    });
    eHtml += `</tbody></table>`;
    document.getElementById("modI_elbowTable").innerHTML = eHtml;

    let vHtml = "";
    res.viva_questions.forEach((v, i) => {
      vHtml += `<div style="margin-bottom:10px; padding:10px; background:#f4f6fb; border-radius:8px;">
        <div style="font-weight:700; font-size:13px;">V${i+1}: ${v.q}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;"><b>Answer:</b> ${v.a}</div>
      </div>`;
    });
    document.getElementById("modI_vivaDetail").innerHTML = vHtml;
  } catch (e) { console.error(e); }
}

async function loadModuleJ() {
  try {
    const res = await fetch(API + "/api/curriculum/module_j").then(r => r.json());
    let dHtml = `<table><thead><tr><th>Point</th><th>A (2)</th><th>B (4)</th><th>C (10)</th><th>D (12)</th><th>E (18)</th></tr></thead><tbody>`;
    const labels = ["A", "B", "C", "D", "E"];
    labels.forEach(l1 => {
      dHtml += `<tr><td><b>${l1}</b></td>`;
      labels.forEach(l2 => {
        dHtml += `<td>${res.distance_matrix[l1][l2]}</td>`;
      });
      dHtml += `</tr>`;
    });
    dHtml += `</tbody></table>`;
    document.getElementById("modJ_distMatrix").innerHTML = dHtml;

    let wHtml = `<table><thead><tr><th>Step</th><th>Joined Subtrees</th><th>Merge Height h</th><th>Cluster Size</th></tr></thead><tbody>`;
    res.ward_merges.merge_sequence.forEach(m => {
      wHtml += `<tr><td>Step ${m.step}</td><td><b>${m.join}</b></td><td><b>h = ${m.height}</b></td><td>${m.cluster_size}</td></tr>`;
    });
    wHtml += `</tbody></table>`;
    wHtml += `<div class="warn-box" style="margin-top:10px;">✂️ <b>Optimal Cut:</b> Largest gap is between h=5.72 and h=11.32 (gap = 5.60) → Cut at h=8.0 yields <b>K = 2 clusters: {A,B} and {C,D,E}</b>.</div>`;
    document.getElementById("modJ_wardMerges").innerHTML = wHtml;

    let rHtml = `<table><thead><tr><th>Customer ID</th><th>Annual Spend (₹'000)</th><th>Visits/Month</th><th>Assigned Segment</th></tr></thead><tbody>`;
    res.retail_case_study.customers.forEach(c => {
      rHtml += `<tr><td><b>${c.id}</b></td><td>₹${c.annual_spend_k}k</td><td>${c.visits_pm}</td><td><span class="pill ${c.segment===1?'good':'bad'}">Segment ${c.segment}</span></td></tr>`;
    });
    rHtml += `</tbody></table>`;
    rHtml += `<p style="font-size:12px; color:var(--text-muted); margin-top:8px;">${res.retail_case_study.recommendation}</p>`;
    document.getElementById("modJ_retailDetail").innerHTML = rHtml;

    let vHtml = "";
    res.viva_questions.forEach((v, i) => {
      vHtml += `<div style="margin-bottom:10px; padding:10px; background:#f4f6fb; border-radius:8px;">
        <div style="font-weight:700; font-size:13px;">V${i+1}: ${v.q}</div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;"><b>Answer:</b> ${v.a}</div>
      </div>`;
    });
    document.getElementById("modJ_vivaDetail").innerHTML = vHtml;
  } catch (e) { console.error(e); }
}
"""

# Append JS loader calls inside goToTab
old_gototab = """  if (target === "m6_curriculum") loadM6();
}"""

new_gototab = """  if (target === "m6_curriculum") loadM6();
  if (target === "module_a") loadModuleA();
  if (target === "module_b") loadModuleB();
  if (target === "module_c") loadModuleC();
  if (target === "module_d") loadModuleD();
  if (target === "module_e") loadModuleE();
  if (target === "module_f") loadModuleF();
  if (target === "module_g") loadModuleG();
  if (target === "module_h") loadModuleH();
  if (target === "module_i") loadModuleI();
  if (target === "module_j") loadModuleJ();
}"""

if old_gototab in content:
    content = content.replace(old_gototab, new_gototab)
    print("Updated goToTab function.")
else:
    print("Warning: old goToTab not matched.")

# Append js_code right before </script>
content = content.replace("</script>", js_code + "\n</script>")

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Successfully updated {INDEX_PATH} with complete UI for Modules A through J.")
