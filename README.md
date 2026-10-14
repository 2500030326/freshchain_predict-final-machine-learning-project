# FreshPredict — Flask edition

Covers exactly four session topics on the fruit storage-quality dataset:

| Session | Topic | Script |
|---|---|---|
| M1 · 5-8 | Data loading & EDA | `src/eda.py` |
| M2 · 9-10 | Feature Scaling & Standardization | `src/scaling.py` |
| M2 · 11-12 | Encoding Categorical Features | `src/encoding.py` |
| M2 · 13-14 | Handling Missing Values | `src/missing_values.py` |
| Regression | Continuous Environmental Temperature Modeling (OLS) | `src/linear_regression.py` |
| Regression | Binary Spoilage Classification (Logit & Odds Ratios) | `src/logistic_regression.py` |
| M3 · 1-8 | Decision Trees, Impurity (Entropy/Gini), MSE & Pruning | `src/decision_tree.py` |
| M3 · 9-14 | Random Forest, Bagging, OOB & Majority Voting | `src/random_forest.py` |
| M3 · 15-16 | Model Comparison & Test Set Evaluation | `src/model_comparison.py` |
| M3 · 18 | Production Spoilage Prediction API | `src/predict.py` |

Plus `src/train_model.py`, which trains all candidate models (Entropy DT, Gini DT,
Pre-pruned DT, Cost-Complexity Pruned DT, Random Forest, Logistic Regression), selects the
champion, and `app.py`, a Flask app that serves a real-time dashboard over all of it.

## How to run

```bash
cd freshpredict2
pip install -r requirements.txt

python src/eda.py       # REQUIRED FIRST — generates EDA plots/report
python app.py            # starts the dashboard (auto-trains model on first run)
```

Then open **http://127.0.0.1:5000** in your browser.

`app.py` automatically trains the model on first launch if none exists yet
— you don't need to run `train_model.py` separately, though you can
(`python src/train_model.py`) if you want to retrain after changing data.

Optionally, also run these before starting `app.py` so their tabs are
populated (otherwise the dashboard tells you which script to run):

```bash
python src/scaling.py
python src/encoding.py
python src/missing_values.py
python src/linear_regression.py   # Generates Linear Regression analysis & residual plots
python src/logistic_regression.py # Generates Logistic Regression analysis, ROC, & odds ratios
python src/decision_tree.py       # Generates Decision Tree & Pruning reports
python src/random_forest.py       # Generates Random Forest & OOB reports
python src/model_comparison.py    # Generates Model Comparison & Confusion Matrices
```

## The dashboard

- **Overview** — live stats + charts, auto-refreshing every 8s
- **Dataset** — preview table of cleaned data
- **EDA** — data structure, quality checks, univariate/bivariate/multivariate plots
- **Feature Scaling** — Min-Max vs Standard vs Robust, before/after table, KNN accuracy experiment
- **Encoding** — One-Hot, Ordinal (shown as a pitfall), leak-free Target encoding, embedding preview
- **Missing Values** — synthetic MCAR injection + deletion/mean/median/KNN/indicator strategy comparison
- **Linear Regression** — Continuous temperature OLS modeling, R², RMSE, residual diagnostics, coefficients
- **Logistic Regression** — Binary spoilage classification, Sigmoid curve, ROC-AUC, Odds Ratios (e^β), threshold tuning
- **Predict** — live form → real model inference
- **Model Info** — test metrics for every candidate model
- **Live Feed** — auto-refreshes every 2 seconds, showing predictions as they happen (real-time)
- **Decision Tree & Pruning** — Root entropy, Info Gain, Gini index, MSE demo, Pre-pruning & Cost-Complexity (Weakest-Link) pruning curves, tree visualization
- **Random Forest** — Bootstrap sampling statistics, Random feature selection, Majority voting breakdown on test samples, OOB evaluation, feature importance
- **Model Comparison** — Held-out test set benchmark table, performance comparison chart, confusion matrices for all 5 architectures

## Notes

- This dataset has no real missing values, so `missing_values.py`
  synthetically injects ~8% MCAR gaps to demonstrate and compare the four
  strategies — exactly how the session deck validates each technique
  before recommending it. This is explained on the dashboard's Missing
  Values tab.
- Flask's dev server is used (`app.run()`), appropriate for coursework/demo
  use. For production you'd swap in gunicorn/waitress — not needed here.
