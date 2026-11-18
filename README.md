# Predicting 30-Day Customer Liquidity Stress

### Business Objective
Forecast whether mobile money customers (`N = 40,000` training records) will experience short-term liquidity stress within the next 30 days (`liquidity_stress_next_30d`), and generate calibrated risk scores for `30,000` unseen test customers.

### Methodology & Feature Engineering
* **Exploratory Audit:** Analyzed 183 demographic, activity, and 6-month balance features (`85%` stable vs. `15%` stressed), uncovering a sharp `M4`-to-`M3` balance depletion cliff.
* **Domain Feature Engineering:** Created 20 trajectory, volatility, and cash-burn features (`202` total predictors), including `bal_drop_m3_m4`, `bal_ratio_m1_m6`, `bal_cv_6m`, and `m1_tx_to_bal_ratio`.
* **Leak-Free Validation:** Evaluated Logistic Regression, Random Forest, and LightGBM using 5-fold `StratifiedKFold` inside encapsulated `scikit-learn` pipelines.

### Key Results & SHAP Explainability
* **Top Model:** Tuned **LightGBM Classifier** (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=100`) achieved **ROC-AUC = 0.8842** and **PR-AUC = 0.6376** (over **4.2x** the 15.0% baseline).
* **Key Risk Drivers:** SHAP analysis confirmed that recent balance drops (`bal_drop_m3_m4` #1, `bal_ratio_m1_m6` #2) and declining deposit inflows drive impending liquidity distress, enabling proactive micro-credit offers and repayment grace periods.

### Repository Contents
* `Financial_Stress_Capstone.ipynb` — Complete end-to-end modeling & SHAP explainability notebook
* `Financial_Stress_Capstone.html` — Rendered interactive HTML report
* `financial_stress_predictions.xls` — Calibrated test set risk probabilities & binary classifications (`N = 30,000`)
* `requirements.txt` — Python package dependencies
