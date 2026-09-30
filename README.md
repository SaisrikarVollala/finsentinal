# Credit Risk Prediction and Explainable Default Analytics

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-11%20passed-green.svg)](https://pytest.org/)

A portfolio-quality machine learning and explainable AI system designed to predict **next-month credit card payment defaults** (`default_payment_next_month`) for existing cardholders using the public [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009).

---

## 🎯 Problem Statement & Scope

### Task Objective
Predict whether an existing revolving credit card customer will default on their required credit card bill payment in the following month (October 2005), given their 6-month historical billing statements, repayment records, credit limits, and demographic attributes from April to September 2005.

> **Critical Domain Scope**: This is an **existing cardholder account monitoring and delinquency mitigation system**, *not* a new-applicant loan underwriting scorecard. New loan applicants lack multi-month billing statements and payment histories; active cardholder risk models leverage dynamic delinquency and liquidity trajectories.

---

## 🏗️ Project Architecture & Structure

```
credit-risk-prediction/
├── data/
│   ├── raw/                           # Raw downloaded archives & Excel/CSV data
│   ├── processed/                     # Validated, clean dataset
│   ├── data_dictionary.json          # Machine-readable schema & column definitions
│   └── data_dictionary.md            # Markdown reference table
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # End-to-end EDA notebook with insights
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Automated download, schema validation, cleaning
│   ├── features.py                    # Leak-free feature engineering & ratio safety
│   ├── preprocessing.py              # ColumnTransformer, scaling, stratified splits
│   ├── train.py                       # 5-fold Stratified CV, imbalance tuning, model training
│   ├── evaluate.py                    # Test set evaluation, ROC/PR curves, calibration
│   └── predict.py                     # Production inference API & Tree SHAP explainability
├── models/
│   ├── xgboost_model.joblib           # Champion fitted model
│   ├── randomforest_model.joblib      # Parallel ensemble model
│   ├── logisticregression_model.joblib# Baseline linear model
│   ├── preprocessor.joblib            # Fitted ColumnTransformer (train only)
│   ├── feature_names.json             # Serialized post-transform feature metadata
│   ├── cv_results.json                # Cross-validation performance summaries
│   └── test_evaluation_metrics.json   # Held-out test benchmark metrics
├── reports/
│   └── figures/                       # High-resolution diagnostic figures (01 to 11)
├── tests/
│   ├── test_data_loader.py            # Ingestion, schema, and ID exclusion tests
│   ├── test_features.py               # Zero-division safety & feature logic tests
│   ├── test_preprocessing.py          # Stratified split & pipeline leakage tests
│   └── test_predict.py                # Inference schema, bounds, and SHAP tests
├── app.py                             # Interactive 5-tab Streamlit dashboard
├── requirements.txt                   # Frozen dependency specifications
├── README.md                          # Project documentation & interview Q&A
└── .gitignore                         # Version control exclusions
```

---

## 🔬 Modeling Methodology & Key Innovations

### 1. Leak-Free Feature Engineering (`src/features.py`)
All engineered features rely strictly on backward-looking data (April to September 2005) available at prediction time $t$:
- **Credit Utilization Proxies**: $\text{UTILIZATION}_i = \frac{\text{BILL\_AMT}_i}{\max(\text{LIMIT\_BAL}, 1)}$, with average and latest utilization clamped to $[-0.5, 3.0]$.
- **Payment-to-Bill Ratios**: With explicit zero-denominator and negative balance protections:
  $$\text{PAY\_TO\_BILL\_RATIO} = \begin{cases} 1.0 & \text{if } \text{BILL\_AMT} \le 0 \text{ and } \text{PAY\_AMT} \ge 0 \\ 0.0 & \text{if } \text{BILL\_AMT} \le 0 \text{ and } \text{PAY\_AMT} < 0 \\ \min\left(\frac{\text{PAY\_AMT}}{\text{BILL\_AMT}}, 2.0\right) & \text{if } \text{BILL\_AMT} > 0 \end{cases}$$
- **Delinquency Dynamics**: 
  - `TOTAL_DELAYED_MONTHS`: Count of months where repayment status $> 0$.
  - `MAX_DELAY_MONTHS`: Maximum delinquency severity over the 6-month window.
  - `DELAY_TREND`: Trajectory difference ($\text{PAY\_0} - \text{PAY\_6}$), signaling deteriorating vs improving behavior.
  - `EVER_SEVERE_DELAY`: Indicator if cardholder ever reached $\ge 2$ months past due.
- **Statement Expansion**: `BILL_GROWTH` ($\text{BILL\_AMT1} - \text{BILL\_AMT6}$) and `NET_UNPAID_EXPOSURE_1`.

### 2. Imbalance-Aware Training & Threshold Optimization (`src/train.py`)
- Dataset default rate is **22.12%** ($\approx 3.52:1$ negative-to-positive ratio).
- Avoided synthetic oversampling (SMOTE) to preserve natural probability calibration and avoid artificial point-mass interpolation in high dimensions.
- Incorporated cost-sensitive loss weighting (`scale_pos_weight = 3.52` in XGBoost, `class_weight='balanced'` in Logistic Regression).
- Derived the optimal classification probability threshold ($\theta^* = 0.59$) strictly on validation folds via 5-Fold Stratified Cross-Validation to maximize F1-score and balance precision vs recall.

---

## 📊 Empirical Evaluation & Held-Out Test Benchmarks

Evaluated on an untouched, held-out test partition of **6,000 cardholders** (1,327 defaults, 4,673 non-defaults):

| Model | Test ROC-AUC | Test PR-AUC | Brier Score | Decision Threshold | Test Precision | Test Recall | Test F1-Score | Test Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.7520 | 0.5013 | 0.1928 | 0.50 (Std)<br>**0.60 (Opt)** | 43.43%<br>**48.08%** | 60.74%<br>**54.79%** | 50.64%<br>**51.22%** | 77.53%<br>**83.20%** |
| **Random Forest (Parallel Trees)** | 0.7778 | 0.5528 | 0.1758 | 0.50 (Std)<br>**0.54 (Opt)** | 47.63%<br>**51.36%** | 59.91%<br>**56.82%** | 53.07%<br>**53.95%** | 81.30%<br>**84.72%** |
| **XGBoost (Champion Model)** | **0.7799** | **0.5578** | **0.1797** | 0.50 (Std)<br>**0.59 (Opt)** | 45.89%<br>**52.68%** | 63.45%<br>**55.46%** | 53.26%<br>**54.04%** | 78.75%<br>**85.85%** |

### Key Benchmark Observations:
1. **PR-AUC Lift**: XGBoost achieves a PR-AUC of **0.5578**, representing a **+152% lift** over the random no-skill baseline (0.2212).
2. **Threshold Impact**: Adjusting the decision threshold from 0.50 to 0.59 reduces False Positives by **33.4%** (from 993 to 661 accounts), minimizing unnecessary customer friction and collection outreach costs.
3. **Probability Calibration**: Brier scores of ~0.179 confirm well-calibrated probabilities across bins, ensuring reliability for risk-tier assignments.

---

## 🔍 Explainability & Responsible AI Insights

- **Global Feature Importance**: Delinquency metrics (`MAX_DELAY_MONTHS`, `TOTAL_DELAYED_MONTHS`, `PAY_0`) account for over 55% of global model gain, followed by total payments (`SUM_PAY_AMT`) and utilization.
- **Local SHAP Attribution**: Decomposes individual cardholder risk into additive log-odds contributions (e.g., entering 2-month delay on `PAY_0` adds +0.89 to default log-odds).
- **Ethical & Regulatory Considerations**:
  - Demographic markers (`SEX`, `MARRIAGE`, `AGE`) have minimal contribution (<3% global importance).
  - In commercial deployment, protected demographic attributes must be excluded or subjected to disparate impact analysis under Fair Lending regulations (e.g. ECOA / Regulation B).
  - Feature importances reflect statistical correlations within Taiwan cardholders in 2005 and do not establish direct individual causation.

---

## 🚀 Installation & Quickstart

### 1. Prerequisites & Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/credit-risk-prediction.git
cd credit-risk-prediction

# Create and activate Python virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. End-to-End Execution Pipeline
```bash
# Step 1: Download data, validate schema, and generate data dictionary
python -u -m src.data_loader

# Step 2: Generate exploratory data analysis charts
python -u -m src.eda

# Step 3: Train models, run 5-Fold Stratified CV, and tune thresholds
python -u -m src.train

# Step 4: Evaluate models on held-out test set & generate diagnostic figures
python -u -m src.evaluate

# Step 5: Run the automated pytest suite (11 test cases)
pytest -v tests
```

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to access the 5-tab application.

---

## 💼 Technical Interview Q&A

### Q1: How did you prevent data leakage during feature engineering and model preprocessing?
**Answer**:
1. **Temporal Isolation**: All features are calculated backward in time using only historical statements from month M1 (September 2005) back to M6 (April 2005). No target information or post-September data was incorporated.
2. **Strict Preprocessing Fit**: The dataset was partitioned into train (80%) and test (20%) using stratified sampling *before* fitting any scaler or encoder. The `ColumnTransformer` (median imputers, standard scalers, one-hot encoders) was fitted exclusively on `X_train` and only applied (`.transform()`) to `X_test`.
3. **Cross-Validation Isolation**: Out-of-fold predictions (`cross_val_predict`) and threshold searches were conducted within cross-validation loops, preventing the test set from influencing threshold selection.

---

### Q2: Why is PR-AUC preferred over ROC-AUC and Accuracy for imbalanced credit risk modeling?
**Answer**:
1. **Accuracy Paradox**: In a dataset with a 22.12% default rate, a trivial model predicting "0" for all customers scores 77.88% accuracy while failing to detect 100% of defaults.
2. **ROC-AUC Distortion**: ROC-AUC plots False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$) on the x-axis. Because the majority class (non-defaults: 77.88%) is large, $\text{TN}$ dominates the denominator, keeping $\text{FPR}$ artificially low even when the model generates hundreds of False Positives.
3. **PR-AUC Precision**: Precision-Recall AUC focuses directly on the minority class ($\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$ vs $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$). A high PR-AUC proves that when the model flags a cardholder for default, it maintains high positive predictive value without overwhelming risk operations with false alarms.

---

### Q3: How does adjusting the classification threshold affect the financial trade-off between False Positives and False Negatives?
**Answer**:
- **False Negative (FN) - Missed Default**: A customer defaults without proactive intervention. The financial loss is substantial (e.g., unpaid charge-off balance of NT$ 50,000+).
- **False Positive (FP) - False Alarm**: A safe customer is flagged as high risk. The cost is minor (e.g., automated SMS reminder, soft credit line freeze, or small customer friction costing NT$ 1,000 - 2,000).
- **Threshold Dynamics**:
  - Lowering the threshold ($\theta \to 0.30$) prioritizes **Recall**, capturing nearly all defaults at the expense of more false alarms.
  - Raising the threshold ($\theta \to 0.59$) prioritizes **Precision**, eliminating 334 false alarms while preserving a 55.5% capture rate on true defaults.
  - In production, the optimal threshold $\theta^*$ is chosen by minimizing total expected portfolio cost:
    $$\text{Total Cost}(\theta) = \text{FN}(\theta) \times \text{Cost}_{\text{default}} + \text{FP}(\theta) \times \text{Cost}_{\text{friction}}$$

---

## 📜 License & Citation
- Dataset Citation: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
- License: Open-source educational and portfolio distribution under the MIT License.
