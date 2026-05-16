# Diabetes Project — Codex Agent Guide

Diabetes risk classification and prevention policy proposal. Python data science project using BRFSS dataset.  
**Codex = implementation. Claude = planning and review.**

---

## Role

Codex implements notebooks and `src/` modules based on plans and context provided by Claude. Do not add unrequested features. When a goal is ambiguous, ask before writing code.

---

## Working Principles

### 1. Think Before Coding

State assumptions when they affect behavior. If a request has multiple valid interpretations, clarify before writing code.

### 2. Prefer Simplicity

Implement only what was requested.

- Do not add unrequested features or abstractions.
- If 200 lines can be expressed clearly in 50, simplify.
- Rule of thumb: if a senior data scientist would call it needlessly complex, simplify it.

### 3. Make Minimal Changes

Change only what is necessary.

- Do not refactor working cells unless the task requires it.
- Follow the existing notebook style and cell structure.
- Remove imports or variables made unused by your change.

### 4. Work Toward Verifiable Goals

For multi-step work, confirm the expected output before writing code:

```text
1. [Step] -> Verify: [method]
2. [Step] -> Verify: [method]
```

---

## Codex Operating Notes

- Keep file reads and edits scoped. Avoid broad recursive work.
- Process large files in chunks instead of loading or rewriting them wholesale.
- Avoid unnecessary parallel tool execution.
- Do not revert user changes unless explicitly asked.
- Prefer existing project patterns over new abstractions.

---

## Project Structure

```text
diabetes_project/
├── data/
│   └── diabetes_indicators.csv          # 253,680 rows, 21 features
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory data analysis
│   ├── 02_preprocessing.ipynb           # Preprocessing (core)
│   ├── 03_modeling.ipynb                # Model training and evaluation
│   └── 04_decision.ipynb                # Risk profiling and policy proposal
├── src/
│   ├── preprocessing.py                 # Reusable preprocessing functions
│   ├── modeling.py                      # Reusable modeling functions
│   └── visualization.py                 # Reusable visualization functions
├── outputs/
│   ├── figures/                         # Saved plots (.png)
│   └── models/                          # Saved models (.pkl)
└── requirements.txt
```

---

## Dataset

| Item | Value |
|------|-------|
| Source | BRFSS Diabetes Health Indicators (Kaggle) |
| Rows | 253,680 |
| Features | 21 (BMI, blood pressure, age, smoking, exercise, cholesterol, etc.) |
| Target | `Diabetes_binary` (0: normal, 1: diabetic/pre-diabetic) |
| Class ratio | ~86% normal / ~14% diabetic (imbalanced) |
| Primary metric | Recall (cost of missed detection > cost of false positive) |

---

## Context Header (add to each notebook)

Always include this header at the top of each notebook so context is self-contained:

```python
# CONTEXT: Diabetes risk classification project
# DATASET: BRFSS Diabetes Health Indicators (253,680 rows, 21 features)
# TARGET: Diabetes_binary (0/1), class imbalance 86:14
# GOAL: [specific goal for this notebook]
# PRIORITY: Maximize Recall (missed detection cost > false positive cost)
# OUTPUT: [expected outputs — plots, tables, model files, etc.]
```

---

## Per-Step Implementation Notes

### STEP 1 — EDA (`01_EDA.ipynb`)

Goal: understand data structure and visualize feature-target relationships.

Key tasks:
- Load data and check shape, dtypes, describe
- Visualize target class distribution and quantify imbalance ratio
- Feature histograms (all 21 features)
- Feature distribution by target class (boxplot / violinplot)
- Correlation heatmap
- Add brief medical interpretation note per feature

### STEP 2 — Preprocessing (`02_preprocessing.ipynb`)

Goal: compare preprocessing strategies and select the best pipeline.

Key tasks:
- Outlier detection: compare IQR vs Z-score side by side with visualization
- Missing value imputation: compare mean, median, KNN — show distribution change after each
- Class imbalance: compare SMOTE, random undersampling, SMOTE+undersampling — use logistic regression to compare Recall
- Feature engineering: check encoding, compare StandardScaler vs MinMaxScaler, VIF for multicollinearity
- Final pipeline: wrap chosen strategies in `sklearn.pipeline.Pipeline`, train/test split 8:2 with `stratify=True`

```python
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer, KNNImputer
```

### STEP 3 — Modeling (`03_modeling.ipynb`)

Goal: compare 3 models and select the best one; explain feature importance.

Models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Evaluation:
- Confusion matrix for each model (side by side)
- Recall, Precision, F1, AUC-ROC comparison table
- ROC Curve overlay
- SHAP summary plot and beeswarm plot for the selected model

```python
import shap
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Note: run SHAP on a subsample first to verify memory usage before applying to full test set.

### STEP 4 — Decision (`04_decision.ipynb`)

Goal: translate analysis results into actionable policy proposals.

Key tasks:
- Compute risk score from model prediction probabilities
- Classify into 3 risk tiers: high / medium / low
- Feature mean comparison table per tier
- Write a risk persona for the high-risk group
- Four policy scenarios: mandatory screening, weight management, exercise program, smoking+hypertension package
- Threshold tuning: visualize Recall vs Precision tradeoff; justify chosen threshold

---

## `src/` Module Conventions

Functions in `src/` must be importable from notebooks without side effects.

- `preprocessing.py`: data loading, outlier/imputation/scaling/balancing functions
- `modeling.py`: training, evaluation, metric computation functions
- `visualization.py`: reusable plot functions (accept axes or figure as argument)

Each function should do one thing. No global state.

---

## Output Conventions

- Figures: save to `outputs/figures/{step}_{description}.png` at 150 dpi minimum
- Models: save to `outputs/models/{model_name}.pkl` using `joblib`
- Tables: print as pandas DataFrame or use `df.to_markdown()` for inline display

---

## Memory and Performance Notes

253k rows can cause memory issues with certain operations.

- SHAP: use a sample of 1,000–5,000 rows for initial validation
- KNN imputation: expensive on large datasets — confirm it is the chosen strategy before running on full data
- Avoid loading the full dataset multiple times in the same notebook session
- Save intermediate results to disk if recomputation is expensive
