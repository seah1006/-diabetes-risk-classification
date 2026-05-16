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

### 3. Make Minimal Changes

Change only what is necessary.

- Do not refactor working cells unless the task requires it.
- Follow the existing notebook style and cell structure.
- Remove imports or variables made unused by your change.

### 4. Work Toward Verifiable Goals

For multi-step work, confirm the expected output before writing code.

---

## Project Structure

```text
diabetes_project/
├── data/
│   └── diabetes_indicators.csv          # 253,680 rows, 21 features (NOT in repo — download from Kaggle)
├── notebooks/
│   ├── 01_preprocessing.ipynb           # ✅ DONE — preprocessing pipeline
│   ├── 02_EDA.ipynb                     # ✅ DONE — exploratory data analysis
│   ├── 03_statistical_analysis.ipynb    # ✅ DONE — t-test, chi-square, odds ratio
│   └── 04_decision.ipynb                # ✅ DONE — risk tiers + policy scenarios
├── src/
│   ├── preprocessing.py                 # ✅ DONE
│   ├── statistical_analysis.py          # ✅ DONE
│   └── visualization.py                 # ✅ DONE
├── outputs/
│   ├── figures/                         # Generated plots
│   └── models/
│       ├── train_test_split.pkl         # ✅ EXISTS — preprocessed X_train, X_test, y_train, y_test
│       ├── best_model.pkl               # ✅ EXISTS — but no notebook documents how it was created
│       ├── logistic_regression.pkl      # ✅ EXISTS
│       ├── random_forest.pkl            # ✅ EXISTS
│       └── xgboost.pkl                  # ✅ EXISTS
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

```python
# CONTEXT: Diabetes risk classification project
# DATASET: BRFSS Diabetes Health Indicators (253,680 rows, 21 features)
# TARGET: Diabetes_binary (0/1), class imbalance 86:14
# GOAL: [specific goal for this notebook]
# PRIORITY: Maximize Recall (missed detection cost > false positive cost)
# OUTPUT: [expected outputs — plots, tables, model files, etc.]
```

---

## `src/` Module Conventions

Functions in `src/` must be importable from notebooks without side effects.

- `preprocessing.py`: data loading, outlier/imputation/scaling/balancing functions
- `visualization.py`: reusable plot functions

Each function should do one thing. No global state.

---

## Output Conventions

- Figures: save to `outputs/figures/{step}_{description}.png` at 150 dpi minimum
- Models: save to `outputs/models/{model_name}.pkl` using `joblib`
- Tables: print as pandas DataFrame

---

## Memory and Performance Notes

- SHAP: use a sample of 1,000 rows for initial validation — do NOT skip this step
- Avoid loading the full dataset multiple times in the same notebook session
- Save intermediate results to disk if recomputation is expensive
