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
│   ├── 03_modeling.ipynb                # ❌ TODO — ML model comparison + SHAP (see TASK below)
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

## ❌ CURRENT TASK — `03_modeling.ipynb`

**This is the only notebook that needs to be created from scratch.**

The trained model files already exist in `outputs/models/`, but there is no notebook that documents the training, comparison, and selection process. Create `03_modeling.ipynb` to fill this gap.

### Context

```python
# CONTEXT: Diabetes risk classification project
# DATASET: BRFSS Diabetes Health Indicators (253,680 rows, 21 features)
# TARGET: Diabetes_binary (0/1), class imbalance 86:14
# GOAL: Train 3 models, compare by Recall, explain feature importance with SHAP
# PRIORITY: Maximize Recall (missed detection cost > false positive cost)
# OUTPUT: metrics table, 6 figures, model .pkl files in outputs/models/
```

### Prerequisites

- `outputs/models/train_test_split.pkl` exists — load with:
  ```python
  X_train, X_test, y_train, y_test = joblib.load('../outputs/models/train_test_split.pkl')
  ```
- `X_train`, `X_test` are numpy arrays (already preprocessed — StandardScaled, SMOTE applied to train)
- Feature order: `['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']`

### Sections to implement

#### 섹션 1 — 데이터 로드
- Load `train_test_split.pkl`
- Print shapes and class ratio of y_train

#### 섹션 2 — 모델 학습
Train these 3 models on `(X_train, y_train)`:
```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
}
```

#### 섹션 3 — 평가 지표 비교 테이블
Compute and display as DataFrame: Recall, Precision, F1, AUC-ROC for each model.  
Sort by Recall descending.

#### 섹션 4 — Confusion Matrix (3개 나란히)
`ConfusionMatrixDisplay.from_estimator` for each model side by side (1×3 subplots).  
Save to `outputs/figures/03_confusion_matrices.png`.

#### 섹션 5 — ROC Curve 오버레이
All 3 ROC curves on one plot with AUC in legend.  
Save to `outputs/figures/03_roc_curves.png`.

#### 섹션 6 — Recall 주 지표 선택 근거 (markdown 셀)
Write a markdown cell explaining why Recall is the primary metric:
- False Negative cost vs False Positive cost asymmetry
- Medical consequence of missed detection

#### 섹션 7 — 피처 중요도: Random Forest
Bar chart of `feature_importances_`.  
Save to `outputs/figures/03_rf_feature_importance.png`.

#### 섹션 8 — 피처 중요도: XGBoost
Bar chart of `feature_importances_`.  
Save to `outputs/figures/03_xgb_feature_importance.png`.

#### 섹션 9 — SHAP 분석
**IMPORTANT: Use subsample of 1,000 rows first to avoid OOM. Do NOT run on full test set.**

```python
import shap

# Use the model with highest Recall from section 3
best_model = trained['XGBoost']  # update if different model wins

sample_idx = np.random.choice(len(X_test), size=1000, replace=False)
X_sample = X_test[sample_idx]

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)

# Summary plot → save to outputs/figures/03_shap_summary.png
# Beeswarm plot → save to outputs/figures/03_shap_beeswarm.png
```

#### 섹션 10 — 피처 중요도 해석 (markdown 셀)
Based on SHAP results, write the top 5 features and their medical interpretation.  
Format: "→ 이 결과는 STEP 4 정책 시나리오 수립의 근거로 활용된다."

#### 섹션 11 — 모델 저장
Save all 3 trained models + best model to `outputs/models/`.  
Use `joblib.dump`. Overwrite existing files.

### Language rules (same as other notebooks)
- Markdown cell titles and explanations → Korean
- Code comments (`#`) → do NOT write
- `print()` labels → Korean
- Variable/function names → English

### Done criteria
- [ ] 3 models trained and evaluated
- [ ] Metrics comparison table printed
- [ ] 6 figure files saved to `outputs/figures/`
- [ ] SHAP ran on subsample (memory safe)
- [ ] Feature importance interpretation markdown cell exists
- [ ] Model pkl files saved to `outputs/models/`

### Do NOT
- Do NOT tune hyperparameters
- Do NOT run SHAP on the full dataset — subsample only
- Do NOT modify any other existing notebooks

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
