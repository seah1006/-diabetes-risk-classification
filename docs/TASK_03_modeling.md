# TASK 03 — 모델링 (`03_modeling.ipynb` + `src/modeling.py`)

## 역할

Codex가 수행. 완료 후 Claude가 모델 선택 근거를 검토하고 정책 제안 방향을 확정한다.

## 언어 규칙

노트북 내 모든 텍스트는 **한국어**로 작성한다.

- markdown 셀 제목, 설명, 해석, 선택 근거 작성란 → 한국어
- 코드 내 주석(`#`) → **작성하지 않는다** (발표용)
- `print()` 출력 레이블 → 한국어
- 변수명·함수명은 영문 유지 (코드 가독성)

## 전제 조건

- TASK 02 완료: `outputs/models/train_test_split.pkl` 존재
- 전처리 전략 확정 (Claude 리뷰 후 TASK 02 섹션 6이 채워진 상태)

## 목표

로지스틱 회귀, 랜덤 포레스트, XGBoost 세 모델을 학습·비교하고  
SHAP으로 피처 중요도를 분석한다. 최적 모델을 `outputs/models/`에 저장한다.

---

## 작업 목록

### 섹션 1 — 데이터 로드

```python
import joblib

X_train, X_test, y_train, y_test = joblib.load('../outputs/models/train_test_split.pkl')

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train class ratio:\n{pd.Series(y_train).value_counts(normalize=True)}")
```

### 섹션 2 — 모델 학습

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0),
}

trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained[name] = model
    print(f"{name} trained.")
```

### 섹션 3 — 평가 지표 비교 테이블

```python
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

rows = []
for name, model in trained.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        'Model':     name,
        'Recall':    recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'AUC-ROC':   roc_auc_score(y_test, y_prob),
    })

metrics_df = pd.DataFrame(rows).set_index('Model').round(4)
print(metrics_df)
```

### 섹션 4 — Confusion Matrix (3개 나란히)

```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, trained.items()):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
```

그림을 `outputs/figures/03_confusion_matrices.png`에 저장.

### 섹션 5 — ROC Curve 오버레이

```python
from sklearn.metrics import roc_curve

plt.figure(figsize=(8, 6))
for name, model in trained.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve Comparison')
plt.legend()
```

그림을 `outputs/figures/03_roc_curves.png`에 저장.

### 섹션 6 — Recall 주 지표 선택 근거 (markdown 셀)

아래 내용을 markdown 셀로 작성한다:

```markdown
## Recall을 주 지표로 선택한 이유

당뇨 위험군 분류에서 두 가지 오류 비용이 비대칭적이다:

- **False Negative (미탐지)**: 실제 당뇨 위험군을 정상으로 분류 → 진단 지연, 합병증 위험
- **False Positive (오탐지)**: 정상인을 위험군으로 분류 → 불필요한 검사 비용

미탐지 비용이 오탐지 비용보다 훨씬 크므로 Recall(재현율)을 최우선 지표로 설정한다.
AUC-ROC는 클래스 불균형 상황에서 전반적인 판별력을 보조 확인하는 데 사용한다.
```

### 섹션 7 — 피처 중요도: 랜덤 포레스트

```python
feature_names = ...  # X_train의 컬럼명 (DataFrame이면 .columns, numpy면 원본 컬럼 사용)

rf = trained['Random Forest']
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.plot.bar()
plt.title('Random Forest Feature Importance')
plt.tight_layout()
```

그림을 `outputs/figures/03_rf_feature_importance.png`에 저장.

### 섹션 8 — 피처 중요도: XGBoost

```python
xgb = trained['XGBoost']
xgb_importances = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
xgb_importances.plot.bar()
plt.title('XGBoost Feature Importance')
plt.tight_layout()
```

그림을 `outputs/figures/03_xgb_feature_importance.png`에 저장.

### 섹션 9 — SHAP 분석

**주의**: 전체 test set에 바로 적용하면 메모리/시간 문제가 생길 수 있다.  
반드시 샘플 1,000~2,000개로 먼저 확인 후 전체로 확장한다.

```python
import shap

# 최고 Recall 모델에 적용 (섹션 3 결과 기반으로 선택)
best_model = trained['XGBoost']  # Claude 리뷰 후 변경 가능

# 샘플로 먼저 검증
sample_idx = np.random.choice(len(X_test), size=1000, replace=False)
X_sample = X_test[sample_idx] if isinstance(X_test, np.ndarray) else X_test.iloc[sample_idx]

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X_sample,
                  feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('../outputs/figures/03_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Beeswarm plot
plt.figure()
shap.summary_plot(shap_values, X_sample,
                  feature_names=feature_names, plot_type='violin', show=False)
plt.tight_layout()
plt.savefig('../outputs/figures/03_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 섹션 10 — 피처 중요도 해석 (markdown 셀)

분석 결과를 바탕으로 아래 형식으로 작성한다:

```markdown
## 당뇨 발병에 가장 영향을 주는 요인

SHAP 분석 기준 상위 5개 피처:

1. **[피처명]**: [방향성 및 의학적 해석]
2. **[피처명]**: ...
3. ...

→ 이 결과는 STEP 4 정책 시나리오 수립의 근거로 활용된다.
```

### 섹션 11 — 모델 저장

```python
import joblib

for name, model in trained.items():
    filename = name.lower().replace(' ', '_')
    joblib.dump(model, f'../outputs/models/{filename}.pkl')
    print(f"Saved: {filename}.pkl")

# 최종 선택 모델도 별도 저장
joblib.dump(best_model, '../outputs/models/best_model.pkl')
print("Saved: best_model.pkl")
```

---

## `src/modeling.py` 구현

```python
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'Recall':    recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'AUC-ROC':   roc_auc_score(y_test, y_prob),
    }

def make_metrics_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results).T.round(4)
```

---

## 완료 기준

- [ ] 3개 모델이 모두 학습됨
- [ ] 평가 지표 비교 테이블이 출력됨
- [ ] 6개 시각화 파일이 `outputs/figures/`에 저장됨
  - `03_confusion_matrices.png`
  - `03_roc_curves.png`
  - `03_rf_feature_importance.png`
  - `03_xgb_feature_importance.png`
  - `03_shap_summary.png`
  - `03_shap_beeswarm.png`
- [ ] 모델 pkl 파일이 `outputs/models/`에 저장됨
- [ ] SHAP 분석이 샘플 서브셋으로 실행됨 (메모리 안전)
- [ ] 피처 중요도 해석 markdown 셀이 존재함
- [ ] `src/modeling.py` 2개 함수 구현됨

## 하지 말 것

- 하이퍼파라미터 튜닝 금지 (요청하지 않음)
- SHAP을 전체 데이터셋에 바로 적용하지 않는다 — 샘플 먼저
- 섹션 10 해석에 근거 없는 내용 작성 금지 (SHAP 수치 기반으로만)

---

## Claude 리뷰 포인트 (Codex 완료 후)

1. 3개 모델 Recall 수치 확인 → 최종 모델 선택 및 문서화
2. SHAP 상위 피처 확인 → STEP 4 정책 타겟 그룹 결정
3. best_model이 올바르게 저장되었는지 확인
