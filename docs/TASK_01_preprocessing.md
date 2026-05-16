# TASK 02 — 전처리 (`02_preprocessing.ipynb` + `src/preprocessing.py`)

## 역할

Codex가 수행. 완료 후 Claude가 각 비교 실험 결과를 검토하고 최종 전략을 선택한다.

## 언어 규칙

노트북 내 모든 텍스트는 **한국어**로 작성한다.

- markdown 셀 제목, 설명, 해석, 선택 근거 작성란 → 한국어
- 코드 내 주석(`#`) → **작성하지 않는다** (발표용)
- `print()` 출력 레이블 → 한국어 (예: `print(f"훈련 데이터 크기: {X_train.shape}")`)
- 변수명·함수명은 영문 유지 (코드 가독성)

## 전제 조건

- TASK 01 완료 (EDA 결과 확인, 결측치/이상치 규모 파악)
- `data/diabetes_indicators.csv` 존재

## 목표

이상치, 결측치, 클래스 불균형에 대한 여러 전략을 비교 실험하고,  
선택된 전략으로 `sklearn Pipeline`을 구성한다.  
재사용 함수는 `src/preprocessing.py`로 추출한다.

---

## 작업 목록

### 섹션 1 — 데이터 로드

```python
df = pd.read_csv('../data/diabetes_indicators.csv')
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

print(f"Shape: {df.shape}")
print(f"Class ratio:\n{y.value_counts(normalize=True).round(3)}")
```

### 섹션 2 — 이상치 탐지 비교 (IQR vs Z-score)

BMI를 대표 피처로 양쪽 방법을 비교한다.

```python
# IQR
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1
iqr_mask = (df['BMI'] < Q1 - 1.5 * IQR) | (df['BMI'] > Q3 + 1.5 * IQR)

# Z-score
z_scores = np.abs(stats.zscore(df['BMI']))
z_mask = z_scores > 3

print(f"IQR 이상치 수: {iqr_mask.sum()} ({iqr_mask.mean():.2%})")
print(f"Z-score 이상치 수: {z_mask.sum()} ({z_mask.mean():.2%})")
```

시각화: 같은 피처에 대해 boxplot + 히스토그램을 나란히(2×2 서브플롯) 출력한다.  
이상치 범위(상/하한선)를 수직선으로 표시한다.

**결과를 markdown 셀에 정리**: 어느 방식이 몇 개를 잡는지, 어느 방식이 더 적절한지 근거 작성란 남기기.  
(실제 선택은 Claude 리뷰 후 결정)

그림을 `outputs/figures/02_outlier_comparison.png`에 저장.

### 섹션 3 — 결측치 처리 비교

BRFSS 데이터는 결측치가 없을 수 있으나 파이프라인 설계 목적으로 비교한다.

3가지 전략을 적용한 후 BMI 분포 변화를 시각화한다:

```python
from sklearn.impute import SimpleImputer, KNNImputer

# 결측치가 없으면 테스트용으로 일부 값을 NaN으로 치환
test_df = df.copy()
test_df.loc[test_df.sample(frac=0.05, random_state=42).index, 'BMI'] = np.nan

strategies = {
    'Mean':   SimpleImputer(strategy='mean'),
    'Median': SimpleImputer(strategy='median'),
    'KNN':    KNNImputer(n_neighbors=5),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, imputer) in zip(axes, strategies.items()):
    filled = imputer.fit_transform(test_df[['BMI']])
    ax.hist(filled, bins=40)
    ax.set_title(name)
plt.suptitle('BMI Distribution After Imputation')
```

그림을 `outputs/figures/02_imputation_comparison.png`에 저장.

**결과 markdown 셀**: 각 방법의 분포 변화 요약 및 선택 근거 작성란 남기기.

### 섹션 4 — 클래스 불균형 처리 비교

3가지 전략을 로지스틱 회귀로 빠르게 비교한다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000, random_state=42)

strategies = {
    'No Sampling':        lr,
    'SMOTE':              ImbPipeline([('smote', SMOTE(random_state=42)), ('lr', lr)]),
    'Undersampling':      ImbPipeline([('rus', RandomUnderSampler(random_state=42)), ('lr', lr)]),
    'SMOTE+Undersampling': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('rus', RandomUnderSampler(random_state=42)),
        ('lr', lr)
    ]),
}

results = {}
for name, model in strategies.items():
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='recall')
    results[name] = scores.mean()
    print(f"{name}: Recall = {scores.mean():.4f} ± {scores.std():.4f}")
```

막대 그래프로 Recall 비교 출력.  
그림을 `outputs/figures/02_sampling_comparison.png`에 저장.

**결과 markdown 셀**: 선택 근거 작성란 남기기.

### 섹션 5 — 피처 엔지니어링

#### 스케일링 비교

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

continuous_features = ['BMI', 'MentHlth', 'PhysHlth']

fig, axes = plt.subplots(len(continuous_features), 2, figsize=(12, 10))
for i, col in enumerate(continuous_features):
    std_vals = StandardScaler().fit_transform(df[[col]])
    mmx_vals = MinMaxScaler().fit_transform(df[[col]])
    axes[i, 0].hist(std_vals, bins=40)
    axes[i, 0].set_title(f'{col} — StandardScaler')
    axes[i, 1].hist(mmx_vals, bins=40)
    axes[i, 1].set_title(f'{col} — MinMaxScaler')
plt.tight_layout()
```

그림을 `outputs/figures/02_scaling_comparison.png`에 저장.

#### 다중공선성 확인 (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_df = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values('VIF', ascending=False)

print(vif_df)
```

VIF > 10인 피처를 별도로 표시한다.

### 섹션 6 — 최종 파이프라인 구성 (Claude 선택 후 채울 자리)

이 섹션은 **Claude 리뷰 후** 선택된 전략으로 채운다.  
지금은 구조만 잡아둔다:

```python
# ---- Claude 선택 결과 반영 자리 ----
# 이상치 처리 방식: [IQR / Z-score / 없음]
# 결측치 대체 방식: [Mean / Median / KNN]
# 클래스 불균형 방식: [SMOTE / Undersampling / 결합]
# 스케일러: [Standard / MinMax]

# 예시 구조 (선택 후 수정)
pipeline = ImbPipeline([
    ('imputer',  SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('sampler',  SMOTE(random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
print(f"X_train shape after pipeline: {X_train_res.shape}")
print(f"y_train class ratio: {pd.Series(y_train_res).value_counts(normalize=True)}")
```

train/test split 결과를 저장한다:

```python
import joblib
joblib.dump((X_train_res, X_test, y_train_res, y_test), '../outputs/models/train_test_split.pkl')
print("Saved: outputs/models/train_test_split.pkl")
```

### 섹션 7 — 전처리 전후 통계 비교 테이블

전처리 전과 후의 기술 통계를 나란히 출력한다:

```python
before = df.describe()
after  = pd.DataFrame(X_train_res, columns=X.columns).describe()

comparison = pd.concat([before, after], keys=['Before', 'After'], axis=1)
print(comparison)
```

---

## `src/preprocessing.py` 구현

섹션 2~5에서 인라인으로 작성한 로직 중 반복 사용될 함수를 추출한다.

구현해야 할 함수:

```python
def load_data(path: str) -> pd.DataFrame:
    """Load CSV and return DataFrame."""

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return boolean mask of IQR outliers."""

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of Z-score outliers."""

def build_pipeline(imputer, scaler, sampler=None):
    """Return assembled imblearn Pipeline."""
```

---

## 완료 기준

- [ ] 7개 섹션 모두 실행 가능
- [ ] 4개 비교 시각화 그림이 `outputs/figures/` 에 저장됨
- [ ] 각 비교 섹션 뒤에 선택 근거를 작성할 markdown 셀이 존재함
- [ ] 섹션 6 파이프라인 구조 코드가 존재함 (전략은 비어 있어도 됨)
- [ ] `src/preprocessing.py` 4개 함수가 구현됨

## 하지 말 것

- 섹션 6의 전략을 임의로 선택하지 않는다 — 구조만 잡고 주석으로 표시
- 모델 학습 금지 (TASK 03에서 수행)
- statsmodels가 없으면 VIF 섹션을 건너뛰고 주석 처리

---

## Claude 리뷰 포인트 (Codex 완료 후)

1. IQR vs Z-score 비교 그래프 확인 → 이상치 처리 방식 선택
2. 불균형 처리별 Recall 수치 확인 → 샘플링 전략 선택
3. VIF 결과 확인 → 피처 제거 여부 결정
4. 위 결정을 바탕으로 섹션 6 파이프라인 코드를 완성하도록 Codex에게 추가 지시
