# 당뇨 위험군 분류 및 예방 정책 제안 — 발표 대본

> **데이터**: BRFSS Diabetes Health Indicators (Kaggle)  
> **행 수**: 253,680행 / **피처**: 21개 / **타겟**: `Diabetes_binary` (0: 정상, 1: 당뇨·전당뇨)  
> **클래스 비율**: 정상 86.07% / 당뇨 13.93% (불균형)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [STEP 1 — 전처리](#2-step-1--전처리-01_preprocessingipynb)
3. [STEP 2 — EDA / 시각화](#3-step-2--eda--시각화-02_edaipynb)
4. [STEP 3 — 통계 분석](#4-step-3--통계-분석-03_statistical_analysisipynb)
5. [STEP 4 — 의사결정 및 정책 제안](#5-step-4--의사결정-및-정책-제안-04_decisionipynb)
6. [결론 및 한계점](#6-결론-및-한계점)

---

## 1. 프로젝트 개요

### 왜 이 주제인가

당뇨병은 한국과 미국 모두에서 주요 만성 질환이다. 미국 BRFSS(행동위험요인 감시 시스템) 데이터를 활용해 **어떤 생활 습관·건강 지표가 당뇨 위험을 높이는지** 통계적으로 규명하고, 그 결과를 **실제 예방 정책**으로 연결하는 것이 목표다.

### 분석 흐름

```
원본 데이터
    ↓
01 전처리      이상치·결측치·불균형 처리 전략 비교 → 최적 파이프라인
    ↓
02 EDA         피처 분포, 타겟과의 관계, 상관관계 시각화
    ↓
03 통계 분석   카이제곱 / t-검정 / 로지스틱 회귀 오즈비 → 위험 요인 규명
```

---

## 2. STEP 1 — 전처리 (`01_preprocessing.ipynb`)

### 목표

이상치·결측치·클래스 불균형 처리 방식을 각각 비교 실험한 뒤, 가장 높은 Recall을 내는 전략을 선택해 최종 파이프라인을 구성한다.

---

### 섹션 1 — 데이터 로드

```python
df = load_data(DATA_PATH)
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']
continuous_features = ['BMI', 'MentHlth', 'PhysHlth']

print(f'데이터 형태: {df.shape}')
print(f'클래스 비율:\n{y.value_counts(normalize=True).round(4)}')
print(f'전체 결측치 수: {df.isnull().sum().sum()}')
```

**설명:**  
- `load_data()`는 `src/preprocessing.py`의 헬퍼 함수로 CSV를 읽어온다.
- 타겟인 `Diabetes_binary`를 분리해 `X`(피처)와 `y`(레이블)로 나눈다.
- 연속형 피처는 BMI, 정신건강 일수(MentHlth), 신체건강 일수(PhysHlth) 세 가지다.
- 출력 결과: 정상 86.07%, 당뇨 13.93%, **결측치 0개** (이 데이터셋은 결측치가 없다).

---

### 섹션 2 — 이상치 탐지 비교: IQR vs Z-score

```python
iqr_mask = detect_outliers_iqr(bmi)
z_mask   = detect_outliers_zscore(bmi, threshold=3.0)

print(f'IQR 이상치 수: {iqr_mask.sum()} ({iqr_mask.mean():.2%})')
print(f'Z-score 이상치 수: {z_mask.sum()} ({z_mask.mean():.2%})')
```

**설명:**
- IQR 방식은 사분위수 범위(Q3−Q1)의 1.5배를 기준으로 이상치를 판별한다.
- Z-score 방식은 평균에서 표준편차의 3배를 벗어나는 값을 이상치로 본다.
- 결과: IQR은 약 **3.88%**, Z-score는 약 **1.17%**를 이상치로 분류한다.
- **최종 선택: Z-score 클리핑** — IQR은 BMI 41 이상을 모두 이상치로 잡는데, 비만 1~3단계(BMI 30~40)가 실제 의학적으로 유효한 값이므로 범위를 과도하게 좁히는 IQR보다 Z-score 기준이 더 적합하다.
- 중요: **제거(drop) 방식이 아닌 클리핑(clip)**을 사용한다. 이상치를 삭제하면 데이터 손실이 발생하므로, 상·하한값으로 대체하는 클리핑을 적용한다.

---

### 섹션 3 — 결측치 대체 비교

```python
strategies = {
    '평균': SimpleImputer(strategy='mean'),
    '중앙값': SimpleImputer(strategy='median'),
    'KNN': KNNImputer(n_neighbors=5),
}

test_df.loc[missing_index, 'BMI'] = np.nan
```

**설명:**
- 이 데이터셋은 실제 결측치가 없어, 비교를 위해 BMI 값의 5%를 인위적으로 NaN으로 만들었다.
- 세 전략 모두 분포 형태를 잘 보존한다.
- KNN은 이웃 5개를 참조해 더 정밀하지만 253,680행에서 연산 비용이 크다.
- **최종 선택: 중앙값(Median) 대체** — 연산 비용이 낮고, 이상치에 영향을 덜 받아 안전하다. 파이프라인 안전망으로 포함한다.

---

### 섹션 4 — 클래스 불균형 처리 비교 ← 핵심 섹션

```python
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

strategies = {
    '샘플링 없음': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)),
    ]),
    'SMOTE': ImbPipeline([...('smote', SMOTE(random_state=42))...]),
    '언더샘플링': ImbPipeline([...('rus', RandomUnderSampler(random_state=42))...]),
    'SMOTE+언더샘플링': ImbPipeline([
        ...
        ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('rus', RandomUnderSampler(sampling_strategy=1.0, random_state=42)),
    ]),
}

for name, model in strategies.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall', n_jobs=1)
    print(f'{name}: Recall = {scores.mean():.4f} ± {scores.std():.4f}')
```

**설명:**
- `StratifiedKFold(n_splits=3)`는 클래스 비율을 유지하면서 데이터를 3개로 나눠 교차 검증한다.
- `ImbPipeline`은 imbalanced-learn의 파이프라인으로 SMOTE·언더샘플링을 안전하게 연결한다.
- SMOTE(`sampling_strategy=0.5`): 소수 클래스(당뇨)를 다수 클래스의 50% 수준까지 합성 생성한다.
- RandomUnderSampler(`sampling_strategy=1.0`): 다수 클래스를 소수 클래스와 1:1 비율이 될 때까지 줄인다.

| 전략 | 3-fold CV Recall |
|------|-----------------|
| 샘플링 없음 | 0.1578 |
| 언더샘플링 | 0.7645 |
| SMOTE+언더샘플링 | 0.7616 |
| SMOTE | 약 0.75 |

- **최종 선택: SMOTE + 언더샘플링 결합** — Recall이 높고, 단순 언더샘플링보다 원본 데이터 정보를 더 보존한다.

---

### 섹션 5 — 피처 엔지니어링 점검

**스케일링 비교**

```python
for col in continuous_features:
    std_vals = StandardScaler().fit_transform(df[[col]]).ravel()
    mmx_vals = MinMaxScaler().fit_transform(df[[col]]).ravel()
```

- StandardScaler: 평균 0, 표준편차 1로 변환. 이상치에 상대적으로 강인하다.
- MinMaxScaler: 0~1 범위로 압축. 이상치가 있으면 다른 값들이 좁은 범위에 몰린다.
- **최종 선택: StandardScaler** — 이상치 클리핑 후에도 분포가 고르며 로지스틱 회귀에 적합하다.

**다중공선성 점검 (VIF)**

```python
vif_df = pd.DataFrame({
    '피처': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
}).sort_values('VIF', ascending=False)
```

- VIF(분산팽창인수)가 10을 초과하면 다중공선성이 의심된다.
- VIF > 10 피처: Education(29.5), CholCheck(23.2), AnyHealthcare(20.8), BMI(18.1), Income(14.2), GenHlth(10.7)
- **피처 제거하지 않음**: 로지스틱 회귀의 L2 정규화가 다중공선성을 완화하며, 제거 시 예측력 손실이 더 크다.

---

### 섹션 6 — 최종 파이프라인 구성 ← 핵심 코드

```python
class ZScoreClipper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, threshold=3.0):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None):
        self.means_ = X_df[self.columns].mean()
        self.stds_  = X_df[self.columns].std(ddof=0).replace(0, np.nan)
        self.lower_bounds_ = self.means_ - self.threshold * self.stds_
        self.upper_bounds_ = self.means_ + self.threshold * self.stds_
        return self

    def transform(self, X):
        for col in self.columns:
            X_df[col] = X_df[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df
```

- `BaseEstimator`, `TransformerMixin`을 상속해 sklearn 파이프라인에 바로 연결 가능한 커스텀 변환기를 만든다.
- `fit()`에서 훈련 데이터의 평균·표준편차를 저장하고, `transform()`에서 그 범위로 클리핑한다.
- 테스트 데이터에는 훈련 데이터 기준으로 계산된 경계값을 그대로 적용해 **데이터 누수를 방지**한다.

```python
final_pipeline = ImbPipeline([
    ('clipper',     ZScoreClipper(columns=continuous_features, threshold=3.0)),
    ('imputer',     SimpleImputer(strategy='median')),
    ('scaler',      StandardScaler()),
    ('smote',       SMOTE(sampling_strategy=0.5, random_state=42)),
    ('undersampler', RandomUnderSampler(sampling_strategy=1.0, random_state=42)),
])

X_train_res, y_train_res = final_pipeline.fit_resample(X_train, y_train)
```

- 파이프라인 5단계: Z-score 클리핑 → 중앙값 대체 → 표준화 → SMOTE → 언더샘플링
- `fit_resample()`로 훈련 데이터에 전체 파이프라인 적용. 테스트 데이터는 샘플링 단계 없이 앞 3단계만 적용한다.
- 결과: X_train_res (174,666행), X_test (50,736행)

```python
joblib.dump((X_train_res, X_test_processed, y_train_res, y_test),
            MODEL_DIR / 'train_test_split.pkl')
```

- 전처리 완료된 데이터를 pkl 파일로 저장해 이후 노트북에서 재사용한다.

---

## 3. STEP 2 — EDA / 시각화 (`02_EDA.ipynb`)

### 목표

원본 데이터의 구조를 파악하고 피처와 타겟의 관계를 시각화해 분석 방향을 잡는다.

---

### 섹션 2 — 타겟 클래스 분포

```python
counts = df['Diabetes_binary'].value_counts().sort_index()
ratios = df['Diabetes_binary'].value_counts(normalize=True).sort_index()
class_summary = pd.DataFrame({'건수': counts.astype(int), '비율': ratios, '비율(%)': ratios * 100})
class_summary.index = class_summary.index.map({0.0: '0 - 정상', 1.0: '1 - 당뇨/전당뇨'})
```

**설명:**
- `value_counts(normalize=True)`로 비율을 계산하고, `map()`으로 인덱스를 읽기 쉽게 변환한다.
- 막대 그래프에 건수와 비율을 함께 레이블로 표시해 불균형 정도를 직관적으로 보여준다.
- **정상 218,334명(86.07%) vs 당뇨 35,346명(13.93%)** — 약 6.2:1의 불균형.

---

### 섹션 3 — 피처별 분포 히스토그램

```python
fig, axes = plt.subplots(5, 5, figsize=(20, 16))
for i, col in enumerate(feature_cols):
    df[col].hist(ax=axes_flat[i], bins=30, color='steelblue', edgecolor='white')
```

**설명:**
- 21개 피처를 5×5 그리드에 한 번에 시각화한다.
- 이진 피처(0/1)는 막대가 두 개만 나타나고, BMI·Age처럼 연속형 피처는 분포 형태를 확인할 수 있다.
- BMI는 오른쪽 꼬리가 긴 분포(12~98)를 보이며, 이것이 이상치 처리가 필요한 이유다.

---

### 섹션 4 — 타겟별 피처 분포 비교

```python
continuous_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income', 'Education']
for i, col in enumerate(continuous_features):
    df.boxplot(column=col, by='Diabetes_binary', ax=ax, grid=False)
```

**설명:**
- `boxplot(by='Diabetes_binary')`로 당뇨군(1)과 정상군(0)의 분포를 나란히 비교한다.
- **BMI**: 당뇨군 중앙값이 정상군보다 뚜렷이 높다 → 비만과 당뇨의 연관성.
- **Age**: 당뇨군이 전반적으로 고령층에 집중 → 노화가 위험 요인.
- **Income**: 당뇨군이 저소득 구간에 더 많이 분포 → 사회경제적 요인.

---

### 섹션 5 — 상관관계 히트맵

```python
corr = df.corr(numeric_only=True)
top_corr = (corr['Diabetes_binary']
    .drop('Diabetes_binary')
    .sort_values(key=lambda values: values.abs(), ascending=False)
    .head(5))

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
```

**설명:**
- `corr(numeric_only=True)`로 모든 수치형 피처 간 피어슨 상관계수를 계산한다.
- `Diabetes_binary`와 상관계수가 높은 상위 5개 피처: GenHlth, HighBP, BMI, Age, DiffWalk.
- `cmap='coolwarm'`: 양의 상관은 붉은색, 음의 상관은 파란색, 중립은 흰색으로 표현.
- 이 상관관계 순위는 이후 통계 검정(STEP 3) 결과와 일치하는지 교차 확인하는 근거로 쓴다.

---

### 섹션 6 — 의학적 해석

| 피처 | 당뇨와의 관계 |
|------|---------------|
| BMI | 높을수록 인슐린 저항성 증가 |
| HighBP | 고혈압과 당뇨는 공통 심혈관 대사 경로 |
| Age | 중년 이후 위험 급상승 |
| GenHlth | 전반적 건강 자기 평가가 낮을수록 만성 질환 부담 |
| PhysActivity | 운동 부족은 수정 가능한 위험 요인 |

---

### 섹션 7 — 이상치 및 결측치 확인

```python
bmi_outlier_count = (df['BMI'] > 60).sum()
total_missing     = df.isnull().sum().sum()
```

- BMI 최대값 98 — 실제 고도 비만 환자값으로 의학적으로 유효하다.
- 결측치 0 → 중앙값 대체는 파이프라인 안전망으로만 남긴다.

---

## 4. STEP 3 — 통계 분석 (`03_statistical_analysis.ipynb`)

### 목표

머신러닝 없이 고전 통계 검정만으로 당뇨 위험 요인을 규명한다.  
이진 피처 → 카이제곱 검정, 연속형 피처 → 독립표본 t-검정, 전체 → 로지스틱 회귀 오즈비.

---

### 섹션 1 — 데이터 로드

```python
df = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test = joblib.load(SPLIT_PATH)

binary_features     = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', ...]
continuous_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income', 'GenHlth']
```

**설명:**
- 통계 검정(카이제곱, t-검정)은 **원본 데이터**를 사용한다 — 비율을 보존해야 실제 모집단 특성을 반영하기 때문이다.
- 로지스틱 회귀는 **전처리 완료 데이터**(`train_test_split.pkl`)를 사용한다 — 스케일링이 없으면 계수 크기가 피처 단위에 따라 왜곡되기 때문이다.

---

### 섹션 2 — 그룹별 기술통계 비교

```python
group_stats = df.groupby('Diabetes_binary')[continuous_features].agg(['mean', 'std']).round(3)
mean_diff   = (group_means.loc[1.0] - group_means.loc[0.0]).sort_values(ascending=False)

sns.heatmap(group_means, annot=True, fmt='.2f', cmap='coolwarm')
```

**설명:**
- `groupby('Diabetes_binary')`로 당뇨군(1)과 정상군(0)을 나눠 통계를 계산한다.
- `agg(['mean', 'std'])`로 평균과 표준편차를 동시에 계산한다.
- 히트맵의 색상 강도가 두 그룹 간 차이를 시각적으로 보여준다.

---

### 섹션 3 — 카이제곱 검정 (이진 피처)

```python
for col in binary_features:
    ct = pd.crosstab(df[col], df['Diabetes_binary'])
    chi2, p_value, dof, expected = chi2_contingency(ct)
    chi2_results.append({'Feature': col, 'Chi2': chi2, 'p-value': p_value, 'Significant': p_value < 0.05})
```

**설명:**
- `pd.crosstab()`으로 피처값(0/1)과 당뇨 여부(0/1)의 2×2 분할표를 만든다.
- `chi2_contingency()`는 두 범주형 변수가 독립인지 검정한다. p < 0.05이면 유의미한 연관이 있다.
- **결과: 14개 이진 피처 모두 p < 0.05로 유의함** — 이 데이터셋에서 이진 피처 전체가 당뇨와 통계적으로 연관된다.
- 바차트로 Chi2 통계량 크기를 비교 — 값이 클수록 연관성이 강하다.

---

### 섹션 4 — 독립표본 t-검정 (연속형 피처)

```python
diabetic = df[df['Diabetes_binary'] == 1]
normal   = df[df['Diabetes_binary'] == 0]

for col in continuous_features:
    t_stat, p_value = ttest_ind(diabetic[col], normal[col], equal_var=False)
```

**설명:**
- `equal_var=False`는 Welch의 t-검정으로, 두 그룹의 분산이 다를 경우에도 정확하다.
- 두 그룹의 평균이 유의미하게 다른지 검정한다. 표본이 크면 매우 작은 차이도 유의해지므로 **t-통계량 크기**가 실질적 중요도를 나타낸다.

| 순위 | 피처 | 당뇨군 평균 | 정상군 평균 | t-통계량 |
|------|------|------------|------------|---------|
| 1 | GenHlth | 3.291 | 2.385 | **156.1** |
| 2 | Age | 9.379 | 7.814 | 111.3 |
| 3 | BMI | 31.944 | 27.806 | 99.9 |
| 4 | Income | 5.210 | 6.190 | -78.1 (저소득일수록 위험) |
| 5 | PhysHlth | 7.954 | 3.641 | 69.0 |

---

### 섹션 5 — 로지스틱 회귀 오즈비 ← 핵심 섹션

```python
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_df, y_train)

coef_df = pd.DataFrame({
    'Feature':    feature_names,
    'Coefficient': lr.coef_[0],
    'Odds Ratio':  np.exp(lr.coef_[0]),
})
```

**설명:**
- 로지스틱 회귀 계수 `β`를 지수 변환(`e^β`)하면 **오즈비(Odds Ratio)**가 된다.
- **오즈비 해석**: OR = 2.0이면 해당 피처가 1단위 증가할 때 당뇨 위험이 2배가 된다는 의미다.
- OR > 1: 위험 증가 / OR < 1: 위험 감소 / OR = 1.0 기준선(효과 없음).
- 전처리된 데이터(표준화 완료)를 사용했으므로 피처 간 계수 크기를 직접 비교할 수 있다.

| 순위 | 피처 | 오즈비 | 해석 |
|------|------|--------|------|
| 1 | GenHlth | **1.977** | 건강 상태 1단계 나쁠수록 당뇨 위험 약 2배 |
| 2 | BMI | 1.743 | 표준화 1단위 증가 시 약 1.7배 |
| 3 | Age | 1.682 | 연령 증가에 따른 위험 상승 |
| 4 | HighBP | 1.439 | 고혈압 있을 시 약 1.4배 |
| 5 | HighChol | 1.346 | 고콜레스테롤 있을 시 약 1.3배 |

---

## 5. STEP 4 — 의사결정 및 정책 제안 (`04_decision.ipynb`)

### 목표

마지막 노트북에서는 모델 예측 확률을 실제 정책 판단에 사용할 수 있도록 위험 단계로 바꾼다.
단순히 “맞췄다/틀렸다”가 아니라, 어떤 사람을 먼저 관리해야 하는지와 어떤 정책 개입이 필요한지를 제안하는 단계다.

### 위험 단계 분류

```python
def classify_risk(prob):
    if prob >= 0.6:
        return '높음'
    elif prob >= 0.3:
        return '중간'
    return '낮음'

prob_df['risk_tier'] = prob_df['prob'].apply(classify_risk)
```

발표할 때는 이 기준이 절대적인 의학 기준이 아니라, 모델 확률을 정책적으로 해석하기 위한 분석 기준이라고 설명한다.

### 고위험군 프로파일

고위험군은 BMI와 주관적 건강 상태가 높고, 고혈압 비율이 높으며, 신체 활동이 부족한 방향으로 나타난다.
이 결과는 앞의 EDA와 통계 분석에서 확인한 위험 요인과 일관된다.

| 항목 | 해석 |
|------|------|
| BMI | 높을수록 당뇨 위험 증가 |
| HighBP | 고혈압이 있는 집단의 위험 증가 |
| PhysActivity | 신체 활동 부족 시 위험 증가 |
| Age | 연령대가 높을수록 위험 증가 |
| GenHlth | 주관적 건강 상태가 나쁠수록 위험 증가 |

### 정책 시나리오

| 시나리오 | 대상 | 개입 방식 |
|---------|------|----------|
| A | 고위험군 전체 | 정기 혈당 검사 의무화 |
| B | BMI 30 이상 고위험군 | 체중 관리 프로그램 연계 |
| C | 운동 부족 중·고위험군 | 지역 운동 프로그램 제공 |
| D | 흡연과 고혈압이 동반된 중·고위험군 | 금연·혈압 관리 패키지 |

이 부분은 발표의 결론과 직접 연결된다. 모델은 예측 도구이고, 정책 시나리오는 그 예측을 실제 예방 전략으로 바꾸는 해석 단계다.

### 임계값 선택

기본 임계값에서 Recall은 약 0.761이고, Recall 0.8 이상을 목표로 하면 임계값을 약 0.460까지 낮출 수 있다.
이 경우 더 많은 위험군을 찾을 수 있지만 Precision은 낮아진다.
의료 선별 목적에서는 미탐지 비용이 더 크므로, 본 프로젝트는 Recall 중심 임계값을 권장한다.

## 6. 결론 및 한계점

### 핵심 발견

1. **가장 강한 위험 요인**: GenHlth(전반 건강 자기 평가), BMI, 연령 — t-통계량 및 오즈비 기준 상위 3위
2. **수정 가능한 위험 요인**: PhysActivity(운동), Smoker(흡연), BMI — 생활 습관 개입 가능
3. **사회경제적 요인**: Income이 음의 오즈비 방향으로 유의 — 저소득층이 더 취약
4. **모든 이진 피처**: 카이제곱 검정 결과 14개 전체가 p < 0.05로 당뇨와 통계적 연관 확인

### 정책 제언

통계 분석에서 도출된 위험 요인을 기반으로 개입 가능성이 높은 순서로 제안한다.

| 우선순위 | 대상 | 근거 | 개입 방식 |
|---------|------|------|----------|
| 1순위 | BMI ≥ 30 | 오즈비 1.743, 수정 가능 | 체중 관리 프로그램 연계 |
| 2순위 | PhysActivity = 0 | EDA에서 당뇨군 비율 높음, 수정 가능 | 지역 운동 프로그램 무료 제공 |
| 3순위 | 흡연 + 고혈압 복합 | 카이제곱 유의, 복합 위험 | 금연·혈압 관리 패키지 |
| 4순위 | 고령층 (Age ≥ 9) | t-통계량 2위(111.3), 수정 불가 | 정기 혈당 검사 의무화 |

### 한계점

1. **교차 설계 연구 아님** — BRFSS는 단면 조사로 인과관계보다는 연관성을 보여준다.
2. **자기 보고 데이터** — 설문 응답 기반이라 측정 오류(recall bias) 가능성이 있다.
3. **표본 크기 효과** — 253,680행에서는 매우 작은 차이도 통계적으로 유의해지므로 효과 크기(t-통계량, 오즈비)로 실질적 중요도를 판단해야 한다.

---

*분석 도구: Python 3, pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, scipy*

---
