# TASK 03 — 통계 분석 (`03_statistical_analysis.ipynb`)

## 역할

Codex가 수행. 완료 후 Claude가 한국어 패치 및 리뷰.

## 언어 규칙

- 모든 텍스트는 **영어**로 작성 (한국어 변환은 별도 처리)
- 코드 내 주석(`#`)은 작성하지 않는다 (발표용)
- 변수명·함수명은 영문 유지

## 전제 조건

- `data/diabetes_indicators.csv` 존재
- `outputs/models/train_test_split.pkl` 존재 (클리핑 적용된 전처리 데이터)

## 목표

ML 모델 없이 순수 통계 분석으로 당뇨 위험 요인을 규명한다.  
이진 피처는 카이제곱 검정, 연속형 피처는 t-검정으로 유의성을 검증하고,  
로지스틱 회귀 계수(오즈비)로 각 요인의 영향력 크기를 정량화한다.

---

## 사용 데이터

통계 검정은 원본 데이터 비율을 보존하기 위해 `diabetes_indicators.csv`를 직접 사용.  
로지스틱 회귀는 `train_test_split.pkl`의 전처리 완료 데이터 사용.

---

## 작업 목록

### 섹션 1 — 데이터 로드

```python
df = pd.read_csv('../data/diabetes_indicators.csv')
X_train, X_test, y_train, y_test = joblib.load('../outputs/models/train_test_split.pkl')

binary_features = ['HighBP','HighChol','CholCheck','Smoker','Stroke',
                   'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
                   'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','DiffWalk','Sex']
continuous_features = ['BMI','MentHlth','PhysHlth','Age','Education','Income','GenHlth']
```

### 섹션 2 — 그룹별 기술통계 비교

당뇨(1)·정상(0) 그룹별 연속형 피처 평균·표준편차를 나란히 비교한다.

```python
group_stats = df.groupby('Diabetes_binary')[continuous_features].agg(['mean','std']).round(3)
```

히트맵으로 그룹별 평균 차이 시각화.  
그림을 `outputs/figures/03_group_stats_heatmap.png`에 저장.

### 섹션 3 — 카이제곱 검정 (이진 피처)

```python
from scipy.stats import chi2_contingency

chi2_results = []
for col in binary_features:
    ct = pd.crosstab(df[col], df['Diabetes_binary'])
    chi2, p, dof, _ = chi2_contingency(ct)
    chi2_results.append({'Feature': col, 'Chi2': chi2, 'p-value': p, 'Significant': p < 0.05})

chi2_df = pd.DataFrame(chi2_results).sort_values('Chi2', ascending=False)
```

Chi2 통계량 기준 바차트 출력.  
그림을 `outputs/figures/03_chi2_results.png`에 저장.

### 섹션 4 — t-검정 (연속형 피처)

```python
from scipy.stats import ttest_ind

ttest_results = []
diabetic = df[df['Diabetes_binary'] == 1]
normal   = df[df['Diabetes_binary'] == 0]

for col in continuous_features:
    t_stat, p = ttest_ind(diabetic[col], normal[col])
    ttest_results.append({'Feature': col, 't-stat': t_stat, 'p-value': p,
                          'Mean (Diabetic)': diabetic[col].mean(),
                          'Mean (Normal)':   normal[col].mean()})

ttest_df = pd.DataFrame(ttest_results).sort_values('t-stat', ascending=False)
```

t-통계량 기준 바차트 출력.  
그림을 `outputs/figures/03_ttest_results.png`에 저장.

### 섹션 5 — 로지스틱 회귀 (통계 모델)

전처리 완료 데이터로 학습. ML이 아닌 통계적 확률 모델로 활용.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

feature_names = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
                 'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
                 'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
                 'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

coef_df = pd.DataFrame({
    'Feature':  feature_names,
    'Coeff':    lr.coef_[0],
    'Odds Ratio': np.exp(lr.coef_[0])
}).sort_values('Odds Ratio', ascending=False)
```

오즈비 바차트 출력 (1.0 기준선 포함).  
그림을 `outputs/figures/03_odds_ratio.png`에 저장.

### 섹션 6 — 주요 발견 정리 (markdown 셀)

검정 결과를 토대로 아래 항목을 작성한다:

```markdown
## Key Statistical Findings

**Significant binary features (chi-square, p < 0.05):** [all/most/list exceptions]

**Top features by group mean difference (t-test):**
1. [feature]: diabetic mean X vs normal mean Y
2. ...

**Strongest predictors by Odds Ratio:**
1. [feature]: OR = X (Y% higher odds per unit increase)
2. ...

→ These findings will directly inform the risk profiling and policy scenarios in Step 4.
```

---

## 저장 규칙

```python
import json
with open('path/to/03_statistical_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

---

## 완료 기준

- [ ] 4개 그림 파일이 `outputs/figures/`에 저장됨
- [ ] 카이제곱·t-검정 결과 테이블 출력됨
- [ ] 오즈비 바차트 1.0 기준선 포함
- [ ] 섹션 6 발견 정리 markdown 셀 존재
- [ ] 코드 내 주석 없음
