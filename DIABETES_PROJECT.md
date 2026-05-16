# 당뇨 위험군 분류 및 예방 정책 제안
> 기말 데이터 분석 프로젝트 | Python | 개인 프로젝트

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 당뇨 위험군을 분류하고, 위험군 프로파일 기반 예방 정책 제안 |
| **데이터셋** | [BRFSS Diabetes Health Indicators Dataset (Kaggle)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| **행 수** | 약 253,680행 |
| **피처 수** | 21개 (BMI, 혈압, 나이, 흡연, 운동 여부, 콜레스테롤 등) |
| **타겟** | `Diabetes_binary` (0: 정상, 1: 당뇨/전당뇨) |
| **평가 지표** | Recall 중심 (미탐지 비용이 크므로) + F1, AUC-ROC |

---

## 📁 디렉토리 구조

```
diabetes_project/
├── data/
│   └── diabetes_indicators.csv          # Kaggle에서 다운로드
├── notebooks/
│   ├── 01_EDA.ipynb                     # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb           # 전처리 (핵심)
│   ├── 03_modeling.ipynb                # 모델 학습 및 평가
│   └── 04_decision.ipynb                # 의사결정 및 정책 제안
├── src/
│   ├── preprocessing.py                 # 전처리 함수 모듈
│   ├── modeling.py                      # 모델 학습 함수 모듈
│   └── visualization.py                 # 시각화 함수 모듈
├── outputs/
│   ├── figures/                         # 시각화 결과 저장
│   └── models/                          # 학습된 모델 저장 (.pkl)
├── requirements.txt
└── README.md
```

---

## 🔧 환경 설정

### requirements.txt
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=1.7
shap>=0.42
jupyter
```

### 설치
```bash
pip install -r requirements.txt
```

---

## 📊 단계별 작업 명세

---

### STEP 1 · EDA (`01_EDA.ipynb`)

**목표**: 데이터 구조 파악 및 타겟과 피처 간 관계 시각화

#### 작업 목록
- [ ] 데이터 로드 및 기초 정보 확인 (`shape`, `dtypes`, `describe`)
- [ ] 타겟 클래스 분포 시각화 → 불균형 비율 수치화
- [ ] 피처별 분포 히스토그램 (21개 피처 일괄)
- [ ] 타겟별 피처 분포 비교 (박스플롯 / violinplot)
- [ ] 상관관계 히트맵
- [ ] 의학적 해석 메모 작성 (각 피처가 당뇨와 어떤 관계인지)

#### 주요 확인 포인트
```python
# 클래스 불균형 확인
df['Diabetes_binary'].value_counts(normalize=True)
# 예상: 정상 86%, 당뇨 14% 내외

# 이상치 후보 피처 확인
# BMI: 정상 범위 10~80 / 0값 없어야 함
# Age: 1~13 (범주형 인코딩됨)
```

---

### STEP 2 · 전처리 (`02_preprocessing.ipynb`) ⭐ 핵심

**목표**: 다양한 전처리 전략을 비교하여 최적 파이프라인 선택

#### 2-1. 이상치 탐지 및 처리
- [ ] IQR 방식으로 이상치 범위 계산
- [ ] Z-score 방식으로 이상치 범위 계산
- [ ] **두 방식 결과 비교 시각화** (어느 방식이 더 적절한지 근거 제시)
- [ ] 선택한 방식으로 이상치 처리 (제거 or 대체 결정 및 근거 명시)

```python
# IQR 방식
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df['BMI'] < Q1 - 1.5*IQR) | (df['BMI'] > Q3 + 1.5*IQR)

# Z-score 방식
from scipy import stats
z_scores = np.abs(stats.zscore(df['BMI']))
outlier_mask_z = z_scores > 3
```

#### 2-2. 결측치 처리
- [ ] 결측치 현황 파악 (`isnull().sum()`)
- [ ] **3가지 대체 전략 비교**:
  - 평균값 대체 (Mean Imputation)
  - 중앙값 대체 (Median Imputation)
  - KNN 대체 (KNN Imputation)
- [ ] 각 전략 적용 후 분포 변화 시각화
- [ ] 선택 전략 근거 명시

```python
from sklearn.impute import SimpleImputer, KNNImputer

mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')
knn_imputer = KNNImputer(n_neighbors=5)
```

#### 2-3. 클래스 불균형 처리
- [ ] 현재 불균형 비율 시각화
- [ ] **3가지 전략 비교**:
  - SMOTE (오버샘플링)
  - Random Undersampling
  - SMOTE + Undersampling 결합
- [ ] 각 전략 적용 후 모델 Recall 비교 (간단한 로지스틱 회귀로 테스트)
- [ ] 최적 전략 선택 및 근거 명시

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

smote = SMOTE(random_state=42)
rus = RandomUnderSampler(random_state=42)
```

#### 2-4. 피처 엔지니어링
- [ ] 범주형 피처 확인 및 인코딩 (이미 수치형이므로 확인만)
- [ ] 연속형 피처 스케일링 (StandardScaler vs MinMaxScaler 비교)
- [ ] 불필요 피처 제거 여부 판단 (VIF 기반 다중공선성 확인)

#### 2-5. 최종 전처리 파이프라인 정리
- [ ] 선택한 전략들을 sklearn Pipeline으로 통합
- [ ] Train/Test split (8:2, stratify=True)
- [ ] 전처리 전후 데이터 통계 비교 테이블 작성

---

### STEP 3 · 모델링 (`03_modeling.ipynb`)

**목표**: 3개 모델 비교 후 최적 모델 선택, 피처 중요도 분석

#### 3-1. 모델 학습 및 비교
- [ ] 로지스틱 회귀 (Baseline)
- [ ] 랜덤 포레스트
- [ ] XGBoost

#### 3-2. 평가 지표
- [ ] Confusion Matrix 시각화 (3개 모델 나란히)
- [ ] Recall, Precision, F1, AUC-ROC 비교 테이블
- [ ] ROC Curve 오버레이 시각화
- [ ] **Recall을 주 지표로 선택한 이유** 서술 (미탐지 = 미진단 환자 발생)

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

metrics = {
    'Logistic Regression': {...},
    'Random Forest': {...},
    'XGBoost': {...}
}
```

#### 3-3. 피처 중요도 분석
- [ ] 랜덤 포레스트 Feature Importance 바차트
- [ ] XGBoost Feature Importance 바차트
- [ ] SHAP Value 분석 (선택 모델 기준)
  - Summary plot
  - Beeswarm plot
- [ ] **"어떤 요인이 당뇨 발병에 가장 영향을 주는가"** 해석 작성

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

### STEP 4 · 의사결정 (`04_decision.ipynb`) ⭐ 핵심

**목표**: 분석 결과를 실제 정책 제안으로 연결

#### 4-1. 고위험군 프로파일링
- [ ] 모델 예측 확률 기반 위험 점수 산출
- [ ] 위험도 3단계 분류 (고위험 / 중위험 / 저위험)
- [ ] 각 그룹별 피처 평균 비교 테이블
- [ ] **위험군 페르소나 작성** (예: "BMI 35 이상, 운동 안 함, 50대 이상 남성")

#### 4-2. 정책 시나리오 제안 (4개)

| 시나리오 | 대상 | 개입 방식 | 기대 효과 |
|---------|------|----------|----------|
| 시나리오 A | 고위험군 전체 | 정기 혈당 검사 의무화 | 조기 진단율 향상 |
| 시나리오 B | BMI 고위험군 | 체중 관리 프로그램 연계 | BMI 감소 → 발병률 감소 |
| 시나리오 C | 운동 부족 그룹 | 지역 운동 프로그램 무료 제공 | 생활 습관 개선 |
| 시나리오 D | 흡연 + 고혈압 복합 | 금연·혈압 관리 패키지 지원 | 복합 위험 요소 제거 |

- [ ] 각 시나리오별 대상 인원 수 추정 (데이터 기반)
- [ ] 시나리오별 비용 효율성 간단 비교 (정성적)

#### 4-3. 모델 기반 효과 추정
- [ ] "조기 개입 시 고위험군 중 몇 %가 발병 전 탐지 가능한가" 수치화
- [ ] 임계값(threshold) 조정에 따른 Recall vs Precision 트레이드오프 시각화
- [ ] 최적 임계값 선택 근거 제시

---

## ⏱️ 작업 일정 (2주)

| 일차 | 작업 |
|------|------|
| Day 1~2 | 데이터 다운로드, EDA 완료 |
| Day 3~5 | 전처리 (이상치, 결측치, 불균형) |
| Day 6~7 | 피처 엔지니어링, 파이프라인 구성 |
| Day 8~9 | 모델 학습 및 비교 |
| Day 10~11 | SHAP 분석, 피처 중요도 해석 |
| Day 12~13 | 의사결정 시나리오 작성 |
| Day 14 | 시각화 정리, README 작성 |

---

## 📝 보고서 구성 (제출용)

1. **서론** - 문제 정의, 데이터 소개
2. **EDA** - 주요 시각화 및 인사이트
3. **전처리** - 각 단계별 비교 실험 결과 및 선택 근거
4. **모델링** - 모델 비교 및 최종 선택 근거
5. **피처 중요도** - SHAP 기반 해석
6. **의사결정** - 위험군 프로파일 + 정책 시나리오 4개
7. **결론** - 한계점 및 개선 방향

---

## 💡 Codex 활용 팁

각 노트북 상단에 아래 주석을 붙여서 Codex에게 컨텍스트 제공:

```python
# CONTEXT: 당뇨 위험군 분류 프로젝트
# DATASET: BRFSS Diabetes Health Indicators (253,680행, 21 피처)
# TARGET: Diabetes_binary (0/1)
# GOAL: [각 노트북 목표 작성]
# PRIORITY: Recall 최대화 (미탐지 비용 > 오탐지 비용)
```

Codex에게 요청할 때 유용한 프롬프트 패턴:
- `"IQR과 Z-score 이상치 탐지 결과를 나란히 비교하는 시각화 코드 작성해줘"`
- `"SMOTE, Undersampling, 결합 전략 세 가지를 로지스틱 회귀로 빠르게 비교하는 코드"`
- `"SHAP summary plot과 beeswarm plot을 같이 출력하는 코드"`
- `"위험도 3단계로 나눠서 각 그룹 피처 평균 비교 테이블 만들어줘"`
