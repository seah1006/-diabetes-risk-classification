# 당뇨 위험군 분류 및 예방 정책 제안

기말평가 발표용 데이터 분석 프로젝트입니다. BRFSS 건강 지표 데이터를 사용해 당뇨 및 전당뇨 위험군을 분류하고, 미탐지 비용이 큰 의료 문제라는 점을 반영해 Recall(재현율)을 중심 지표로 사용합니다.

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 데이터 | BRFSS Diabetes Health Indicators Dataset |
| 규모 | 253,680행, 21개 피처 |
| 타겟 | `Diabetes_binary` (0: 정상, 1: 당뇨/전당뇨) |
| 클래스 비율 | 정상 약 86%, 당뇨/전당뇨 약 14% |
| 핵심 기준 | Recall 우선, 필요 시 Precision과의 균형 검토 |
| 최종 활용 | 위험군 분류, 고위험군 프로파일, 예방 정책 시나리오 |

## 현재 산출물

```text
diabetes_project/
├── data/
│   └── diabetes_indicators.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_statistical_analysis.ipynb
│   └── 04_decision.ipynb
├── src/
│   ├── preprocessing.py
│   ├── statistical_analysis.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   └── models/
│       ├── train_test_split.pkl
│       ├── preprocessing_pipeline.pkl
│       ├── best_model.pkl
│       ├── logistic_regression.pkl
│       ├── random_forest.pkl
│       └── xgboost.pkl
└── requirements.txt
```

## 발표 흐름

1. 문제 정의
   당뇨는 조기 발견과 생활습관 개입이 중요한 질환이므로, 정상과 당뇨/전당뇨를 구분하는 모델을 만들고 고위험군을 정책 대상으로 해석한다.

2. 전처리
   이상치, 결측치, 클래스 불균형 처리 전략을 비교한다. 클래스 불균형이 뚜렷하므로 SMOTE와 언더샘플링 결합 전략을 사용해 Recall을 높이는 방향으로 학습 데이터를 구성한다.

3. 탐색적 데이터 분석
   타겟 분포, 피처별 분포, 당뇨 여부별 차이, 상관관계를 시각화한다. BMI, 고혈압, 건강 상태, 연령, 보행 어려움 등이 주요 위험 신호로 드러난다.

4. 통계 분석
   카이제곱 검정, t-검정, 로지스틱 회귀 오즈비를 통해 위험 요인을 설명한다. 이 단계는 모델 예측 결과를 정책 제안으로 연결하기 위한 근거 역할을 한다.

5. 의사결정
   예측 확률을 낮음, 중간, 높음 3단계로 나누고 고위험군 프로파일을 정리한다. 이후 정기 혈당 검사, 체중 관리, 운동 프로그램, 금연·혈압 관리 패키지로 정책 시나리오를 제시한다.

## 발표 핵심 메시지

- 데이터는 정상 클래스가 많은 불균형 구조이므로 Accuracy보다 Recall이 중요하다.
- BMI, 고혈압, 주관적 건강 상태, 연령은 반복적으로 강한 위험 요인으로 확인된다.
- 모델 결과는 단순 예측에서 끝나지 않고, 고위험군을 정의하고 예방 정책 대상을 추정하는 데 사용된다.
- 임계값을 낮추면 더 많은 위험군을 찾을 수 있지만 Precision은 낮아지므로, 의료 선별 목적에서는 Recall 중심 선택이 타당하다.

## 실행 방법

```bash
pip install -r diabetes_project/requirements.txt
```

노트북은 `01_preprocessing.ipynb`부터 `04_decision.ipynb`까지 순서대로 실행합니다.
