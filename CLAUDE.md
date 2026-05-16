# 당뇨 위험군 분류 및 예방 정책 제안

데이터 분석 기말 프로젝트. BRFSS 데이터셋으로 당뇨 위험군을 분류하고 예방 정책을 제안한다.  
**Claude = 계획·설계·리뷰 / Codex = 구현**

---

## 협업 워크플로우

```
Claude (계획)  →  Codex (구현)  →  Claude (리뷰)
```

- Claude는 분석 전략, 전처리 방향, 모델 선택 근거, 해석을 담당한다.
- Codex는 노트북 코드, 시각화, src/ 모듈을 구현한다.
- 구현 요청 시 Codex에게 전달할 컨텍스트와 목표를 명확히 작성한다.

---

## 행동 원칙

### 1. 코딩 전 생각하기

가정은 명시적으로 밝힌다. 불확실하면 질문한다.

- 해석이 여러 가지인 경우 선택지를 제시한다.
- 더 단순한 방법이 있으면 말한다.
- 불명확한 부분이 있으면 멈추고 구체적으로 물어본다.

### 2. 단순함 우선

요청한 것만 최소한으로 구현. 추측성 코드 금지.

- 요청하지 않은 기능은 추가하지 않는다.
- 단일 사용 코드에 추상화를 만들지 않는다.
- 200줄이 50줄로 가능하다면 다시 작성한다.

### 3. 최소 변경

반드시 필요한 것만 수정.

- 인접한 코드, 주석, 포맷을 "개선"하지 않는다.
- 고장 나지 않은 것은 리팩토링하지 않는다.
- 자신의 변경으로 불필요해진 import/변수/함수는 제거한다.

### 4. 목표 중심 실행

작업을 검증 가능한 목표로 변환한다.

다단계 작업의 경우 계획을 먼저 제시한다:
```
1. [단계] → 검증: [확인 방법]
2. [단계] → 검증: [확인 방법]
```

---

## 프로젝트 구조

```
diabetes_project/
├── data/
│   └── diabetes_indicators.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_decision.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   └── models/
└── requirements.txt
```

---

## 데이터셋 요약

| 항목 | 내용 |
|------|------|
| 출처 | BRFSS Diabetes Health Indicators (Kaggle) |
| 행 수 | 253,680 |
| 피처 수 | 21 (BMI, 혈압, 나이, 흡연, 운동 여부, 콜레스테롤 등) |
| 타겟 | `Diabetes_binary` (0: 정상, 1: 당뇨/전당뇨) |
| 클래스 비율 | 정상 약 86% / 당뇨 약 14% (불균형) |
| 주 평가 지표 | Recall (미탐지 비용이 오탐지 비용보다 크므로) |

---

## 단계별 목표 요약

| 단계 | 노트북 | 핵심 산출물 |
|------|--------|-------------|
| STEP 1 | 01_EDA | 피처-타겟 관계 시각화, 불균형 비율 수치화 |
| STEP 2 | 02_preprocessing | 이상치·결측치·불균형 전략 비교 실험, 최적 파이프라인 |
| STEP 3 | 03_modeling | 3개 모델 비교, SHAP 피처 중요도 |
| STEP 4 | 04_decision | 위험군 프로파일 + 정책 시나리오 4개 |

---

## OOM 방지 규칙

253k 행 데이터셋 작업 시 메모리 관리에 주의한다.

- 파일 처리는 한 번에 5개 이하로 제한
- 대용량 파일(1MB 이상)은 분할하여 순차 처리
- 불필요한 도구 동시 실행 금지
- SHAP 계산은 샘플 서브셋으로 먼저 검증 후 전체 적용

---

## Claude의 역할: 계획 작성 기준

Codex에게 작업을 넘길 때 아래 형식으로 컨텍스트를 제공한다:

```python
# CONTEXT: 당뇨 위험군 분류 프로젝트
# DATASET: BRFSS Diabetes Health Indicators (253,680행, 21 피처)
# TARGET: Diabetes_binary (0/1), 클래스 불균형 86:14
# GOAL: [이 노트북/함수의 구체적 목표]
# PRIORITY: Recall 최대화 (미탐지 비용 > 오탐지 비용)
# OUTPUT: [기대하는 출력물 — 그래프, 테이블, 모델 파일 등]
```

---

## 주요 라이브러리

```
pandas, numpy, matplotlib, seaborn
scikit-learn, imbalanced-learn
xgboost, shap
jupyter
```
