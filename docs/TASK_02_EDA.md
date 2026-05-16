# TASK 01 — EDA (`01_EDA.ipynb`)

## 역할

Codex가 수행. 완료 후 Claude가 결과를 리뷰하고 전처리 전략을 결정한다.

## 전제 조건

- `data/diabetes_indicators.csv` 존재 (Kaggle에서 수동 다운로드)
- TASK 00 완료 (디렉토리 구조, 노트북 셸 존재)

## 목표

데이터 구조를 파악하고 타겟과 피처 간의 관계를 시각화한다.  
분석 코드를 `01_EDA.ipynb`에 채운다.

---

## 작업 목록

### 섹션 1 — 데이터 로드 및 기초 정보

```python
df = pd.read_csv('../data/diabetes_indicators.csv')

print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.head())
```

출력해야 할 것:
- shape: (행 수, 열 수)
- dtypes: 모든 피처 타입 확인
- describe: 수치 요약 통계

### 섹션 2 — 타겟 클래스 분포

```python
counts = df['Diabetes_binary'].value_counts()
ratios = df['Diabetes_binary'].value_counts(normalize=True)
```

- 막대 그래프로 count와 비율(%) 동시 표시
- 예상 비율: 정상 ~86%, 당뇨/전당뇨 ~14%
- 그림을 `outputs/figures/01_class_distribution.png`에 저장

### 섹션 3 — 피처별 분포 히스토그램

21개 피처 전체를 서브플롯 격자로 한 번에 출력한다.

```python
fig, axes = plt.subplots(5, 5, figsize=(20, 16))
for i, col in enumerate(df.columns[:-1]):  # 타겟 제외
    df[col].hist(ax=axes.flatten()[i], bins=30)
    axes.flatten()[i].set_title(col)
plt.tight_layout()
```

- 그림을 `outputs/figures/01_feature_distributions.png`에 저장

### 섹션 4 — 타겟별 피처 분포 비교

연속형 피처(BMI, MentHlth, PhysHlth 등)에 대해 타겟 값(0/1)별 boxplot을 그린다.

```python
continuous_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income', 'Education']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(continuous_features):
    df.boxplot(column=col, by='Diabetes_binary', ax=axes.flatten()[i])
plt.suptitle('Feature Distribution by Diabetes Status')
plt.tight_layout()
```

- 그림을 `outputs/figures/01_feature_by_target.png`에 저장

### 섹션 5 — 상관관계 히트맵

```python
corr = df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
```

- 타겟(`Diabetes_binary`)과 상관관계 높은 피처 상위 5개를 별도 출력한다
- 그림을 `outputs/figures/01_correlation_heatmap.png`에 저장

### 섹션 6 — 의학적 해석 메모 (markdown 셀)

아래 피처들에 대해 당뇨와의 관계를 한 줄씩 작성한다:

| 피처 | 당뇨와의 관계 |
|------|--------------|
| BMI | 비만(BMI ≥ 30)은 인슐린 저항성 증가와 직결 |
| HighBP | 고혈압은 당뇨 위험 인자이자 공동 발생 빈도 높음 |
| HighChol | 이상지질혈증은 대사 이상과 동반 |
| Smoker | 흡연은 인슐린 감수성 저하 |
| PhysActivity | 운동 부족은 당뇨 위험 요인 |
| Age | 45세 이상부터 발병 위험 급증 |
| GenHlth | 전반적 건강 상태 자기 보고 (1=매우 좋음, 5=나쁨) |
| HvyAlcoholConsump | 과도한 음주는 간 기능 및 혈당 조절 영향 |

### 섹션 7 — 이상치 후보 확인 (다음 단계 준비)

```python
# BMI 범위 확인: 정상 10~80
print(f"BMI range: {df['BMI'].min()} ~ {df['BMI'].max()}")
print(f"BMI outlier (>60) count: {(df['BMI'] > 60).sum()}")

# 0이어선 안 되는 피처 확인
zero_check = ['BMI']
for col in zero_check:
    print(f"{col} zero count: {(df[col] == 0).sum()}")
```

결측치 현황도 확인한다:
```python
print(df.isnull().sum())
print(f"Total missing: {df.isnull().sum().sum()}")
```

---

## 완료 기준

- [ ] 7개 섹션이 모두 실행 가능한 셀로 존재한다
- [ ] `outputs/figures/` 에 4개 그림 파일이 저장된다
- [ ] 상관관계 상위 5개 피처가 텍스트로 출력된다
- [ ] 결측치 현황 출력이 포함된다
- [ ] 의학적 해석 markdown 셀이 존재한다

## 하지 말 것

- 전처리 실행 금지 (이 단계는 탐색만)
- 모델 학습 금지
- `src/` 함수 구현 금지 (이 단계에서는 인라인 코드로 작성)
- 그림 파일을 노트북 외부에서 별도 스크립트로 저장하지 않는다

---

## Claude 리뷰 포인트 (Codex 완료 후)

1. 클래스 불균형 비율 수치 확인 → Phase 2 불균형 처리 전략 결정
2. 결측치 현황 → 결측치 처리 방식 결정
3. BMI 이상치 규모 → IQR vs Z-score 선택 방향
4. 상관관계 높은 피처 → 피처 선택 여부 결정
