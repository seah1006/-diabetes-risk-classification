# TASK 04 — 의사결정 (`04_decision.ipynb`)

## 역할

Codex가 수행. 완료 후 Claude가 정책 시나리오 서술을 보완한다.

## 언어 규칙

노트북 내 모든 텍스트는 **한국어**로 작성한다.

- markdown 셀 제목, 설명, 해석, 페르소나 서술, 정책 시나리오 → 한국어
- 코드 내 주석(`#`) → **작성하지 않는다** (발표용)
- `print()` 출력 레이블 → 한국어
- 변수명·함수명은 영문 유지 (코드 가독성)

## 전제 조건

- TASK 03 완료: `outputs/models/best_model.pkl` 존재
- TASK 02 완료: `outputs/models/train_test_split.pkl` 존재
- 원본 데이터 `data/diabetes_indicators.csv` 존재

## 목표

모델 예측 결과를 위험도 3단계로 분류하고,  
고위험군 페르소나와 예방 정책 시나리오 4개를 데이터 수치 기반으로 작성한다.

---

## 작업 목록

### 섹션 1 — 데이터 및 모델 로드

```python
import joblib

best_model = joblib.load('../outputs/models/best_model.pkl')
X_train, X_test, y_train, y_test = joblib.load('../outputs/models/train_test_split.pkl')

df = pd.read_csv('../data/diabetes_indicators.csv')
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

feature_names = X.columns.tolist()
```

### 섹션 2 — 위험 점수 산출

```python
y_prob = best_model.predict_proba(X_test)[:, 1]

prob_df = pd.DataFrame({
    'prob':   y_prob,
    'actual': y_test.values if hasattr(y_test, 'values') else y_test
})

print(prob_df['prob'].describe())
```

### 섹션 3 — 위험도 3단계 분류

```python
def classify_risk(prob: float) -> str:
    if prob >= 0.6:
        return 'high'
    elif prob >= 0.3:
        return 'medium'
    return 'low'

prob_df['risk_tier'] = prob_df['prob'].apply(classify_risk)

tier_counts = prob_df['risk_tier'].value_counts()
tier_ratios = prob_df['risk_tier'].value_counts(normalize=True)

print(tier_counts)
print(tier_ratios.round(3))
```

파이 차트 또는 막대 그래프로 3단계 분포를 시각화한다.  
그림을 `outputs/figures/04_risk_tiers.png`에 저장.

### 섹션 4 — 그룹별 피처 평균 비교 테이블

```python
# test set에 원본 피처 결합
test_indices = y_test.index if hasattr(y_test, 'index') else range(len(y_test))
X_test_df = pd.DataFrame(X_test, columns=feature_names, index=test_indices)
X_test_df['risk_tier'] = prob_df['risk_tier'].values

group_means = X_test_df.groupby('risk_tier').mean().round(3)
print(group_means.T)  # 피처를 행으로, 위험 단계를 열로
```

이 테이블을 markdown 셀로도 출력한다 (`group_means.T.to_markdown()`).

### 섹션 5 — 고위험군 페르소나 (markdown 셀)

섹션 4 결과를 바탕으로 고위험군 특징을 서술한다.

```markdown
## 고위험군 페르소나

**"당뇨 고위험 프로파일"**

| 항목 | 고위험군 평균 | 저위험군 평균 |
|------|--------------|--------------|
| BMI | [값] | [값] |
| HighBP | [값] | [값] |
| PhysActivity | [값] | [값] |
| Age | [값] | [값] |
| GenHlth | [값] | [값] |

→ 대표 페르소나: BMI [X] 이상, [고혈압/흡연/운동 부족] 해당, [Y]대 이상
```

### 섹션 6 — 정책 시나리오 4개 수치화

각 시나리오의 대상 인원 수를 데이터 기반으로 추정한다.

```python
# 전체 데이터 기준 예측 확률 계산
# 주의: 전체 253k행에 predict_proba 적용 시 메모리 확인
# 필요 시 배치로 나눠서 처리

batch_size = 50000
all_probs = []
for i in range(0, len(X), batch_size):
    batch = X.iloc[i:i+batch_size]
    probs = best_model.predict_proba(batch)[:, 1]
    all_probs.extend(probs)

df['risk_prob'] = all_probs
df['risk_tier'] = pd.cut(df['risk_prob'],
                          bins=[0, 0.3, 0.6, 1.0],
                          labels=['low', 'medium', 'high'])

total = len(df)

# 시나리오 A: 고위험군 전체 → 정기 혈당 검사 의무화
scenario_a = (df['risk_tier'] == 'high').sum()
print(f"시나리오 A 대상: {scenario_a:,}명 ({scenario_a/total:.1%})")

# 시나리오 B: BMI 고위험군 (BMI ≥ 30) + 고위험 티어
scenario_b = ((df['risk_tier'] == 'high') & (df['BMI'] >= 30)).sum()
print(f"시나리오 B 대상: {scenario_b:,}명 ({scenario_b/total:.1%})")

# 시나리오 C: 운동 부족 그룹 (PhysActivity == 0) + 중/고위험
scenario_c = ((df['risk_tier'].isin(['medium', 'high'])) & (df['PhysActivity'] == 0)).sum()
print(f"시나리오 C 대상: {scenario_c:,}명 ({scenario_c/total:.1%})")

# 시나리오 D: 흡연 + 고혈압 복합 위험군 + 중/고위험
scenario_d = ((df['risk_tier'].isin(['medium', 'high'])) &
              (df['Smoker'] == 1) & (df['HighBP'] == 1)).sum()
print(f"시나리오 D 대상: {scenario_d:,}명 ({scenario_d/total:.1%})")
```

결과를 표로 정리한다:

| 시나리오 | 대상 | 개입 방식 | 데이터 기준 대상 인원 | 비율 |
|---------|------|----------|-------------------|----|
| A | 고위험군 전체 | 정기 혈당 검사 의무화 | [값] | [값] |
| B | BMI ≥ 30 + 고위험 | 체중 관리 프로그램 연계 | [값] | [값] |
| C | 운동 부족 + 중/고위험 | 지역 운동 프로그램 무료 제공 | [값] | [값] |
| D | 흡연+고혈압 복합 + 중/고위험 | 금연·혈압 관리 패키지 | [값] | [값] |

### 섹션 7 — Threshold 조정 트레이드오프

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.title('Precision / Recall vs Threshold')
plt.legend()
plt.axvline(x=0.3, color='red', linestyle='--', label='Current threshold')

plt.subplot(1, 2, 2)
plt.plot(recall[:-1], precision[:-1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
```

그림을 `outputs/figures/04_threshold_tradeoff.png`에 저장.

```python
# 임계값 선택 근거: F1 최대 임계값 vs Recall 0.8 보장 임계값 비교
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_f1_thresh  = thresholds[np.argmax(f1_scores[:-1])]
recall_80_thresh = thresholds[np.where(recall[:-1] >= 0.8)[0][-1]]

print(f"F1 최대 임계값: {best_f1_thresh:.3f}")
print(f"Recall ≥ 0.8 보장 임계값: {recall_80_thresh:.3f}")
```

### 섹션 8 — 조기 개입 효과 추정 (markdown 셀)

```markdown
## 조기 개입 효과 추정

- 전체 데이터 중 고위험군([값]명, [비율])을 조기에 개입한다면
- 현재 모델 Recall([값]) 기준으로 그 중 약 [값]%를 발병 전 탐지 가능
- 임계값을 [값]으로 낮추면 Recall이 [값]%까지 상승하나 Precision이 [값]%로 하락

→ **추천 임계값**: [값] — Recall [값]%, 정책 대상 인원 [값]명
```

---

## 완료 기준

- [ ] 위험도 3단계 분류 및 분포 시각화
- [ ] 그룹별 피처 평균 비교 테이블 (출력 + markdown)
- [ ] 고위험군 페르소나 markdown 셀
- [ ] 시나리오 4개 대상 인원 수치 계산
- [ ] Threshold 트레이드오프 그래프 저장
- [ ] 조기 개입 효과 추정 수치화
- [ ] 4개 그림 파일이 `outputs/figures/`에 저장됨

## 하지 말 것

- 정책 제안 내용을 데이터 수치 없이 정성적으로만 작성하지 않는다
- 전체 253k 행에 SHAP을 추가로 실행하지 않는다 (TASK 03에서 완료)
- 임계값을 임의로 확정하지 않는다 — 수치를 보여주고 선택은 Claude에게

---

## Claude 리뷰 포인트 (Codex 완료 후)

1. 시나리오별 대상 인원 현실성 검토
2. 최종 임계값 선택 및 근거 문서화
3. 정책 시나리오 서술 보완 (비용 효율성 정성 비교)
4. 결론 섹션 작성 (한계점 및 개선 방향)
