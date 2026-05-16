# TASK 00 — 프로젝트 초기 세팅

## 역할

Codex가 수행. Claude가 검토.

## 목표

프로젝트 디렉토리 구조, 의존성 파일, 노트북 셸, src 모듈 스텁을 생성한다.  
코드를 실행하거나 데이터를 처리하지 않는다. 구조와 골격만 만든다.

---

## 작업 목록

### 1. 디렉토리 생성

아래 디렉토리를 생성한다 (없는 것만):

```
diabetes_project/
├── data/
├── notebooks/
├── src/
└── outputs/
    ├── figures/
    └── models/
```

### 2. requirements.txt 생성

`diabetes_project/requirements.txt`:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
imbalanced-learn>=0.11
xgboost>=1.7
shap>=0.42
scipy>=1.11
joblib>=1.3
jupyter
```

### 3. 노트북 셸 생성

각 노트북은 아래 구조로 생성한다. 셀 내용은 CONTEXT 헤더와 섹션 마커만 포함한다. 실제 구현 코드는 각 TASK에서 채운다.

#### `notebooks/01_EDA.ipynb`

셀 1 (markdown):
```markdown
# STEP 1 — EDA
BRFSS Diabetes Health Indicators 데이터셋의 구조를 파악하고 피처-타겟 관계를 시각화한다.
```

셀 2 (code):
```python
# CONTEXT: 당뇨 위험군 분류 프로젝트
# DATASET: BRFSS Diabetes Health Indicators (253,680행, 21 피처)
# TARGET: Diabetes_binary (0/1), 클래스 불균형 86:14
# GOAL: 데이터 구조 파악 및 피처-타겟 관계 시각화
# PRIORITY: Recall 최대화 (미탐지 비용 > 오탐지 비용)
# OUTPUT: 타겟 분포, 피처 분포, 상관관계 히트맵
```

셀 3 (code):
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.dpi'] = 150
```

#### `notebooks/02_preprocessing.ipynb`

셀 1 (markdown):
```markdown
# STEP 2 — 전처리
이상치·결측치·클래스 불균형 처리 전략을 비교하고 최적 파이프라인을 구성한다.
```

셀 2 (code):
```python
# CONTEXT: 당뇨 위험군 분류 프로젝트
# DATASET: BRFSS Diabetes Health Indicators (253,680행, 21 피처)
# TARGET: Diabetes_binary (0/1), 클래스 불균형 86:14
# GOAL: 이상치/결측치/불균형 전략 비교 실험 → 최적 sklearn Pipeline 구성
# PRIORITY: Recall 최대화
# OUTPUT: 전략 비교 시각화, 최종 전처리 파이프라인, train/test split 저장
```

셀 3 (code):
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

plt.rcParams['figure.dpi'] = 150
```

#### `notebooks/03_modeling.ipynb`

셀 1 (markdown):
```markdown
# STEP 3 — 모델링
로지스틱 회귀, 랜덤 포레스트, XGBoost를 비교하고 SHAP으로 피처 중요도를 분석한다.
```

셀 2 (code):
```python
# CONTEXT: 당뇨 위험군 분류 프로젝트
# DATASET: BRFSS Diabetes Health Indicators (253,680행, 21 피처)
# TARGET: Diabetes_binary (0/1)
# GOAL: 3개 모델 비교 + SHAP 피처 중요도 분석
# PRIORITY: Recall 최대화
# OUTPUT: 모델별 평가 지표 테이블, ROC Curve, SHAP summary/beeswarm plot
```

셀 3 (code):
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)

plt.rcParams['figure.dpi'] = 150
```

#### `notebooks/04_decision.ipynb`

셀 1 (markdown):
```markdown
# STEP 4 — 의사결정
위험군 프로파일링과 예방 정책 시나리오를 작성한다.
```

셀 2 (code):
```python
# CONTEXT: 당뇨 위험군 분류 프로젝트
# DATASET: BRFSS Diabetes Health Indicators (253,680행, 21 피처)
# TARGET: Diabetes_binary (0/1)
# GOAL: 위험도 3단계 분류, 고위험군 페르소나 작성, 정책 시나리오 4개 수치화
# PRIORITY: Recall / threshold 조정 트레이드오프
# OUTPUT: 위험군 분류표, 그룹별 피처 평균 비교, threshold 곡선
```

셀 3 (code):
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

plt.rcParams['figure.dpi'] = 150
```

### 4. src 모듈 스텁 생성

#### `src/__init__.py`
빈 파일.

#### `src/preprocessing.py`

```python
"""Reusable preprocessing functions for diabetes project."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline


def load_data(path: str) -> pd.DataFrame:
    """Load dataset and do minimal dtype validation."""
    raise NotImplementedError


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return boolean mask of outliers using IQR method."""
    raise NotImplementedError


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of outliers using Z-score method."""
    raise NotImplementedError


def build_pipeline(imputer, scaler, sampler=None) -> Pipeline:
    """Assemble sklearn-compatible pipeline from components."""
    raise NotImplementedError
```

#### `src/modeling.py`

```python
"""Reusable modeling and evaluation functions for diabetes project."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, classification_report
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Return dict of Recall, Precision, F1, AUC-ROC."""
    raise NotImplementedError


def make_metrics_table(results: dict) -> pd.DataFrame:
    """Convert {model_name: metrics_dict} to comparison DataFrame."""
    raise NotImplementedError
```

#### `src/visualization.py`

```python
"""Reusable visualization functions for diabetes project."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_class_distribution(y: np.ndarray, ax=None) -> plt.Axes:
    """Bar chart of target class counts and ratios."""
    raise NotImplementedError


def plot_roc_curves(models: dict, X_test, y_test, ax=None) -> plt.Axes:
    """Overlay ROC curves for multiple models."""
    raise NotImplementedError


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save figure to outputs/figures/."""
    raise NotImplementedError
```

---

## 완료 기준

- [ ] 디렉토리 구조가 위 트리와 일치한다
- [ ] `requirements.txt`가 존재하고 위 패키지를 포함한다
- [ ] 4개 노트북 파일이 존재하고 CONTEXT 헤더 셀을 포함한다
- [ ] `src/` 에 `__init__.py`, `preprocessing.py`, `modeling.py`, `visualization.py` 가 존재한다
- [ ] 모든 함수는 `raise NotImplementedError` 스텁으로 존재한다 (실제 구현은 TASK_01~04에서)

## 하지 말 것

- 데이터 로드 또는 실행 금지 (데이터 파일이 아직 없을 수 있음)
- 스텁 함수를 미리 구현하지 않는다
- 노트북에 분석 코드를 미리 채우지 않는다
