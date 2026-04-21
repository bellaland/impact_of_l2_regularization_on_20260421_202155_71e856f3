# Downloaded Datasets

This directory contains locally prepared datasets for small-data regularization experiments.
Data files are intentionally ignored by git through `datasets/.gitignore`.

## Recommended Primary Datasets

### Dataset 1: sklearn Wine
- Source: `sklearn.datasets.load_wine`
- Size: 178 samples, 13 features
- Format: CSV
- Task: multiclass classification
- Splits: no official split; create stratified train/validation/test splits
- License: distributed with scikit-learn; originally from UCI
- Local file: `datasets/sklearn_wine/sklearn_wine.csv`

Download instructions:
```python
from sklearn.datasets import load_wine
import pandas as pd

data = load_wine(as_frame=True)
df = data.frame
df.to_csv("datasets/sklearn_wine/sklearn_wine.csv", index=False)
```

Loading:
```python
import pandas as pd
df = pd.read_csv("datasets/sklearn_wine/sklearn_wine.csv")
```

Notes:
- Best direct match to the prompt background.
- Very small, so repeated-seed evaluation is important.

### Dataset 2: sklearn Breast Cancer
- Source: `sklearn.datasets.load_breast_cancer`
- Size: 569 samples, 30 features
- Format: CSV
- Task: binary classification
- Splits: no official split; use stratified repeated hold-out or cross-validation
- License: distributed with scikit-learn
- Local file: `datasets/sklearn_breast_cancer/sklearn_breast_cancer.csv`

Download instructions:
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer(as_frame=True)
df = data.frame
df.to_csv("datasets/sklearn_breast_cancer/sklearn_breast_cancer.csv", index=False)
```

### Dataset 3: UCI Glass Identification
- Source: UCI ML Repository via `ucimlrepo` (dataset id 42)
- Size: 214 samples, 9 features
- Format: CSV
- Task: multiclass classification
- Local file: `datasets/uci_glass_identification/uci_glass_identification.csv`

Download instructions:
```python
from ucimlrepo import fetch_ucirepo
import pandas as pd

glass = fetch_ucirepo(id=42)
df = pd.concat([glass.data.features, glass.data.targets], axis=1)
df.to_csv("datasets/uci_glass_identification/uci_glass_identification.csv", index=False)
```

### Dataset 4: UCI Sonar
- Source: UCI ML Repository via `ucimlrepo` (dataset id 151)
- Size: 208 samples, 60 features
- Format: CSV
- Task: binary classification
- Local file: `datasets/uci_connectionist_bench_sonar/uci_connectionist_bench_sonar.csv`

Download instructions:
```python
from ucimlrepo import fetch_ucirepo
import pandas as pd

sonar = fetch_ucirepo(id=151)
df = pd.concat([sonar.data.features, sonar.data.targets], axis=1)
df.to_csv("datasets/uci_connectionist_bench_sonar/uci_connectionist_bench_sonar.csv", index=False)
```

## Secondary / Optional Datasets

### Dataset 5: sklearn Iris
- Source: `sklearn.datasets.load_iris`
- Size: 150 samples, 4 features
- Format: CSV
- Task: multiclass classification
- Local file: `datasets/sklearn_iris/sklearn_iris.csv`

Reason to keep:
- Useful sanity-check benchmark, but often too easy to separate subtle regularization effects.

### Dataset 6: UCI Wine Quality
- Source: UCI ML Repository direct CSV
- Size: 6,497 rows in the fetched file
- Format: CSV
- Task: regression or ordinal classification
- Local file: `datasets/uci_wine_quality/uci_wine_quality.csv`

Download instructions:
```bash
curl -L https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv \
  -o datasets/uci_wine_quality/uci_wine_quality.csv
```

Loading:
```python
import pandas as pd
df = pd.read_csv("datasets/uci_wine_quality/uci_wine_quality.csv")
```

Notes:
- This fetched artifact is larger than the target small-data regime and should be treated as an optional reference dataset.
- If the experiment runner must stay under 1,000 samples, use a stratified subset and document the sampling seed.

## Sample Data

Each dataset directory contains `samples.json` with the first 10 rows for quick inspection.

## Quick Validation Notes

- All CSVs load successfully with pandas.
- The four strongest small-data candidates for the stated hypothesis are `sklearn_wine`, `sklearn_breast_cancer`, `uci_glass_identification`, and `uci_connectionist_bench_sonar`.
- For this project, classification metrics will be easier to compare consistently than regression metrics.
