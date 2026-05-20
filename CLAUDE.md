# Titanic Survival Prediction

A data science project to predict passenger survival on the Titanic.

## Project goal

Binary classification: predict `Survived` (0 or 1) for each passenger in `test.csv` using features from `train.csv`.

## Data files

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 891 | Labeled data — includes `Survived` target column |
| `test.csv` | 418 | Unlabeled data — no `Survived` column; predictions go here |
| `gender_submission.csv` | 418 | Baseline submission (everyone by sex) — format reference |

## Column schema

| Column | Type | Notes |
|--------|------|-------|
| `PassengerId` | int | Row identifier — not a feature |
| `Survived` | int (0/1) | **Target** — only in train.csv |
| `Pclass` | int (1/2/3) | Ticket class — proxy for socioeconomic status |
| `Name` | string | Includes title (Mr., Mrs., Miss., Master., …) — extractable |
| `Sex` | string | male / female |
| `Age` | float | ~20% missing in train, ~10% in test |
| `SibSp` | int | # siblings / spouses aboard |
| `Parch` | int | # parents / children aboard |
| `Ticket` | string | Ticket number — high cardinality, partly useful as prefix |
| `Fare` | float | 1 missing value in test |
| `Cabin` | string | ~77% missing; deck letter (A–G, T) is extractable |
| `Embarked` | string | C = Cherbourg, Q = Queenstown, S = Southampton; 2 missing in train |

## Planned workflow

1. **Exploratory Data Analysis (EDA)** — distributions, missingness, survival rates by group
2. **Feature engineering** — title extraction from Name, family size, deck from Cabin, fare bins, etc.
3. **Preprocessing** — imputation, encoding, scaling inside a sklearn Pipeline
4. **Modeling** — try several classifiers (LogisticRegression, RandomForest, GradientBoosting, XGBoost/LightGBM), tune with cross-validation
5. **Evaluation** — CV accuracy / AUC; compare against gender-baseline
6. **Submission** — generate `submission.csv` in the same format as `gender_submission.csv`

## Conventions

- All code in Python; use Jupyter notebooks or Marimo for exploration, plain `.py` scripts for reusable pipeline code.
- Keep raw data files untouched; write any derived datasets to a `data/` subdirectory.
- Final predictions file: `submission.csv` with columns `PassengerId`, `Survived`.
