# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars",
#     "pyarrow",
#     "altair",
#     "scikit-learn",
#     "numpy",
#     "pandas",
#     "tabicl",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    import altair as alt

    from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
    )
    from tabicl import TabICLClassifier

    return (
        ColumnTransformer,
        GradientBoostingClassifier,
        GridSearchCV,
        HistGradientBoostingClassifier,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        RandomForestClassifier,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        TabICLClassifier,
        alt,
        cross_val_score,
        mo,
        pd,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Titanic — Survival Model

    Goal: predict `Survived` for the 418 passengers in `test.csv` using a scikit-learn pipeline.

    **EDA-driven feature plan** (from `eda.py`):

    | Column | Decision | Reason |
    |---|---|---|
    | `PassengerId` | drop | row id, no signal |
    | `Name` | drop raw, extract **Title** | Title (Mr/Mrs/Miss/Master) folds gender × age × class |
    | `Sex` | keep | dominant: ♀ 74% vs ♂ 19% |
    | `Pclass` | keep | 1st 63%, 2nd 47%, 3rd 24% |
    | `Age` | keep, median-impute | informative for kids; ~20% missing |
    | `SibSp`, `Parch` | fold into **FamilySize** = SibSp + Parch + 1 | family of 2–4 fared best |
    | `Ticket` | drop | high cardinality, weak signal |
    | `Fare` | keep, median-impute | 1 missing in test |
    | `Cabin` | drop raw, derive **HasCabin** | 77% missing; presence ⇒ ~67% survival |
    | `Embarked` | keep, mode-impute | 2 missing; C 55%, S 34% |

    Engineered: `Title` (Mr/Mrs/Miss/Master/Rare), `FamilySize`, `IsAlone`, `HasCabin`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load data
    """)
    return


@app.cell
def _(pl):
    train_raw = pl.read_csv("train.csv", null_values=[""])
    test_raw = pl.read_csv("test.csv", null_values=[""])
    train_raw.shape, test_raw.shape
    return test_raw, train_raw


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Feature engineering
    """)
    return


@app.cell
def _(pl):
    KEEP_TITLES = ["Mr", "Mrs", "Miss", "Master"]

    def build_features(df: pl.DataFrame):
        df = df.with_columns(
            [
                pl.col("Name").str.extract(r",\s*([^.]+)\.", 1).str.strip_chars().alias("Title"),
                (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
                pl.col("Cabin").is_not_null().cast(pl.Int8).alias("HasCabin"),
            ]
        )
        df = df.with_columns(
            [
                pl.when(pl.col("Title").is_in(KEEP_TITLES))
                .then(pl.col("Title"))
                .otherwise(pl.lit("Rare"))
                .alias("Title"),
                (pl.col("FamilySize") == 1).cast(pl.Int8).alias("IsAlone"),
            ]
        )
        return df.to_pandas()

    return (build_features,)


@app.cell
def _(build_features, test_raw, train_raw):
    train_df = build_features(train_raw)
    test_df = build_features(test_raw)
    train_df[
        ["PassengerId", "Survived", "Sex", "Pclass", "Title", "FamilySize", "IsAlone", "HasCabin", "Age", "Fare"]
    ].head(8)
    return test_df, train_df


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Pipeline definition

    `ColumnTransformer` splits numeric vs categorical:

    - **Numeric** (`Age`, `Fare`, `FamilySize`): median impute → standard scale
    - **Categorical** (`Pclass`, `Sex`, `Embarked`, `Title`, `HasCabin`, `IsAlone`): mode impute → one-hot
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
):
    FEATURES_NUM = ["Age", "Fare", "FamilySize"]
    FEATURES_CAT = ["Pclass", "Sex", "Embarked", "Title", "HasCabin", "IsAlone"]
    ALL_FEATURES = FEATURES_NUM + FEATURES_CAT

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
                ),
                FEATURES_NUM,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                FEATURES_CAT,
            ),
        ]
    )
    preprocessor
    return ALL_FEATURES, FEATURES_CAT, FEATURES_NUM, preprocessor


@app.cell
def _(ALL_FEATURES, train_df):
    X = train_df[ALL_FEATURES]
    y = train_df["Survived"]
    X.shape, y.shape
    return X, y


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Model comparison (5-fold stratified CV)
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    StratifiedKFold,
    TabICLClassifier,
    X,
    cross_val_score,
    pl,
    preprocessor,
    y,
):
    candidate_models = {
        "LogReg": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=6, random_state=42, n_jobs=-1
        ),
        "GradBoost": GradientBoostingClassifier(random_state=42),
        "HistGradBoost": HistGradientBoostingClassifier(random_state=42),
        "TabICLv2": TabICLClassifier(random_state=42, n_estimators=8, verbose=False),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, clf in candidate_models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        n_jobs = 1 if name == "TabICLv2" else -1
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
        rows.append(
            {
                "model": name,
                "acc_mean": round(float(acc.mean()), 4),
                "acc_std": round(float(acc.std()), 4),
                "auc_mean": round(float(auc.mean()), 4),
                "auc_std": round(float(auc.std()), 4),
            }
        )
    cv_results = pl.DataFrame(rows).sort("acc_mean", descending=True)
    cv_results
    return cv, cv_results


@app.cell
def _(alt, cv_results):
    cv_chart = (
        alt.Chart(cv_results)
        .mark_bar()
        .encode(
            x=alt.X("acc_mean:Q", title="CV accuracy (mean)", scale=alt.Scale(domain=[0.75, 0.90])),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.Color("acc_mean:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["model", "acc_mean", "acc_std", "auc_mean", "auc_std"],
        )
        .properties(title="5-fold CV accuracy by model", width=420, height=180)
    )
    cv_chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### TabICLv2 on raw features (no manual preprocessing)

    TabICLv2 is a pretrained tabular transformer that does in-context learning — it handles
    mixed numeric/categorical input natively, so we can skip the `ColumnTransformer` and feed
    it raw columns (with `category` dtype on the categorical ones) to see how it does in its
    intended mode.
    """)
    return


@app.cell
def _(
    ALL_FEATURES,
    FEATURES_CAT,
    StratifiedKFold,
    TabICLClassifier,
    cross_val_score,
    cv_results,
    pl,
    train_df,
    y,
):
    X_raw = train_df[ALL_FEATURES].copy()
    for c in FEATURES_CAT:
        X_raw[c] = X_raw[c].astype("category")

    cv_raw = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tabicl_raw = TabICLClassifier(random_state=42, n_estimators=8, verbose=False)
    acc_raw = cross_val_score(tabicl_raw, X_raw, y, cv=cv_raw, scoring="accuracy", n_jobs=1)
    auc_raw = cross_val_score(tabicl_raw, X_raw, y, cv=cv_raw, scoring="roc_auc", n_jobs=1)

    cv_results_full = pl.concat([
        cv_results,
        pl.DataFrame([{
            "model": "TabICLv2 (raw)",
            "acc_mean": round(float(acc_raw.mean()), 4),
            "acc_std":  round(float(acc_raw.std()), 4),
            "auc_mean": round(float(auc_raw.mean()), 4),
            "auc_std":  round(float(auc_raw.std()), 4),
        }]),
    ]).sort("acc_mean", descending=True)
    cv_results_full
    return (cv_results_full,)


@app.cell
def _(alt, cv_results_full):
    cv_chart_full = (
        alt.Chart(cv_results_full)
        .mark_bar()
        .encode(
            x=alt.X("acc_mean:Q", title="CV accuracy (mean)", scale=alt.Scale(domain=[0.75, 0.90])),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.Color("acc_mean:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["model", "acc_mean", "acc_std", "auc_mean", "auc_std"],
        )
        .properties(title="5-fold CV accuracy — full leaderboard", width=420, height=210)
    )
    cv_chart_full
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Hyperparameter tuning — GradientBoosting

    GradientBoosting led the classical-model comparison, so we tune it with a small grid
    search. (TabICLv2 is competitive without any tuning, but ~30× slower to predict.)
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    GridSearchCV,
    Pipeline,
    X,
    cv,
    preprocessor,
    y,
):
    gb_pipe = Pipeline(
        [("pre", preprocessor), ("clf", GradientBoostingClassifier(random_state=42))]
    )
    gb_grid = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": [0.05, 0.1],
    }
    gb_search = GridSearchCV(gb_pipe, gb_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gb_search.fit(X, y)
    {
        "best_params": gb_search.best_params_,
        "best_cv_acc": round(float(gb_search.best_score_), 4),
    }
    return (gb_search,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Train final model & inspect feature importance
    """)
    return


@app.cell
def _(gb_search):
    final_model = gb_search.best_estimator_
    final_model
    return (final_model,)


@app.cell
def _(FEATURES_CAT, FEATURES_NUM, alt, final_model, pl):
    ohe = final_model.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    cat_names = list(ohe.get_feature_names_out(FEATURES_CAT))
    feat_names = FEATURES_NUM + cat_names
    importances = final_model.named_steps["clf"].feature_importances_

    imp_df = (
        pl.DataFrame({"feature": feat_names, "importance": importances})
        .sort("importance", descending=True)
        .head(15)
    )
    imp_chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Feature importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("importance:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(title="Top-15 feature importances", width=420, height=320)
    )
    imp_chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Predict on test set & write submission
    """)
    return


@app.cell
def _(ALL_FEATURES, final_model, pd, test_df):
    X_test = test_df[ALL_FEATURES]
    test_preds = final_model.predict(X_test).astype(int)
    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": test_preds}
    )
    submission.to_csv("submission.csv", index=False)
    submission.head(10)
    return submission, test_preds


@app.cell
def _(mo, submission, test_preds):
    mo.md(f"""
    Wrote **`submission.csv`** — {len(submission)} rows, predicted survival rate **{test_preds.mean():.1%}**
    (train rate was 38.4%, so this is in the right ballpark).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    **Leaderboard (5-fold stratified CV):**

    | Model | CV Acc | CV AUC | Notes |
    |---|---|---|---|
    | **GradBoost (tuned)** | **0.8473** | — | best acc; grid-searched |
    | TabICLv2 (raw features) | 0.8440 | **0.8917** | best AUC; zero tuning |
    | GradBoost (default) | 0.8417 | 0.8784 | |
    | TabICLv2 (preprocessed) | 0.8372 | 0.8888 | one-hot before TabICL |
    | HistGradBoost | 0.8339 | 0.8819 | |
    | RandomForest | 0.8328 | 0.8729 | |
    | LogReg | 0.8283 | 0.8729 | baseline |

    **Submission model**: tuned `GradientBoostingClassifier(n_estimators=400, max_depth=2, learning_rate=0.1)`
    — narrowly beats TabICLv2 on accuracy and is ~30× faster at inference, which matters for
    iterative experimentation. TabICLv2 is the AUC champion and a strong drop-in alternative.

    **Top features**: `Sex`, `Title_Mr`, `Pclass`, `Fare`, `Age`.

    Submission file `submission.csv` is in this directory and ready to upload.
    """)
    return


if __name__ == "__main__":
    app.run()
