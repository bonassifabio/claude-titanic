import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    return alt, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Titanic — Preliminary EDA

    Exploring `train.csv` (891 labeled passengers). Goal: get a feel for distributions,
    missingness, and which features correlate with `Survived`.
    """)
    return


@app.cell
def _(pl):
    train = pl.read_csv("train.csv", null_values=[""])
    train
    return (train,)


@app.cell
def _(mo):
    mo.md("""
    ## Schema & basic stats
    """)
    return


@app.cell
def _(train):
    train.describe()
    return


@app.cell
def _(mo, pl, train):
    missing = (
        train.null_count()
        .transpose(include_header=True, header_name="column", column_names=["n_missing"])
        .with_columns((pl.col("n_missing") / train.height * 100).round(1).alias("pct_missing"))
        .sort("n_missing", descending=True)
    )
    mo.vstack([mo.md("### Missingness per column"), missing])
    return


@app.cell
def _(mo):
    mo.md("""
    **Notes on missingness:** `Cabin` is mostly missing (~77%), `Age` ~20%, `Embarked` has 2 rows missing.
    These will drive imputation choices later.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Target distribution
    """)
    return


@app.cell
def _(alt, pl, train):
    survived_counts = train.group_by("Survived").len().sort("Survived")
    survived_chart = (
        alt.Chart(survived_counts)
        .mark_bar()
        .encode(
            x=alt.X("Survived:N", title="Survived (0 = No, 1 = Yes)"),
            y=alt.Y("len:Q", title="Passengers"),
            color=alt.Color("Survived:N", legend=None),
            tooltip=["Survived", "len"],
        )
        .properties(title="Overall survival counts", width=300, height=250)
    )
    survival_rate = train.select(pl.col("Survived").mean()).item()
    survived_chart.properties(title=f"Overall survival counts (rate = {survival_rate:.1%})")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Survival by categorical features
    """)
    return


@app.cell
def _(alt, pl, train):
    def survival_by(col: str):
        agg = (
            train.group_by(col)
            .agg(pl.col("Survived").mean().alias("survival_rate"), pl.len().alias("n"))
            .sort(col)
        )
        return (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X(f"{col}:N", title=col),
                y=alt.Y("survival_rate:Q", title="Survival rate", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("survival_rate:Q", scale=alt.Scale(scheme="viridis"), legend=None),
                tooltip=[col, alt.Tooltip("survival_rate:Q", format=".1%"), "n"],
            )
            .properties(title=f"Survival rate by {col}", width=250, height=220)
        )

    sex_chart = survival_by("Sex")
    pclass_chart = survival_by("Pclass")
    embarked_chart = survival_by("Embarked")
    (sex_chart | pclass_chart | embarked_chart)
    return


@app.cell
def _(mo):
    mo.md("""
    Three of the strongest signals jump out immediately:
    - **Sex**: women survived at ~74%, men at ~19% — by far the dominant feature.
    - **Pclass**: 1st class ~63%, 3rd class ~24% — strong class effect.
    - **Embarked**: Cherbourg passengers fared notably better (often correlated with Pclass).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Age & Fare distributions by survival
    """)
    return


@app.cell
def _(alt, train):
    age_chart = (
        alt.Chart(train.drop_nulls("Age"))
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("Age:Q", bin=alt.Bin(maxbins=30), title="Age"),
            y=alt.Y("count():Q", stack=None, title="Passengers"),
            color=alt.Color("Survived:N", scale=alt.Scale(scheme="set1")),
            tooltip=["count()"],
        )
        .properties(title="Age distribution by survival", width=400, height=260)
    )
    age_chart
    return


@app.cell
def _(alt, pl, train):
    fare_clipped = train.with_columns(pl.col("Fare").clip(0, 300))
    fare_chart = (
        alt.Chart(fare_clipped)
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("Fare:Q", bin=alt.Bin(maxbins=30), title="Fare (clipped at 300)"),
            y=alt.Y("count():Q", stack=None, title="Passengers"),
            color=alt.Color("Survived:N", scale=alt.Scale(scheme="set1")),
        )
        .properties(title="Fare distribution by survival", width=400, height=260)
    )
    fare_chart
    return


@app.cell
def _(mo):
    mo.md("""
    Younger children (especially under 10) survived at higher rates, while the bulk of young-adult males
    did not. Fare distribution is right-skewed; higher fares cluster toward survival (consistent with Pclass).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Survival heatmap: Sex × Pclass
    """)
    return


@app.cell
def _(alt, pl, train):
    heatmap_df = (
        train.group_by(["Sex", "Pclass"])
        .agg(pl.col("Survived").mean().alias("survival_rate"), pl.len().alias("n"))
        .sort(["Sex", "Pclass"])
    )
    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("Pclass:O", title="Pclass"),
            y=alt.Y("Sex:N", title="Sex"),
            color=alt.Color("survival_rate:Q", scale=alt.Scale(scheme="viridis"), title="Survival rate"),
            tooltip=["Sex", "Pclass", alt.Tooltip("survival_rate:Q", format=".1%"), "n"],
        )
        .properties(title="Survival rate: Sex × Pclass", width=300, height=200)
    )
    text = (
        alt.Chart(heatmap_df)
        .mark_text(color="white", fontSize=14)
        .encode(
            x="Pclass:O",
            y="Sex:N",
            text=alt.Text("survival_rate:Q", format=".0%"),
        )
    )
    heatmap + text
    return


@app.cell
def _(mo):
    mo.md("""
    The interaction is stark: 1st-class women survived at ~97%, 3rd-class men at ~14%.
    A model that just predicts "female" already beats baseline; this 2-feature combo is even stronger.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Engineered preview: family size & title
    """)
    return


@app.cell
def _(alt, pl, train):
    family = train.with_columns(
        (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize")
    )
    family_agg = (
        family.group_by("FamilySize")
        .agg(pl.col("Survived").mean().alias("survival_rate"), pl.len().alias("n"))
        .sort("FamilySize")
    )
    family_chart = (
        alt.Chart(family_agg)
        .mark_bar()
        .encode(
            x=alt.X("FamilySize:O", title="Family size (SibSp + Parch + 1)"),
            y=alt.Y("survival_rate:Q", title="Survival rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["FamilySize", alt.Tooltip("survival_rate:Q", format=".1%"), "n"],
        )
        .properties(title="Survival by family size", width=400, height=240)
    )
    family_chart
    return


@app.cell
def _(alt, pl, train):
    titled = train.with_columns(
        pl.col("Name").str.extract(r",\s*([^.]+)\.", 1).alias("Title")
    )
    title_agg = (
        titled.group_by("Title")
        .agg(pl.col("Survived").mean().alias("survival_rate"), pl.len().alias("n"))
        .filter(pl.col("n") >= 5)
        .sort("survival_rate", descending=True)
    )
    title_chart = (
        alt.Chart(title_agg)
        .mark_bar()
        .encode(
            x=alt.X("survival_rate:Q", title="Survival rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Title:N", sort="-x", title="Title (n ≥ 5)"),
            color=alt.Color("survival_rate:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["Title", alt.Tooltip("survival_rate:Q", format=".1%"), "n"],
        )
        .properties(title="Survival by extracted title", width=400, height=240)
    )
    title_chart
    return


@app.cell
def _(mo):
    mo.md("""
    - **Family size**: solo travelers and large families (5+) fared worst; small families (2–4) did best.
    - **Title**: `Mrs.`, `Miss.`, `Master.` survive at high rates; `Mr.` is the lowest. `Master.` (young boys)
      is interesting because it captures young male survival that `Sex` alone misses.

    ## Takeaways for modeling
    - Strongest raw signals: **Sex**, **Pclass**, **Title**, **Fare/Pclass interaction**.
    - **Age** is informative but missing — impute by `Title` group rather than global median.
    - **Cabin** is too sparse to use directly; consider just the deck letter, or a `has_cabin` flag.
    - Engineer **FamilySize** and **IsAlone**.
    """)
    return


if __name__ == "__main__":
    app.run()
