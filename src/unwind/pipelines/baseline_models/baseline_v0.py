import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsforecast import StatsForecast
    from statsforecast.models import (
        Naive,
        HistoricAverage,
        WindowAverage,
        SeasonalNaive,
    )
    from utilsforecast.plotting import plot_series
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Baseline V0
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.function
def preprocess_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw sales data for forecasting."""
    # Filter for items with enough history and drop unneeded columns
    processed = (
        data.groupby("unique_id")
        .filter(lambda x: len(x) >= 28)
        .drop(columns=["unit_price"])
    )
    processed["ds"] = pd.to_datetime(processed["ds"])
    return processed


@app.function
def get_baseline_models():
    """Returns a list of baseline models."""
    return [
        Naive(),
        HistoricAverage(),
        WindowAverage(window_size=7),
        SeasonalNaive(season_length=7),
    ]


@app.function
def train_and_forecast(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Trains baseline models and generates forecasts."""
    models = get_baseline_models()
    sf = StatsForecast(models, freq="D")
    sf.fit(data)
    return sf.predict(h=horizon)


@app.function
def evaluate_baseline_models(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Evaluates baseline models using a simple validation split."""
    # Split into train and validation
    ds_val = data.groupby("unique_id").tail(horizon)
    ds_train_n = data.drop(ds_val.index).reset_index(drop=True)

    models = get_baseline_models()
    sf = StatsForecast(models, freq="D")
    sf.fit(ds_train_n)

    tt_preds = sf.predict(h=horizon)
    ds_eval = pd.merge(ds_val, tt_preds, "left", ["ds", "unique_id"])

    evaluation = evaluate(ds_eval, metrics=[mae])
    # Average metrics across all unique_ids
    final_evaluation = (
        evaluation.drop(columns=["unique_id"])
        .groupby("metric")
        .mean()
        .reset_index()
    )
    return final_evaluation


@app.cell
def _():
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project
    from pathlib import Path

    # Bootstrap Kedro for interactive use
    PROJECT_ROOT = "unwind"
    CWD = Path.cwd()
    try:
        idx = CWD.parts.index(PROJECT_ROOT)
        PROJECT_PATH = Path(*CWD.parts[: idx + 1])
    except ValueError:
        PROJECT_PATH = CWD

    bootstrap_project(PROJECT_PATH)

    with KedroSession.create(PROJECT_PATH, False) as session:
        context = session.load_context()
        catalog = context.catalog
        raw_data = catalog.load("raw_ingestion.ds_daily_sales")
    return (raw_data,)


@app.cell
def _(raw_data):
    ds_train_itm = preprocess_sales_data(raw_data)
    ds_train_itm
    return (ds_train_itm,)


@app.cell
def _(ds_train_itm):
    plot_series(
        ds_train_itm,
        ids=["BAGUETTE", "CROISSANT"],
        max_insample_length=56,
        palette="viridis",
    )
    return


@app.cell
def _(ds_train_itm):
    HORIZON = 7
    preds = train_and_forecast(ds_train_itm, HORIZON)
    preds
    return HORIZON, preds


@app.cell
def _(ds_train_itm, preds):
    plot_series(
        ds_train_itm,
        forecasts_df=preds,
        ids=["BAGUETTE", "CROISSANT"],
        max_insample_length=56,
        palette="viridis",
    )
    return


@app.cell
def _(HORIZON, ds_train_itm):
    evaluation_results = evaluate_baseline_models(ds_train_itm, HORIZON)
    evaluation_results
    return (evaluation_results,)


@app.function
def plot_baseline_v0_evaluation(evaluation_results: pd.DataFrame):
    methods = evaluation_results.columns[1:].tolist()
    values = evaluation_results.iloc[0, 1:].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values, color="skyblue", edgecolor="navy")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) * 0.01),  # Dynamic offset
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel("Methods")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("Models Performance")

    fig.tight_layout()

    return fig


@app.cell
def _(evaluation_results, mo):
    mo.ui.matplotlib(plot_baseline_v0_evaluation(evaluation_results).gca())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
