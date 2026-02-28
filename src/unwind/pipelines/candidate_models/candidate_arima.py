import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full", auto_download=["ipynb"])

with app.setup:
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Candidate: ARIMA

    AutoARIMA: optimized parameters (p,i, q, P,Q)
    """)
    return


@app.cell
def _():
    import marimo as mo  # noqa: PLC0415

    return (mo,)


@app.function
def get_arima_models():
    """Returns a list of candidate ARIMA models."""
    return [
        AutoARIMA(seasonal=False, alias="ARIMA"),
        AutoARIMA(season_length=7, alias="SARIMA"),
    ]


@app.function
def train_and_forecast_arima(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Trains ARIMA models and generates forecasts."""
    models = get_arima_models()
    sf = StatsForecast(models, freq="D")
    sf.fit(data)
    return sf.predict(h=horizon)


@app.function
def evaluate_arima_models(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Evaluates ARIMA models using a simple validation split."""
    # Split into train and validation
    ds_val = data.groupby("unique_id").tail(horizon)
    ds_train_n = data.drop(ds_val.index).reset_index(drop=True)

    models = get_arima_models()
    sf = StatsForecast(models, freq="D")
    sf.fit(ds_train_n)

    tt_preds = sf.predict(h=horizon)
    ds_eval = pd.merge(ds_val, tt_preds, "left", ["ds", "unique_id"])

    evaluation = evaluate(ds_eval, metrics=[mae])
    # Average metrics across all unique_ids
    final_evaluation = (
        evaluation.drop(columns=["unique_id"]).groupby("metric").mean().reset_index()
    )
    return final_evaluation


@app.cell
def _():
    from pathlib import Path  # noqa: PLC0415

    from kedro.framework.session import KedroSession  # noqa: PLC0415
    from kedro.framework.startup import bootstrap_project  # noqa: PLC0415

    from unwind.pipelines.baseline_models.baseline_v0 import (  # noqa: PLC0415
        plot_baseline_v0_evaluation,
        preprocess_sales_data,
    )

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
        ds_raw = catalog.load("raw_ingestion.ds_daily_sales")

    ds_train_itm = preprocess_sales_data(ds_raw)
    return (ds_train_itm, plot_baseline_v0_evaluation)


@app.cell
def _(ds_train_itm):
    HORIZON = 7
    arima_forecasts = train_and_forecast_arima(ds_train_itm, HORIZON)
    arima_forecasts
    return arima_forecasts, HORIZON


@app.cell
def _(HORIZON, ds_train_itm):
    evaluation_results = evaluate_arima_models(ds_train_itm, HORIZON)
    evaluation_results
    return (evaluation_results,)


@app.cell
def _(evaluation_results, mo, plot_baseline_v0_evaluation):
    mo.ui.matplotlib(plot_baseline_v0_evaluation(evaluation_results).gca())
    return


if __name__ == "__main__":
    app.run()
