from kedro.pipeline import Node, Pipeline

from unwind.pipelines.baseline_models.baseline_v0 import (
    plot_baseline_v0_evaluation,
    preprocess_sales_data,
)

from .candidate_arima import (
    evaluate_arima_models,
    train_and_forecast_arima,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=preprocess_sales_data,
                inputs="raw_ingestion.ds_daily_sales",
                outputs="preprocessed.preprocessed_daily_sales",
                name="preprocess_node_candidate",
            ),
            Node(
                func=train_and_forecast_arima,
                inputs=["preprocessed.preprocessed_daily_sales", "params:horizon"],
                outputs="candidate_models.arima_forecasts",
                name="train_and_forecast_arima_node",
            ),
            Node(
                func=evaluate_arima_models,
                inputs=["preprocessed.preprocessed_daily_sales", "params:horizon"],
                outputs="candidate_models.arima_evaluation",
                name="evaluate_arima_node",
            ),
            Node(
                func=plot_baseline_v0_evaluation,
                inputs=["candidate_models.arima_evaluation"],
                outputs="candidate_models.arima_evaluation_plot",
                name="arima_evaluation_plot_node",
            ),
        ]
    )
