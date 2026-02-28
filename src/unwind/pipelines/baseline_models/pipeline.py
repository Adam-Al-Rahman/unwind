from kedro.pipeline import Node, Pipeline

from unwind.pipelines.baseline_models.baseline_v0 import (
    evaluate_baseline_models,
    plot_baseline_v0_evaluation,
    preprocess_sales_data,
    train_and_forecast,
)

BASELINE_V0 = Pipeline(
    [
        Node(
            func=preprocess_sales_data,
            inputs="raw_ingestion.ds_daily_sales",
            outputs="preprocessed_daily_sales",
            name="preprocess_node",
        ),
        Node(
            func=train_and_forecast,
            inputs=["preprocessed_daily_sales", "params:horizon"],
            outputs="baseline_forecasts",
            name="train_and_forecast_node",
        ),
        Node(
            func=evaluate_baseline_models,
            inputs=["preprocessed_daily_sales", "params:horizon"],
            outputs="baseline_evaluation",
            name="evaluate_baseline_node",
        ),
        Node(
            func=plot_baseline_v0_evaluation,
            inputs=["baseline_evaluation"],
            outputs="baseline_models.model_v0_evaluation_result_plot",
            name="baseline_v0_evaluation_plot",
        ),
    ]
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(nodes=[BASELINE_V0])
