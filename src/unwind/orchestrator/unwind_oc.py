from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, task


@task(
    retries=3,
    retry_delay_seconds=60,
    task_run_name="Kedro: {pipeline_name} ({node_names})",
)
def run_kedro_step(pipeline_name: str, node_names: list[str] | None = None):
    project_path = Path.cwd()
    bootstrap_project(project_path)

    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name=pipeline_name, node_names=node_names)

    return f"Completed {pipeline_name} / {node_names}"


@flow(name="Unwind Time Series", log_prints=True)
def unwind_orchestrator():

    run_kedro_step(pipeline_name="baseline_models", node_names=["preprocess_node"])

    run_kedro_step(
        pipeline_name="baseline_models", node_names=["train_and_forecast_node"]
    )

    baseline_v0_evaluate = run_kedro_step(
        pipeline_name="baseline_models",
        node_names=["evaluate_baseline_node", "baseline_v0_evaluation_plot"],
    )

    return baseline_v0_evaluate


if __name__ == "__main__":
    unwind_orchestrator()
