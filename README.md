# unwind

`unwind` is a data science project focused on high-performance **time series forecasting** for daily sales data. It leverages a modern stack combining **Kedro** for pipeline engineering, **Prefect** for orchestration, and **StatsForecast** for modeling.


## Getting Started

### 1. Installation
Ensure you have a Python environment (3.12+ recommended) and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running Pipelines
You can run the forecasting pipeline in two ways:

**Via Kedro (Standard):**
```bash
kedro run --pipeline=baseline_models
```

**Via Prefect (Orchestrated):**
This provides retries, dynamic logging, and task-level tracking.
```bash
python src/unwind/orchestrator/unwind_oc.py
```

### 3. Interactive Development

Logic is developed directly in **Marimo** notebooks, which serve as the source of truth for Kedro nodes:
```bash
marimo edit src/unwind/pipelines/baseline_models/baseline_v0.py
```

### 4. Visualization & Tracking
- **View Pipeline**: `kedro viz`
- **View Experiments**: `mlflow ui`
- **Prefect UI**: `prefect server start` (then run the orchestrator)

## Features
- **Persisted Intermediates**: Uses Parquet for efficient data sharing between orchestrated tasks.
- **Baseline Models**: Includes Naive, Historic Average, and Seasonal Naive models.
- **Automated Reporting**: Generates evaluation plots and metrics tracked in MLflow.
