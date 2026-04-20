# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A software license usage forecasting system. It trains Prophet and SARIMA (ARIMA) models on synthetic license usage data for three products (Jira, Slack, Zoom), registers them in MLflow, and serves forecasts via a FastAPI backend with a React/Vite frontend.

## Running the Full Stack

```bash
# Start everything (MLflow, trainer, backend, frontend) in order
docker-compose up --build
```

Services: MLflow UI at `:5000`, backend API at `:8000`, frontend at `:3000`.

## Backend Development

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the API locally (requires MLflow at localhost:5000 or .env override)
uvicorn app.main:app --reload

# Run all tests
pytest

# Run a single test file
pytest tests/test_forecaster_service.py -v

# Run tests with coverage
pytest --cov=app

# Lint / format
ruff check .
black .
```

Environment variables (override via `.env` or shell):
- `MLFLOW_TRACKING_URI` — default `http://localhost:5000`
- `MODEL_CACHE_DIR` — default `./models`
- `DATA_DIR` — default `./data`
- `CORS_ORIGINS` — default includes `localhost:3000` and `localhost:5173`

## ML Pipeline

```bash
cd backend

# Generate synthetic training data → backend/data/raw/license_usage.csv
python -m ml.data_generator

# Train Prophet models and register to MLflow
python -m ml.train_prophet

# Train ARIMA models and register to MLflow
python -m ml.train_arima

# Evaluate models
python -m ml.evaluate
```

DVC stages mirror these steps (see `dvc.yaml`). Model hyperparameters are in `backend/ml/params.yaml`.

## Frontend Development

```bash
cd frontend
npm install
npm run dev      # Vite dev server at :5173
npm run build
npm run lint
```

The API base URL is set at build time via `VITE_API_BASE_URL` (default: `http://localhost:8000`).

## Architecture

### Request Flow

1. Frontend (`useForecast` hook) calls `GET /forecast/{product}?horizon_months=N`
2. `forecast.py` router loads the product model via `ModelLoader`, reads historical CSV, calls `generate_forecast()`
3. `forecaster.py` service runs Prophet prediction and returns structured historical + forecast points
4. `recommender.py` derives a license recommendation (`maintain` / `reduce` / `increase`) comparing `predicted_at_renewal` and `yhat_upper` against `PRODUCT_LICENSE_COUNTS`
5. Response rendered as a Recharts line chart with summary cards

### Model Loading (`app/models/loader.py`)

`ModelLoader` is a singleton (initialized at FastAPI lifespan startup). It tries MLflow registry first (stage `Production`, else latest version), then falls back to local `.pkl` files in `MODEL_CACHE_DIR`. Products are always `["Jira", "Slack", "Zoom"]`.

### Key Constraints

- `horizon_months` must be `3`, `6`, or `12` (enforced in router and UI).
- The `/forecast/{product}` GET endpoint only accepts `Jira`, `Slack`, or `Zoom`.
- Historical data is read from `data/raw/license_usage.csv` on every request — it is not cached.
- ARIMA models are trained separately from Prophet but the forecasting service (`forecaster.py`) currently uses Prophet only; ARIMA metrics are tracked in MLflow for comparison.
