# Building a Production Time Series Forecasting ML Pipeline on Windows 11
## From Zero to CI/CD — Using Only Free Tools

---

## What You Are Building

A running system where a user selects a software product (Jira, Slack, or Zoom) and a forecast horizon (3, 6, or 12 months) in a React dashboard. The dashboard calls a FastAPI backend that serves a trained Prophet model tracked in MLflow, all containerized with Docker and automatically tested and deployed via GitHub Actions.

### End-to-End Architecture

```
Data CSV
    ↓
train_prophet.py / train_arima.py
    ↓
MLflow Model Registry (Production stage)
    ↓
FastAPI — ModelLoader singleton (startup load)
    ↓
GET /forecast/{product}?horizon_months=6
    ↓
React useForecast hook → Recharts line chart
```

### Four Docker Services

| Service | Port | Role |
|---------|------|------|
| `mlflow` | 5000 | Model tracking, experiment UI, registry |
| `trainer` | — | One-shot batch job: generates data + trains models |
| `backend` | 8000 | FastAPI REST API serving forecasts |
| `frontend` | 3000 | React dashboard (Nginx in production) |

### Why Each Tool

| Tool | Reason |
|------|--------|
| Python 3.11 | Prophet requires 3.9+; 3.11 is the fastest available |
| Prophet | Handles trend + seasonality + missing dates, built-in uncertainty intervals |
| SARIMA (statsmodels) | Classical baseline; interpretable (p,d,q)(P,D,Q,m) orders |
| MLflow | Free, local-friendly, tracks params/metrics/artifacts, model registry with stage promotion |
| FastAPI | Async, Pydantic validation built-in, auto OpenAPI docs at `/docs` |
| Vite + React 18 | Fastest local dev server, no webpack complexity |
| Docker Desktop | Reproducible environment, eliminates "works on my machine" |
| GitHub Actions | 2,000 min/month free (private), unlimited for public repos |

---

## Phase 1 — Tool Installation (Windows 11, All Free)

### 1.1 Python 3.11

1. Download the Python 3.11.x Windows 64-bit installer from `python.org`
2. Run the installer — check **"Add Python to PATH"** and **"Install for all users"**
3. Open a new terminal and verify:
   ```
   python --version
   ```
   Expected: `Python 3.11.x`
4. Upgrade pip:
   ```
   python -m pip install --upgrade pip
   ```

> **Windows note:** On Windows, `python` and `python3` both point to the same binary when installed via the official installer (unlike Linux where `python` may be Python 2).

### 1.2 Node.js 20 LTS

1. Download the Node.js 20 LTS installer from `nodejs.org`
2. Run with default settings
3. Verify:
   ```
   node --version    # v20.x.x
   npm --version
   ```

### 1.3 Docker Desktop

1. Download Docker Desktop for Windows from `docker.com`
2. During install, enable the **WSL 2 backend** (better performance than Hyper-V)
3. After install, open Docker Desktop and wait for the whale icon in the system tray to show "Running"
4. Verify:
   ```
   docker --version
   docker compose version
   ```

> **WSL 2 note:** Docker Desktop on Windows runs containers in a lightweight Linux VM. This is why Linux-based Dockerfiles (using `apt-get`) work on a Windows host.

### 1.4 Git

1. Download Git for Windows from `git-scm.com`
2. During install choose: "Git Bash Here", "Use Git from Windows Command Prompt", "Use the OpenSSH" bundled SSH client
3. Verify: `git --version`
4. Configure your identity:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "your@email.com"
   ```

### 1.5 VS Code + Extensions

1. Download VS Code from `code.visualstudio.com`
2. Install these extensions (Ctrl+Shift+X, search by name):

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| Python | Microsoft | Python language support |
| Pylance | Microsoft | Fast type checking |
| Ruff | Astral Software | Fast Python linter |
| ESLint | Microsoft | JavaScript linting |
| Prettier | Prettier | Code formatting |
| Docker | Microsoft | Dockerfile syntax + container management |
| Remote - Containers | Microsoft | Dev inside Docker containers |
| GitLens | GitKraken | Git history visualization |
| Thunder Client | Thunder Client | Lightweight API tester (replaces Postman) |

### 1.6 GitHub Account and Repository

1. Create a free account at `github.com`
2. Create a new **public** repository named `license-forecast`
   - Public = unlimited GitHub Actions minutes (private = 2,000 min/month)
   - Do **not** initialize with README — you will push from local
3. Copy the SSH clone URL for later use

> **CI/CD cost note:** This pipeline takes approximately 8–12 minutes per run. On the free tier for public repos, this is unlimited.

### 1.7 Python ML Package Installation

First, understand virtual environments — they isolate your project's packages from the global Python installation:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**Complete `requirements.txt` contents:**

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
pydantic-settings==2.3.0
pandas==2.2.2
numpy==1.26.4
statsmodels==0.14.2
prophet==1.1.5
scikit-learn==1.5.0
mlflow==2.13.2
python-dotenv==1.0.1
httpx==0.27.0
pytest==8.2.2
pytest-asyncio==0.23.7
black==24.4.2
ruff==0.4.7
pytest-cov==5.0.0
```

Also install DVC for pipeline tracking:
```bash
pip install dvc
```

> **Windows + Prophet note:** If `pip install prophet` fails with a compiler error, install **Visual C++ Build Tools 2022** (free from Microsoft), then retry. Prophet bundles pre-compiled Stan binaries for most cases, so this is only needed if your platform doesn't have a pre-built wheel.

---

## Phase 2 — Project Setup

### 2.1 Directory Structure

Create this exact structure:

```
license-forecast/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   └── schemas.py
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── forecast.py
│   │   │   ├── products.py
│   │   │   └── metrics.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── forecaster.py
│   │       └── recommender.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   ├── train_prophet.py
│   │   ├── train_arima.py
│   │   ├── evaluate.py
│   │   ├── params.yaml
│   │   └── metrics/
│   ├── data/raw/
│   ├── models/
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_data_generator.py
│   │   ├── test_forecast_router.py
│   │   └── test_forecaster_service.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/client.js
│   │   ├── hooks/useForecast.js
│   │   ├── components/
│   │   │   ├── ForecastChart.jsx
│   │   │   ├── ProductSelector.jsx
│   │   │   ├── HorizonSelector.jsx
│   │   │   ├── SummaryCards.jsx
│   │   │   └── LoadingSpinner.jsx
│   │   ├── utils/formatters.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── nginx.conf
│   └── Dockerfile
├── mlflow/artifacts/
├── .github/workflows/ci-cd.yml
├── docker-compose.yml
├── dvc.yaml
├── .env
└── .gitignore
```

**Why this layout:**
- `backend/app` — the FastAPI application (served by uvicorn)
- `backend/ml` — the standalone training pipeline (run as batch jobs)
- `backend/data` — generated CSVs (never committed to git)
- `backend/models` — local pickle fallback (never committed to git)

### 2.2 Initialization Commands

```bash
mkdir license-forecast
cd license-forecast
git init
python -m venv .venv
.venv\Scripts\activate
```

### 2.3 `.gitignore`

```
.venv/
__pycache__/
*.pyc
*.pyo
.env
backend/data/
backend/models/
mlflow/
frontend/node_modules/
frontend/dist/
*.pkl
.pytest_cache/
.ruff_cache/
htmlcov/
.coverage
```

### 2.4 `.env` Configuration

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=license_forecasting
BACKEND_PORT=8000
FRONTEND_PORT=3000
MLFLOW_PORT=5000
MODEL_CACHE_DIR=./models
DATA_DIR=./data
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
LOG_LEVEL=INFO
```

> **Security:** `.env` must never be committed to git. The `.gitignore` entry above prevents this. GitHub Actions uses repository secrets for production values.

---

## Phase 3 — Data Layer

**File to create:** `backend/ml/data_generator.py`

### CSV Schema

Prophet has two mandatory column names:

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `ds` | date string | `2023-01-01` | "date stamp" — must be parseable as datetime |
| `y` | integer | `492` | the value being forecasted |
| `product` | string | `Jira` | filter key; added by us, not required by Prophet |

### Time Series Components

Your generator must simulate three components:

1. **Trend:** `base_licenses + (trend_per_month × month_index)` — simulates growth or decline. Zoom has a negative `trend_per_month` (-2.1) for return-to-office decline.

2. **Seasonality:** A 12-element list of multipliers (one per calendar month) captures annual patterns. Jira peaks in January (Q1 planning, multiplier 1.12). Index with `i % 12` so it repeats in multi-year data.

3. **Noise:** `numpy.random.default_rng(seed=42).normal(0, noise_std)` adds Gaussian noise. Fixed seed = reproducible results.

4. **Outliers:** Selected month indices get an extra multiplier — Jira's April budget spike (1.18×), Slack's December onboarding surge (1.22×), Zoom's Feb/Aug dips (0.78×).

### Product Configurations

```python
PRODUCT_CONFIGS = {
    "Jira": {
        "base_licenses": 480,
        "trend_per_month": 4.5,
        "seasonal_pattern": [1.12, 1.05, 1.03, 1.18, 0.95, 0.88,
                              0.82, 0.85, 0.98, 1.05, 1.08, 1.10],
        "noise_std": 12,
        "outlier_indices": [3],        # April (index 3)
        "outlier_magnitude": 1.18,
    },
    "Slack": {
        "base_licenses": 620,
        "trend_per_month": 1.8,
        "seasonal_pattern": [1.05, 1.02, 1.0, 0.98, 0.97, 0.95,
                              0.93, 0.96, 1.02, 1.05, 1.08, 1.22],
        "noise_std": 15,
        "outlier_indices": [11],       # December (index 11)
        "outlier_magnitude": 1.22,
    },
    "Zoom": {
        "base_licenses": 390,
        "trend_per_month": -2.1,       # Declining trend
        "seasonal_pattern": [1.08, 0.78, 1.02, 1.05, 1.0, 0.95,
                              0.90, 0.78, 1.02, 1.05, 1.08, 1.10],
        "noise_std": 10,
        "outlier_indices": [1, 7],     # Feb and Aug (indices 1, 7)
        "outlier_magnitude": 0.78,
    },
}
```

### Two Public Functions to Implement

**`generate_product_data(...)`** — returns a `pd.DataFrame` with columns `ds`, `y`, `product`:
```python
dates = pd.date_range(start=start_date, periods=periods, freq="MS")
# freq="MS" = "Month Start" — first day of each month
value = base_licenses + (trend_per_month * i) + seasonal_effect + noise
value = max(0, int(round(value)))   # clamp to zero, whole numbers
```

**`generate_all_products(output_path=None)`** — loops over `PRODUCT_CONFIGS`, concatenates DataFrames, optionally saves to CSV.

### Run and Verify

```bash
cd backend
python -m ml.data_generator
```

Expected output: `Saved 36 rows to backend/data/raw/license_usage.csv`

Open the CSV and confirm: 36 rows (12 months × 3 products), columns `ds,y,product`, no nulls, `y` values are positive integers.

---

## Phase 4 — Model Training

### 4.1 Prophet Model Training

**File to create:** `backend/ml/train_prophet.py`

#### How Prophet Works

Prophet decomposes a time series as:
```
y(t) = trend(t) + seasonality(t) + holidays(t) + error(t)
```

Key settings for monthly license data:
- `yearly_seasonality=True` — captures annual patterns (January Q1 peaks)
- `weekly_seasonality=False`, `daily_seasonality=False` — disabled (data is monthly)
- `seasonality_mode="multiplicative"` — seasonal effects scale with the trend level, appropriate when data grows/shrinks substantially
- `model.add_seasonality(name="quarterly", period=91.25, fourier_order=3)` — adds a quarterly pattern using 3 Fourier terms
- `changepoint_prior_scale` — controls trend flexibility. Low values (0.01) = rigid trend; high values (0.3) = flexible

#### Hyperparameter Grid

16 combinations per product, 48 total MLflow runs:

```python
PARAM_GRID = {
    "seasonality_mode": ["multiplicative", "additive"],
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.3],
    "seasonality_prior_scale": [1.0, 10.0],
}
```

#### MLflow Run Nesting

- One **parent run** per product: `{ProductName}_search`
- 16 **child runs** inside each parent, one per hyperparameter combination
- Use `mlflow.start_run(nested=True)` for child runs

#### Cross-Validation

Prophet's built-in `cross_validation()` uses a sliding window:
```python
from prophet.diagnostics import cross_validation, performance_metrics
cv_results = cross_validation(
    model,
    initial="240 days",   # first 8 months as initial training window
    period="30 days",     # slide window forward 1 month each fold
    horizon="90 days",    # evaluate 3 months ahead
)
```

On a 12-point dataset this produces 1–2 folds. Always add a try/except fallback to in-sample metrics when `cross_validation()` raises (too few data points).

#### What to Log to MLflow Per Run

```python
mlflow.log_param("product", product_name)
mlflow.log_param("seasonality_mode", params["seasonality_mode"])
mlflow.log_param("changepoint_prior_scale", params["changepoint_prior_scale"])
mlflow.log_param("seasonality_prior_scale", params["seasonality_prior_scale"])
mlflow.log_metric("mae", mae)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("mape", mape)
mlflow.prophet.log_model(model, "model",
    registered_model_name=f"license_forecast_{product_name.lower()}")
```

#### Best Model Promotion

After all 16 runs for a product:
1. Query all versions of `license_forecast_{product}` in the registry
2. Find the version with the lowest MAPE
3. Archive any existing Production-stage versions
4. Promote the best version to Production stage

#### Run Command

```bash
# Start MLflow server first (in a separate terminal):
mlflow server \
  --backend-store-uri sqlite:///./mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 --port 5000

# Then train (in another terminal):
set MLFLOW_TRACKING_URI=http://localhost:5000
cd backend
python -m ml.train_prophet
```

Expected runtime: 3–8 minutes (Stan compilation on first run takes longer).

### 4.2 ARIMA/SARIMA Baseline

**File to create:** `backend/ml/train_arima.py`

SARIMA(p,d,q)(P,D,Q,m) — the seasonal ARIMA:
- `order=(1,1,1)` — AR(1), one differencing pass, MA(1)
- `seasonal_order=(1,1,1,12)` — seasonal period m=12 months (annual cycle)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12))
result = model.fit(disp=False)
forecast = result.forecast(steps=len(test_series))
```

**Evaluation approach:** Train on first 10 months, test on last 2 months.

Logs to MLflow as `arima_{product}` with metrics (MAE, RMSE, MAPE, R²). ARIMA is **not registered** in the model registry — metrics only, for comparison with Prophet.

```bash
set MLFLOW_TRACKING_URI=http://localhost:5000
python -m ml.train_arima
```

### 4.3 Evaluation Metrics

**File to create:** `backend/ml/evaluate.py`

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| MAE | mean(|actual - predicted|) | Average license count error — easy to explain to stakeholders |
| RMSE | sqrt(mean((actual - predicted)²)) | Penalizes large errors more. RMSE >> MAE = occasional big misses |
| MAPE | mean(|actual - predicted| / actual) × 100 | Scale-independent %. Primary selection criterion. Exclude zeros from denominator |
| R² | 1 - SS_res/SS_tot | Variance explained. 1.0 = perfect; <0 = worse than predicting the mean |

### 4.4 params.yaml

**File to create:** `backend/ml/params.yaml`

```yaml
prophet:
  seasonality_mode: multiplicative
  changepoint_prior_scale: 0.05
  seasonality_prior_scale: 10.0
  forecast_horizon: 12

arima:
  order: [1, 1, 1]
  seasonal_order: [1, 1, 1, 12]
```

### 4.5 DVC Pipeline (Optional — Recommended)

**File to create:** `dvc.yaml` at the project root.

DVC tracks data and model dependencies. When `data_generator.py` changes, `dvc repro` re-runs all downstream stages automatically.

```yaml
stages:
  generate_data:
    cmd: python -m ml.data_generator
    deps: [ml/data_generator.py]
    outs: [data/raw/license_usage.csv]

  train_prophet:
    cmd: python -m ml.train_prophet
    deps: [ml/train_prophet.py, data/raw/license_usage.csv, ml/params.yaml]
    params: [ml/params.yaml:prophet]
    metrics: [ml/metrics/prophet_metrics.json]

  train_arima:
    cmd: python -m ml.train_arima
    deps: [ml/train_arima.py, data/raw/license_usage.csv, ml/params.yaml]
    params: [ml/params.yaml:arima]
    metrics: [ml/metrics/arima_metrics.json]
```

```bash
pip install dvc
dvc init
dvc repro   # Runs all three stages in dependency order
```

---

## Phase 5 — Model Registry and Versioning

**File to create:** `backend/app/models/loader.py`

### How MLflow Model Registry Works

Four stages exist per registered model version:

| Stage | Meaning |
|-------|---------|
| None | Newly registered, not yet evaluated |
| Staging | Under review / testing |
| Production | Active — served by the API |
| Archived | Superseded by a newer version |

The API queries for the `Production`-stage version. View the registry at `http://localhost:5000` → "Models" tab.

### Querying the Registry Programmatically

```python
from mlflow import MlflowClient
import mlflow.prophet

client = MlflowClient()

# Get Production version
versions = client.get_latest_versions(
    "license_forecast_jira",
    stages=["Production"]
)

# Load the model
model = mlflow.prophet.load_model(
    f"models:/license_forecast_jira/{versions[0].version}"
)
```

### Local Pickle Fallback

When MLflow is unreachable (network error, service not started) or no Production version exists yet, `ModelLoader` falls back to loading `{product_name.lower()}.pkl` from `MODEL_CACHE_DIR` (default: `./models`).

```python
with open("backend/models/jira.pkl", "wb") as f:
    pickle.dump(trained_prophet_model, f)
```

> **Security:** Never load pickle files from untrusted sources — they execute arbitrary code on load.

### ModelLoader Singleton Pattern

```python
_loader_instance = None   # module-level singleton

def init_loader(tracking_uri, cache_dir):
    global _loader_instance
    _loader_instance = ModelLoader(tracking_uri, cache_dir)
    return _loader_instance

def get_loader():
    if _loader_instance is None:
        raise RuntimeError("ModelLoader not initialized — call init_loader() first")
    return _loader_instance
```

FastAPI uses `Depends(get_loader)` to inject the singleton into every request handler without global state.

---

## Phase 6 — API Layer (FastAPI)

### File Responsibilities

| File | Role |
|------|------|
| `app/config.py` | Pydantic Settings — reads from `.env` and environment |
| `app/models/schemas.py` | Request/response Pydantic validation models |
| `app/models/loader.py` | ModelLoader singleton (MLflow + pickle fallback) |
| `app/services/forecaster.py` | Prophet forecast generation — pure function |
| `app/services/recommender.py` | Business logic: buy_more / reduce / hold |
| `app/routers/forecast.py` | `GET /forecast/{product}` and `POST /forecast/` |
| `app/routers/products.py` | `GET /products/` — list available products |
| `app/routers/metrics.py` | `GET /model-metrics/` — expose MLflow metric summaries |
| `app/main.py` | App factory, CORS middleware, lifespan startup hook |

### 6.1 Configuration — `app/config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "license_forecasting"
    model_cache_dir: str = "./models"
    data_dir: str = "./data"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

`pydantic_settings.BaseSettings` reads from environment variables first, then `.env` file, then falls back to defaults. The `cors_origins` field parses the comma-separated env var `"http://localhost:3000,http://localhost:5173"` automatically into a list.

### 6.2 Pydantic Schemas — `app/models/schemas.py`

```python
from typing import Literal
from pydantic import BaseModel
from datetime import date

class ForecastRequest(BaseModel):
    product: Literal["Jira", "Slack", "Zoom"]
    horizon_months: Literal[3, 6, 12] = 6
    renewal_date: date | None = None

class HistoricalPoint(BaseModel):
    ds: date
    y: float

class ForecastPoint(BaseModel):
    ds: date
    yhat: float
    yhat_lower: float   # 80th-percentile lower bound (from Prophet)
    yhat_upper: float   # 80th-percentile upper bound

class ForecastResponse(BaseModel):
    product: str
    horizon_months: int
    historical: list[HistoricalPoint]
    forecast: list[ForecastPoint]
    trend_direction: Literal["increasing", "decreasing", "stable"]
    recommendation: Literal["buy_more", "reduce", "hold"]
    recommendation_detail: str
    current_usage: float
    predicted_at_renewal: float
    percent_change: float
    renewal_date: date
    model_version: str
    generated_at: str   # ISO 8601 timestamp
```

Using `Literal["Jira", "Slack", "Zoom"]` instead of `str` means:
- Pydantic rejects any other value with an HTTP 422
- FastAPI documents the allowed values in the OpenAPI spec automatically

### 6.3 Application Lifespan — `app/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs at startup — load all three models into memory
    loader = init_loader(
        tracking_uri=settings.mlflow_tracking_uri,
        cache_dir=settings.model_cache_dir
    )
    loader.reload_all()   # Loads Jira, Slack, Zoom from MLflow registry
    yield
    # Shutdown logic (if any) goes here

app = FastAPI(title="License Forecast API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Loading models at startup prevents first-request latency (Prophet models are ~50MB and take 1–3 seconds to deserialize).

### 6.4 Forecast Service — `app/services/forecaster.py`

```python
def generate_forecast(model, historical_df, horizon_months, renewal_date=None):
    # 1. Extend date range horizon_months into the future
    future = model.make_future_dataframe(periods=horizon_months, freq="MS")

    # 2. Run Prophet prediction — returns yhat, yhat_lower, yhat_upper
    forecast_df = model.predict(future)

    # 3. Extract historical points (from original CSV, not Prophet in-sample)
    historical = extract_historical(historical_df, forecast_df)

    # 4. Extract forecast (only the last horizon_months rows)
    forecast_points = extract_forecast_points(forecast_df, horizon_months)

    # 5. Compute trend (3% threshold: +3% = increasing, -3% = decreasing)
    trend = compute_trend_direction(forecast_points)

    # 6. Compute % change from current usage to predicted at renewal
    percent_change = compute_percent_change(current_usage, predicted_at_renewal)

    return {
        "historical": historical,
        "forecast": forecast_points,
        "trend_direction": trend,
        "current_usage": current_usage,
        "predicted_at_renewal": predicted_at_renewal,
        "percent_change": percent_change,
    }
```

### 6.5 Recommender Service — `app/services/recommender.py`

Business logic thresholds (with a 10% safety buffer to avoid under-licensing):

```python
PRODUCT_LICENSE_COUNTS = {"Jira": 550, "Slack": 700, "Zoom": 420}
BUFFER = 1.10    # 10% safety buffer on upper bound

def compute_recommendation(product, predicted_at_renewal, yhat_upper):
    current = PRODUCT_LICENSE_COUNTS[product]
    predicted_need = yhat_upper * BUFFER

    if predicted_need > current * 1.05:
        return "buy_more", f"Upper forecast ({yhat_upper:.0f} + 10% buffer) exceeds current licenses by more than 5%"
    elif predicted_at_renewal < current * 0.80:
        return "reduce", f"Predicted usage ({predicted_at_renewal:.0f}) is below 80% of current licenses"
    else:
        return "hold", f"Usage is within comfortable range of current {current} licenses"
```

### 6.6 CORS Explained

Browsers block JavaScript from calling APIs on a different origin (domain:port). The React app runs on `localhost:3000` or `localhost:5173` (Vite dev server). Without CORS headers, the API call is blocked by the browser.

The `CORSMiddleware` in `main.py` adds the required headers. Always specify exact allowed origins (never use `*` in production — it allows any website to call your API).

### 6.7 Local API Test

```bash
cd backend
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI. Or test with curl:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/forecast/Jira?horizon_months=6"
curl "http://localhost:8000/forecast/Slack?horizon_months=12"
curl "http://localhost:8000/products/"
```

Expected health response:
```json
{
  "status": "healthy",
  "models_loaded": {"Jira": true, "Slack": true, "Zoom": true},
  "mlflow_connected": true
}
```

---

## Phase 7 — Frontend (React)

### 7.1 Scaffold with Vite

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
```

The `-- --template react` syntax: the double `--` separates npm's own arguments from arguments passed to the `create-vite` scaffolding tool.

### 7.2 Install Dependencies

```bash
npm install axios recharts dayjs
npm install -D tailwindcss postcss autoprefixer eslint eslint-plugin-react eslint-plugin-react-hooks
npx tailwindcss init -p
```

### 7.3 Tailwind CSS Configuration

**`frontend/tailwind.config.js`:**
```javascript
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

The `content` array tells Tailwind which files to scan for class names. Unused classes are purged in production builds.

**`frontend/src/index.css`:**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 7.4 Axios API Client — `src/api/client.js`

```javascript
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL, timeout: 30000 });

export const fetchProducts = () => api.get("/products/").then(r => r.data);

export const fetchForecast = (product, horizonMonths) =>
  api.post("/forecast/", { product, horizon_months: horizonMonths }).then(r => r.data);

export const fetchModelMetrics = () => api.get("/model-metrics/").then(r => r.data);
```

`import.meta.env.VITE_API_BASE_URL` — Vite exposes environment variables prefixed with `VITE_` to the browser bundle. Set in `frontend/.env.local` for local dev, injected via Docker ARG at build time.

### 7.5 useForecast Hook — `src/hooks/useForecast.js`

```javascript
import { useState, useEffect, useCallback } from "react";
import { fetchProducts, fetchForecast } from "../api/client";

export function useForecast() {
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("Jira");
  const [horizon, setHorizon] = useState(6);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load product list once on mount
  useEffect(() => {
    fetchProducts().then(setProducts);
  }, []);

  // useCallback prevents infinite re-render loop in the effect below
  const loadForecast = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchForecast(selectedProduct, horizon);
      setForecastData(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, [selectedProduct, horizon]);

  // Re-fetch whenever product or horizon changes
  useEffect(() => { loadForecast(); }, [loadForecast]);

  return { products, selectedProduct, setSelectedProduct,
           horizon, setHorizon, forecastData, loading, error,
           refresh: loadForecast };
}
```

**Why `useCallback` is critical here:** Without it, `loadForecast` would be a new function reference on every render, causing the `useEffect` to run on every render (because `loadForecast` is in its dependency array), creating an infinite re-fetch loop.

### 7.6 ForecastChart Component — `src/components/ForecastChart.jsx`

The chart uses Recharts `ComposedChart` to mix Line and Area on the same plot.

**Data merging trick:** Merge historical and forecast into one array with `undefined` for inapplicable values:

```javascript
function buildChartData(historical, forecast) {
  const hist = historical.map(p => ({ date: p.ds, actual: p.y }));
  const fcast = forecast.map(p => ({
    date: p.ds,
    forecast: p.yhat,
    ciLow: p.yhat_lower,
    ciHigh: p.yhat_upper,
  }));
  return [...hist, ...fcast];  // actual is undefined on forecast rows; forecast is undefined on historical rows
}
```

Recharts skips `undefined` values (does not draw a dot or connect the line through them) when `connectNulls={false}`.

**Confidence interval band:** Stack two `Area` components — `ciLow` with `fill="transparent"` (invisible base) and `ciHigh` with a light orange fill. This produces the shaded confidence band.

**Component composition:**
```jsx
<ComposedChart data={chartData}>
  <Line dataKey="actual" stroke="#3b82f6" dot={true} connectNulls={false} />
  <Line dataKey="forecast" stroke="#f97316" strokeDasharray="6 3" dot={false} connectNulls={false} />
  <Area dataKey="ciLow" fill="transparent" stroke="none" />
  <Area dataKey="ciHigh" fill="#fed7aa" stroke="none" fillOpacity={0.4} />
  <ReferenceLine x={renewal_date} stroke="#ef4444" strokeDasharray="4 4" label="Renewal" />
  <ResponsiveContainer width="100%" height={350} />
</ComposedChart>
```

### 7.7 SummaryCards Component — `src/components/SummaryCards.jsx`

Four cards displayed in a responsive grid:

| Card | Data | Color Logic |
|------|------|-------------|
| Current Usage | `current_usage` | Neutral |
| Predicted at Renewal | `predicted_at_renewal + renewal_date` | Neutral |
| Projected Change | `percent_change` with ↑↓→ icons | Green if positive, red if negative |
| Recommendation | `recommendation + recommendation_detail` | Green=hold, Yellow=buy_more, Red=reduce |

### 7.8 Local Frontend Test

Create `frontend/.env.local`:
```
VITE_API_BASE_URL=http://localhost:8000
```

Then:
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. Click each product button, change the horizon selector, and verify the chart updates without page reload.

---

## Phase 8 — Containerization (Docker)

### 8.1 Backend Dockerfile — `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim

# System deps for Prophet (compiler) and health check (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements FIRST — maximizes Docker layer cache
# When only app code changes, this pip install layer is skipped
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/raw data/processed models mlflow/artifacts

EXPOSE 8000

# --host 0.0.0.0: required — without it uvicorn binds to localhost inside
# the container and is unreachable from outside
# --workers 1: 3 models × 50MB × N workers = OOM on a laptop; keep at 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**System packages explained:**
- `gcc`, `g++` — Prophet's cmdstanpy may need to compile Stan models on first fit
- `libgomp1` — Prophet uses OpenMP for parallelism
- `curl` — used by Docker health check (`curl -f http://localhost:8000/health`)
- `--no-install-recommends` + `rm -rf /var/lib/apt/lists/*` — minimize image size

### 8.2 Frontend Dockerfile — `frontend/Dockerfile` (Multi-Stage)

Multi-stage builds discard the Node.js build stage after copying the compiled output. Final image = Nginx + static files (~20MB vs ~400MB with Node).

```dockerfile
# Stage 1: Build
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci                        # npm ci: deterministic, faster than npm install in CI/Docker
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build                 # Output: /app/dist/

# Stage 2: Serve
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
# "daemon off;" keeps nginx in foreground — required for Docker to know the container is alive
```

**`npm ci` vs `npm install`:** Use `npm install` when adding packages locally. Use `npm ci` in Docker and CI — it's deterministic (reads `package-lock.json` exactly), deletes `node_modules` first, and is faster.

### 8.3 Nginx Config — `frontend/nginx.conf`

```nginx
server {
    listen 3000;
    root /usr/share/nginx/html;
    index index.html;

    # SPA routing: unknown paths fall back to index.html (React Router handles them)
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy /api/ calls to the backend container (Docker internal DNS)
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
    }
}
```

The `try_files $uri $uri/ /index.html` trick: when a user navigates directly to `/products/Jira`, Nginx looks for that file, fails, and falls back to `index.html`. React Router then handles the URL client-side.

### 8.4 docker-compose.yml

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.2
    ports: ["${MLFLOW_PORT:-5000}:5000"]
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0 --port 5000
    volumes: ["./mlflow:/mlflow"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      retries: 5

  trainer:
    build: ./backend
    depends_on:
      mlflow: { condition: service_healthy }
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    command: >
      bash -c "
        python -m ml.data_generator &&
        python -m ml.train_prophet &&
        python -m ml.train_arima
      "
    volumes:
      - ./backend/data:/app/data
      - ./backend/models:/app/models
      - ./mlflow:/app/mlflow
    restart: "no"    # One-shot batch job — exits after training completes

  backend:
    build: ./backend
    ports: ["${BACKEND_PORT:-8000}:8000"]
    depends_on:
      mlflow: { condition: service_healthy }
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MODEL_CACHE_DIR: ./models
      DATA_DIR: ./data
    volumes:
      - ./backend/data:/app/data
      - ./backend/models:/app/models
      - ./mlflow:/app/mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      retries: 5

  frontend:
    build:
      context: ./frontend
      args:
        VITE_API_BASE_URL: http://localhost:${BACKEND_PORT:-8000}
    ports: ["${FRONTEND_PORT:-3000}:3000"]
    depends_on:
      backend: { condition: service_healthy }
```

**Startup order:** `mlflow` healthy → `trainer` starts → `backend` starts (in parallel) → `backend` healthy → `frontend` starts

**Volume strategy:**
- `./backend/data:/app/data` — trainer writes CSV, backend reads it
- `./backend/models:/app/models` — shared pickle fallback cache
- `./mlflow:/app/mlflow` — shared MLflow database and artifacts

**Port syntax:** `${MLFLOW_PORT:-5000}` means "use the env var value if set, otherwise use 5000".

### 8.5 Running the Full Stack

```bash
docker compose up --build
```

**Expected startup sequence:**
1. MLflow container starts → health check begins (15–30 seconds to pass)
2. Trainer starts → runs data generation + training (3–10 minutes)
3. Backend starts → loads models from MLflow registry
4. Frontend starts after backend health check passes

**Useful commands:**
```bash
docker compose logs -f trainer      # Watch training progress
docker compose logs -f backend      # Watch API startup
docker compose down                 # Stop all services
docker compose down -v              # Stop and delete volumes
docker compose up --build backend   # Rebuild only backend
```

---

## Phase 9 — CI/CD Pipeline (GitHub Actions)

**File to create:** `.github/workflows/ci-cd.yml`

### 9.1 Job Dependency Graph

```
lint-backend ──→ test-backend ──→ build-backend-image ──→ register-model
lint-frontend ──→ build-frontend
```

Jobs without `needs:` run in parallel. Jobs with `needs:` wait for the specified jobs to succeed.

### 9.2 Workflow Trigger

```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```

Docker image push and model promotion run **only on push to main** (not on PRs):
```yaml
if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

### 9.3 lint-backend Job

```yaml
lint-backend:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: pip
    - run: pip install ruff black
    - run: ruff check backend/
    - run: black --check backend/
```

- `ruff check` — fast linter (unused imports, undefined variables, style issues)
- `black --check` — verifies formatting without modifying files; exits non-zero if any file would need reformatting

**Fix locally before pushing:**
```bash
ruff check --fix backend/
black backend/
```

### 9.4 test-backend Job

```yaml
test-backend:
  needs: [lint-backend]
  runs-on: ubuntu-latest
  services:
    mlflow:
      image: ghcr.io/mlflow/mlflow:v2.13.2
      ports: ["5000:5000"]
      options: >-
        --health-cmd "curl -f http://localhost:5000/health"
        --health-interval 10s
        --health-retries 5
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: pip
    - working-directory: backend
      run: pip install -r requirements.txt
    - working-directory: backend
      env:
        MLFLOW_TRACKING_URI: http://localhost:5000
      run: |
        python -m ml.data_generator
        pytest tests/ -v --cov=app --cov-report=xml
    - uses: codecov/codecov-action@v4
      continue-on-error: true
```

**Service containers:** GitHub Actions spins up a Docker container for MLflow alongside the test runner VM. The container is accessible at `http://localhost:5000` (the mapped port, not `http://mlflow:5000`).

### 9.5 lint-frontend and build-frontend Jobs

```yaml
lint-frontend:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with:
        node-version: "20"
        cache: npm
        cache-dependency-path: frontend/package-lock.json
    - working-directory: frontend
      run: npm ci && npm run lint

build-frontend:
  needs: [lint-frontend]
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with: { node-version: "20", cache: npm, cache-dependency-path: frontend/package-lock.json }
    - working-directory: frontend
      run: npm ci && npm run build
    - uses: actions/upload-artifact@v4
      with:
        name: frontend-dist
        path: frontend/dist/
```

`cache-dependency-path`: Points to the lockfile that determines the cache key. When `package-lock.json` changes, cache is invalidated and `npm ci` re-downloads.

### 9.6 Docker Image Build and Push

```yaml
build-backend-image:
  needs: [test-backend]
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  runs-on: ubuntu-latest
  permissions:
    contents: read
    packages: write
  steps:
    - uses: actions/checkout@v4
    - uses: docker/setup-buildx-action@v3
    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}    # Built-in, no setup needed
    - uses: docker/metadata-action@v5
      id: meta
      with:
        images: ghcr.io/${{ github.repository }}/backend
        tags: |
          type=sha,prefix=sha-
          type=raw,value=latest
    - uses: docker/build-push-action@v5
      with:
        context: ./backend
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

**ghcr.io:** GitHub Container Registry — free and integrated with GitHub. No separate account. Auth via `GITHUB_TOKEN` (automatic).

**Docker layer cache (`type=gha`):** When only application code changes (not requirements), Docker skips re-running `pip install`. This cuts build time from 5 minutes to under 1 minute on cache hits.

**Tags:** `sha-abc1234` for traceability + `latest` for convenience.

### 9.7 MLflow Model Promotion Job

```yaml
register-model:
  needs: [build-backend-image]
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.11", cache: pip }
    - working-directory: backend
      run: pip install mlflow prophet
    - working-directory: backend
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python -c "
        from ml.train_prophet import promote_best_model
        for product in ['Jira', 'Slack', 'Zoom']:
            promote_best_model(product)
        "
```

**Setting up the GitHub Secret:**
1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `MLFLOW_TRACKING_URI`
4. Value: your production MLflow server URL

> **Never hardcode the MLflow URL** in the YAML file — secrets are masked in logs; hardcoded URLs are not.

---

## Phase 10 — Verification

### 10.1 Per-Phase Verification Checklist

| Phase | Command | Expected Result |
|-------|---------|----------------|
| 1. Tools | `python --version` | `3.11.x` |
| 1. Tools | `node --version` | `v20.x.x` |
| 1. Tools | `docker --version` | `24.x.x` or later |
| 3. Data | `python -m ml.data_generator` | 36-row CSV, no nulls |
| 4. Training | Open `http://localhost:5000` → Models tab | 3 models in Production stage |
| 6. API | `curl http://localhost:8000/health` | `{"status":"healthy","models_loaded":{"Jira":true,...}}` |
| 7. Frontend | Open `http://localhost:5173` | Chart renders with confidence band |
| 8. Docker | `docker compose up --build` | All 4 services healthy |
| 9. CI/CD | `git push origin main` → Actions tab | All jobs green |

### 10.2 End-to-End Test Sequence

1. Start everything:
   ```bash
   docker compose up --build
   ```

2. Watch training complete:
   ```bash
   docker compose logs -f trainer
   ```
   Wait for: `Training complete for Jira, Slack, Zoom`

3. Verify API has models:
   ```bash
   curl http://localhost:8000/health
   ```
   All three `models_loaded` values must be `true`

4. Test each product:
   ```bash
   curl "http://localhost:8000/forecast/Jira?horizon_months=6"
   curl "http://localhost:8000/forecast/Slack?horizon_months=3"
   curl "http://localhost:8000/forecast/Zoom?horizon_months=12"
   ```
   Zoom should show `"trend_direction": "decreasing"` (declining trend configured)

5. Open the frontend at `http://localhost:3000` and verify:
   - 12 historical data points (blue solid line)
   - Forecast line (orange dashed)
   - Confidence band (shaded orange area)
   - Renewal date marker (red vertical dashed line)
   - Recommendation card with correct color
   - Changing product or horizon updates the chart without page reload

6. Trigger CI/CD:
   ```bash
   git add .
   git commit -m "feat: initial pipeline implementation"
   git remote add origin git@github.com:yourusername/license-forecast.git
   git push -u origin main
   ```
   Go to GitHub repository → Actions tab. All jobs should be green.

### 10.3 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `pip install prophet` fails | Missing C++ compiler | Install Visual C++ Build Tools 2022 (free from Microsoft), retry |
| `Connection refused` to MLflow | MLflow not started | Run `mlflow server ...` before backend; in Docker: `docker compose logs mlflow` |
| `"No model available for Jira"` | Training failed or not run | `docker compose logs trainer`; run `python -m ml.train_prophet` manually |
| CORS error in browser console | Origin mismatch | `CORS_ORIGINS` must exactly match including port (e.g., `http://localhost:3000`). Restart backend after changing `.env` |
| Docker health check fails for backend | `curl` missing or API not ready | Check `docker compose logs backend`; ensure mlflow service is healthy first |
| GitHub Actions "Package creation disabled" | ghcr.io not enabled | Profile Settings → Packages → enable "Improved container support" |
| Frontend shows blank/error | API not reachable | Check `VITE_API_BASE_URL` matches backend port; check CORS settings |

---

## Quick Command Reference

```bash
# ─── Environment Setup ───────────────────────────────────────────
python -m venv .venv
.venv\Scripts\activate

# ─── Start MLflow Locally ────────────────────────────────────────
mlflow server \
  --backend-store-uri sqlite:///./mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 --port 5000

# ─── ML Pipeline ─────────────────────────────────────────────────
cd backend
python -m ml.data_generator
set MLFLOW_TRACKING_URI=http://localhost:5000
python -m ml.train_prophet
python -m ml.train_arima
python -m ml.evaluate

# ─── API ──────────────────────────────────────────────────────────
uvicorn app.main:app --reload

# ─── Tests + Lint ─────────────────────────────────────────────────
pytest tests/ -v --cov=app
ruff check . && black --check .
ruff check --fix . && black .      # Auto-fix

# ─── Frontend ─────────────────────────────────────────────────────
cd frontend
npm run dev
npm run build
npm run lint

# ─── Docker Full Stack ────────────────────────────────────────────
docker compose up --build
docker compose logs -f trainer
docker compose logs -f backend
docker compose down
docker compose down -v              # Also delete volumes

# ─── DVC Pipeline ─────────────────────────────────────────────────
dvc init
dvc repro                          # Run all stages in dependency order

# ─── Deploy (triggers CI/CD) ──────────────────────────────────────
git add .
git commit -m "feat: your message"
git push origin main
```

---

## File Creation Order

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `.gitignore` | 2 | Excludes `.venv`, `.env`, `data/`, `models/`, `mlflow/` |
| 2 | `.env` | 2 | Local environment variables |
| 3 | `backend/requirements.txt` | 2 | Python dependencies |
| 4 | `backend/ml/params.yaml` | 4 | Hyperparameter defaults |
| 5 | `backend/ml/data_generator.py` | 3 | Synthetic CSV generation |
| 6 | `backend/ml/evaluate.py` | 4 | MAE/RMSE/MAPE/R² metrics |
| 7 | `backend/ml/train_prophet.py` | 4 | Prophet training + MLflow |
| 8 | `backend/ml/train_arima.py` | 4 | SARIMA training + MLflow |
| 9 | `backend/app/config.py` | 6 | Pydantic Settings |
| 10 | `backend/app/models/schemas.py` | 6 | Request/response Pydantic models |
| 11 | `backend/app/models/loader.py` | 5 | ModelLoader singleton |
| 12 | `backend/app/services/forecaster.py` | 6 | Prophet forecast generation |
| 13 | `backend/app/services/recommender.py` | 6 | Buy/reduce/hold logic |
| 14 | `backend/app/routers/forecast.py` | 6 | GET/POST /forecast endpoints |
| 15 | `backend/app/routers/products.py` | 6 | GET /products/ |
| 16 | `backend/app/routers/metrics.py` | 6 | GET /model-metrics/ |
| 17 | `backend/app/main.py` | 6 | FastAPI app factory, CORS, lifespan |
| 18 | `backend/Dockerfile` | 8 | Backend container image |
| 19 | `frontend/package.json` | 7 | npm dependencies |
| 20 | `frontend/tailwind.config.js` | 7 | Tailwind content paths |
| 21 | `frontend/vite.config.js` | 7 | Vite + React plugin config |
| 22 | `frontend/src/index.css` | 7 | Tailwind directives |
| 23 | `frontend/src/api/client.js` | 7 | Axios API client |
| 24 | `frontend/src/hooks/useForecast.js` | 7 | Data-fetching hook |
| 25 | `frontend/src/components/ForecastChart.jsx` | 7 | Recharts composite chart |
| 26 | `frontend/src/components/SummaryCards.jsx` | 7 | Metric summary cards |
| 27 | `frontend/src/App.jsx` | 7 | Root component |
| 28 | `frontend/nginx.conf` | 8 | Nginx SPA routing + API proxy |
| 29 | `frontend/Dockerfile` | 8 | Multi-stage frontend image |
| 30 | `docker-compose.yml` | 8 | Full stack orchestration |
| 31 | `dvc.yaml` | 4 | ML pipeline DAG |
| 32 | `.github/workflows/ci-cd.yml` | 9 | GitHub Actions pipeline |
