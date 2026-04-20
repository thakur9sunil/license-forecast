# License Forecasting System — Project Workflow & Interview Guide

---

## Databricks Migration Guide

### What is Databricks?

Databricks is a cloud-based data and AI platform built on top of Apache Spark. Think of it as a managed environment that gives you:
- Notebooks (like Jupyter but collaborative and cloud-hosted)
- Managed compute clusters (virtual machines you spin up on demand)
- Built-in MLflow (no setup needed — it just works)
- A model registry and one-click model serving
- Delta Lake (a better version of CSV/Parquet with versioning built in)

In short: everything we set up manually on your laptop — the MLflow server, the Python environment, the model storage — Databricks provides as a managed service.

---

### What Changes vs What Stays the Same

| Component | Local Setup (Current) | Databricks |
|-----------|----------------------|------------|
| MLflow server | You start it manually on port 5000 | Built-in — always running, no setup |
| Python environment | `.venv` + `pip install` | Cluster with pre-installed libraries |
| Data storage | Local CSV file | DBFS (Databricks File System) or Delta table |
| Training scripts | Run from terminal | Run as Databricks notebooks or Jobs |
| Model registry | Local MLflow registry | Unity Catalog Model Registry |
| Model serving | FastAPI on port 8000 | Databricks Model Serving (managed REST endpoint) |
| Frontend | Vite dev server on port 3000 | Same React app — just change the API URL |
| Compiler issues (Prophet/Stan) | Required a fix on Windows | Not needed — Databricks clusters run Linux with full toolchains |

---

### The Notebook Structure (in `databricks/notebooks/`)

We created 4 notebooks that mirror the local pipeline:

```
databricks/
├── notebooks/
│   ├── 01_data_generator.py   ← Generate data → save to DBFS + Delta table
│   ├── 02_train_prophet.py    ← Train Prophet + log to MLflow + register model
│   ├── 03_train_arima.py      ← Train ARIMA benchmark + log to MLflow
│   └── 04_evaluate.py         ← Compare Prophet vs ARIMA side by side
└── cluster_requirements.txt   ← Libraries to install on your cluster
```

---

### Key Differences Explained Simply

**1. No `MLFLOW_TRACKING_URI` needed**

On your laptop we had to tell Python where MLflow was running:
```python
mlflow.set_tracking_uri("http://localhost:5000")
```
On Databricks, MLflow is already integrated into the platform. Every notebook automatically logs to the workspace MLflow. You just remove that line entirely.

**2. File paths use DBFS instead of local paths**

On your laptop:
```python
DATA_PATH = "backend/data/raw/license_usage.csv"
```
On Databricks, all nodes in a cluster share a file system called DBFS (Databricks File System). The path format changes to:
```python
DATA_PATH = "/dbfs/FileStore/license_forecast/data/license_usage.csv"
```
`/dbfs/` is the local mount point that maps to the shared cloud storage. All worker nodes in the cluster can read this path simultaneously.

**3. Data can also be a Delta table**

Instead of just a CSV file, Databricks lets you save data as a **Delta table** — a structured table stored in cloud storage (S3/Azure Blob) with full versioning, ACID transactions, and SQL access. In Notebook 01 we save both a CSV (for compatibility) and a Delta table (for production use).

**4. Model Registry uses Unity Catalog**

On your laptop, model names are simple strings like `"license_forecast_jira"`. In Databricks with Unity Catalog enabled, the full name is:
```
main.license_forecast.license_forecast_jira
```
Format: `<catalog>.<schema>.<model_name>`. Unity Catalog adds access control, lineage tracking, and cross-workspace sharing.

**5. Model stages vs aliases**

On your laptop we used `stage="Production"` to mark the best model. Unity Catalog replaces stages with **aliases** — named pointers to a version. The equivalent is:
```python
client.set_registered_model_alias(model_name, "champion", version)
```
To load: `mlflow.prophet.load_model("models:/main.license_forecast.license_forecast_jira@champion")`

**6. `display()` instead of `print()`**

Databricks notebooks have a built-in `display()` function that renders DataFrames as interactive tables with charts. We use this instead of `print(df)`.

**7. `%pip install` at the top of notebooks**

Each notebook starts with:
```python
# MAGIC %pip install prophet mlflow
```
This installs packages on the cluster before the code runs. In Databricks, `%pip` is the notebook-level way to install packages (instead of `pip install` in the terminal).

---

### Step-by-Step: How to Run This on Databricks

**Step 1 — Create a Databricks Account**
- Sign up at `databricks.com` (free trial available)
- Or use Databricks Community Edition (free, limited but enough for this project)

**Step 2 — Connect Your GitHub Repo**
- In Databricks: Workspace → Git Folders → Add Git Folder
- Paste your GitHub URL: `https://github.com/thakur9sunil/license-forecast`
- Databricks clones your repo into the workspace — all notebooks are available instantly

**Step 3 — Create a Cluster**
- Go to: Compute → Create Cluster
- Choose: Databricks Runtime 14.x ML (this version has Prophet, scikit-learn, MLflow pre-installed)
- Single node is fine for this project (no distributed training needed)
- Install additional libraries from `databricks/cluster_requirements.txt`

**Step 4 — Run Notebooks In Order**
- Open `databricks/notebooks/01_data_generator.py` → Run All
- Open `databricks/notebooks/02_train_prophet.py` → Run All (takes ~15 mins)
- Open `databricks/notebooks/03_train_arima.py` → Run All
- Open `databricks/notebooks/04_evaluate.py` → Run All

**Step 5 — View Results in MLflow**
- Go to: Experiments (left sidebar) → `/Users/license_forecasting`
- All 48 training runs are there — same as your local MLflow UI

**Step 6 — Deploy as a REST Endpoint (Model Serving)**
- Go to: Serving → Create Serving Endpoint
- Select model: `license_forecast_jira` (Production or champion alias)
- Databricks provisions a scalable REST endpoint automatically
- You get a URL like: `https://<workspace>.azuredatabricks.net/serving-endpoints/license_forecast_jira/invocations`
- Update `VITE_API_BASE_URL` in the frontend to point to this URL

---

### What the Databricks Architecture Looks Like

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATABRICKS WORKSPACE (Cloud)                    │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────┐ │
│  │  Notebooks   │   │   MLflow     │   │   Unity Catalog         │ │
│  │  (your code) │──▶│  (built-in)  │──▶│   Model Registry        │ │
│  └──────┬───────┘   └──────────────┘   └────────────┬────────────┘ │
│         │                                            │              │
│  ┌──────▼───────┐                        ┌──────────▼────────────┐ │
│  │   Cluster    │                        │   Model Serving       │ │
│  │  (VMs/Spark) │                        │   (REST endpoint)     │ │
│  └──────┬───────┘                        └──────────┬────────────┘ │
│         │                                            │              │
│  ┌──────▼───────┐                                    │              │
│  │  Delta Lake  │                                    │              │
│  │  (DBFS/S3)   │                                    │              │
│  └──────────────┘                                    │              │
└──────────────────────────────────────────────────────┼─────────────┘
                                                        │
                                               ┌────────▼────────┐
                                               │  React Frontend │
                                               │  (unchanged)    │
                                               └─────────────────┘
```

---

### Why This Matters in an Interview

> "I designed the project to be portable. The local version runs everything on your laptop with a virtual environment and a local MLflow server. The Databricks version replaces those manual pieces with managed cloud services — the MLflow tracking URI goes away, local file paths become DBFS paths, and the FastAPI server is replaced by Databricks Model Serving. The core ML code — the Prophet training logic, the hyperparameter search, the ARIMA benchmark — stays exactly the same. This shows that good ML code is infrastructure-agnostic."

---

## What This Project Does (One-Line Summary)

A full-stack ML application that predicts how many software licenses (Jira, Slack, Zoom) a company will need at renewal time, and recommends whether to buy more, reduce, or hold.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FULL STACK                               │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ React/   │───▶│  FastAPI     │───▶│  MLflow Registry   │    │
│  │ Vite     │    │  Backend     │    │  (Prophet Models)  │    │
│  │ :3000    │◀───│  :8000       │◀───│  :5000             │    │
│  └──────────┘    └──────────────┘    └────────────────────┘    │
│                         │                                       │
│                  ┌──────▼──────┐                                │
│                  │  CSV Data   │                                │
│                  │ (36 rows)   │                                │
│                  └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Development Servers — What They Are, How They Work, and Where They Live

---

### Are These "Virtual Servers"? (Important Concept)

No — these are **not virtual machines or cloud servers**. They are simply **programs running on your own Windows laptop** that listen for network requests on specific port numbers.

Think of it like this: your laptop is the building, and each server is a different department inside that building sitting at a different desk number (port). Anyone inside the building (your browser, your code) can walk up to desk 5000 and talk to MLflow, or desk 8000 and talk to FastAPI.

The word "server" just means "a program that waits for requests and responds to them." It does not mean a separate physical machine or a cloud instance. All three servers here are just Python or Node.js processes running in your terminal — on your own CPU and RAM.

```
Your Laptop (Windows 11)
│
├── Port 3000  ──▶  Vite Dev Server        (Node.js process)
├── Port 8000  ──▶  uvicorn / FastAPI      (Python process)
└── Port 5000  ──▶  MLflow Server          (Python process)
```

---

### Server 1 — MLflow Tracking Server (Port 5000)

**What it is:**
MLflow is an open-source platform for managing machine learning experiments. Its server is a web application that provides both a visual UI (the webpage you see at port 5000) and a REST API that Python code talks to during training.

**How it was created:**
When you install `mlflow` via pip, it includes a built-in server component. We started it with:
```bash
python -m mlflow server \
  --backend-store-uri sqlite:///./mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 --port 5000
```

Breaking this command down:
- `python -m mlflow server` — tells Python to run MLflow's built-in server module
- `--backend-store-uri sqlite:///./mlflow/mlflow.db` — tells MLflow to store all experiment data (run names, metrics, parameters) in a SQLite database file on your hard drive. SQLite is a lightweight database that lives as a single `.db` file — no separate database server needed
- `--default-artifact-root ./mlflow/artifacts` — tells MLflow where to save model files, charts, and other large files on your hard drive
- `--host 0.0.0.0` — means "accept connections from any network interface" (not just localhost)
- `--port 5000` — the desk number this server sits at

**What runs underneath it:**
MLflow's server is built on top of **Flask** (a Python web framework) and served by **Gunicorn/uvicorn** internally. You don't see this — MLflow handles it automatically.

**What it stores on disk:**
```
license-forecast/
└── mlflow/
    ├── mlflow.db          ← SQLite database (experiment runs, metrics, params)
    └── artifacts/
        └── 1/             ← Experiment ID
            └── <run_id>/  ← Each training run
                └── model/ ← Saved Prophet model files
```

**How other services talk to it:**
The training scripts and the FastAPI backend both talk to MLflow through its REST API using the `MLFLOW_TRACKING_URI=http://localhost:5000` environment variable. When training logs a metric, it sends an HTTP POST request to MLflow's API. When FastAPI loads a model, it sends an HTTP GET to fetch the model artifact.

**Is it a virtual server?** No. It is a Python process running on your laptop, storing data in a file on your hard drive (`mlflow.db`).

---

### Server 2 — FastAPI Backend (Port 8000)

**What it is:**
This is the brain of the application — the REST API that sits between the frontend and the ML models. It receives forecast requests from the browser, runs the Prophet model, and returns predictions as JSON.

**How it was created:**
The backend code lives in `backend/app/`. We started it with:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Breaking this down:
- `uvicorn` — this is the actual server program. Uvicorn is an **ASGI server** (Asynchronous Server Gateway Interface) — it is the program that sits between your network and your Python code, receiving HTTP requests and passing them to FastAPI
- `app.main:app` — tells uvicorn where to find the FastAPI application object: in the file `app/main.py`, there is a variable called `app`
- `--reload` — development mode: automatically restarts the server whenever you save a code change (like live reload in a browser)
- `--host 0.0.0.0 --port 8000` — listen on port 8000 on all network interfaces

**What is ASGI and why does it matter?**
ASGI is a standard interface that allows Python web apps to handle many requests at the same time without waiting for one to finish before starting the next (asynchronous). This is important because when FastAPI is generating a forecast, it shouldn't block other users from getting a response. Uvicorn implements this standard and FastAPI is built on top of it.

**What happens at startup:**
When uvicorn starts, FastAPI runs a "lifespan" function — code that executes once before any requests are handled. This is where the `ModelLoader` singleton runs:
1. Connects to MLflow at `http://localhost:5000`
2. For each product (Jira, Slack, Zoom), finds the model version marked as "Production"
3. Downloads and loads that model into RAM
4. If MLflow is unreachable, falls back to loading `.pkl` files from disk

After startup, the models sit in memory and every forecast request uses them instantly — no reloading needed.

**What runs underneath it:**
- **FastAPI** — the Python framework that defines your routes (`/forecast`, `/health`, `/products`) and handles request/response logic
- **uvicorn** — the server that actually receives TCP connections and HTTP packets from the network and passes them to FastAPI
- **Pydantic** — used by FastAPI to validate incoming request data and format outgoing responses

**Auto-generated API documentation:**
FastAPI automatically creates interactive documentation at `http://localhost:8000/docs` — you can open this in a browser and test every endpoint without writing any code.

**Is it a virtual server?** No. It is a Python process (uvicorn) running on your laptop, loaded into your laptop's RAM.

---

### Server 3 — Vite Dev Server / React Frontend (Port 3000)

**What it is:**
This is the development server for the React user interface. It serves the HTML, CSS, and JavaScript files to your browser. This server only exists during development — in production you would build the files into static assets and serve them via Nginx.

**How it was created:**
The frontend code lives in `frontend/src/`. We started it with:
```bash
cd frontend
npm install    # downloaded all JavaScript packages into node_modules/
npm run dev    # started the Vite dev server
```

Breaking this down:
- `npm install` — reads `package.json` and downloads all required JavaScript libraries (React, Recharts, Tailwind, etc.) into a `node_modules/` folder on your hard drive
- `npm run dev` — runs the `dev` script defined in `package.json`, which launches Vite's development server

**What is Vite?**
Vite is a **JavaScript build tool and development server**. In development mode it does two things:
1. **Serves files instantly** — when your browser requests `App.jsx`, Vite transforms it on the fly and sends it. No waiting for a full build
2. **Hot Module Replacement (HMR)** — when you save a React component file, Vite sends only the changed module to the browser and React swaps it in without refreshing the page. You see changes in under 100 milliseconds

**What runs underneath it:**
- **Node.js** — the JavaScript runtime. Vite is a Node.js program. Node.js lets JavaScript run outside a browser, on your laptop directly
- **Vite** — the dev server and module bundler built on top of Node.js
- **React** — the UI library that runs inside the browser (not on the server)
- **Recharts** — a charting library built on top of React for drawing the forecast line charts
- **Tailwind CSS** — processes CSS utility classes into actual styles

**How the browser talks to the backend:**
The React app doesn't talk to MLflow or the ML models directly. It only talks to the FastAPI backend at port 8000. The `useForecast` hook in React calls `GET http://localhost:8000/forecast/Jira?horizon_months=6`, gets back JSON, and passes the data to Recharts to draw the chart.

**Why only for development?**
The Vite dev server adds extra overhead (file watching, source maps, HMR) that you don't want in production. For production deployment, you would run `npm run build` which creates a `dist/` folder of optimized static files (plain HTML, CSS, JS). Those files would be served by a lightweight web server like Nginx (which is already configured in `frontend/nginx.conf`).

**Is it a virtual server?** No. It is a Node.js process running on your laptop.

---

### How All Three Servers Talk to Each Other

```
Browser (Chrome/Edge)
    │
    │  HTTP GET http://localhost:3000
    ▼
┌─────────────────────────┐
│   Vite Dev Server       │  Node.js process — serves React app files
│   Port 3000             │
└──────────┬──────────────┘
           │ React code runs in the browser
           │ useForecast hook sends:
           │ HTTP GET http://localhost:8000/forecast/Jira
           ▼
┌─────────────────────────┐
│   uvicorn + FastAPI     │  Python process — runs the API
│   Port 8000             │
└──────────┬──────────────┘
           │ On startup: loads models from MLflow
           │ HTTP GET http://localhost:5000/api/...
           ▼
┌─────────────────────────┐
│   MLflow Server         │  Python process — stores experiments & models
│   Port 5000             │  Data saved in mlflow/mlflow.db (SQLite file)
└─────────────────────────┘
```

**Request flow in plain English:**
1. You open `http://localhost:3000` — your browser downloads the React app from Vite
2. React renders the UI — you pick "Jira" and "6 months"
3. React calls `http://localhost:8000/forecast/Jira?horizon_months=6`
4. FastAPI receives the request, uses the already-loaded Prophet model, generates a forecast
5. FastAPI returns JSON with historical data + forecast data + recommendation
6. React passes the JSON to Recharts which draws the line chart
7. You see the chart and recommendation card on screen

---

### Summary Table

| Server | Program | Language | Port | Data Storage | Lives In |
|--------|---------|----------|------|-------------|---------|
| MLflow | Flask + Gunicorn (built-in) | Python | 5000 | `mlflow/mlflow.db` (SQLite) + `mlflow/artifacts/` | Python process on your laptop |
| FastAPI | uvicorn | Python | 8000 | RAM (models loaded at startup) | Python process on your laptop |
| React UI | Vite | Node.js | 3000 | No storage — serves files only | Node.js process on your laptop |

**All three are local processes running on your Windows laptop. None of them are virtual machines, cloud instances, or Docker containers (in this development setup). They communicate with each other over your laptop's internal network using localhost (127.0.0.1).**

---

## Step-by-Step Process (In Order)

---

### STEP 1 — Project Setup
**What we did:**
- Created a new folder `license-forecast`
- Initialized a Git repository (`git init`)
- Created a Python virtual environment (`.venv`) to isolate dependencies
- Created `.env` file with configuration (MLflow URI, ports, CORS settings)
- Created `.gitignore` to exclude sensitive files, model binaries, and build artifacts

**Tools Used:** Git, Python venv

**Why this matters in an interview:**
> "I always start by isolating the Python environment to avoid dependency conflicts, and setting up Git early so every change is tracked from day one."

---

### STEP 2 — Install Dependencies
**What we did:**
- Installed all Python packages from `requirements.txt` inside the virtual environment
- Key packages: `prophet`, `statsmodels`, `mlflow`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `pytest`
- Installed DVC (Data Version Control) globally for pipeline versioning

**Tools Used:** pip, DVC

**Why this matters in an interview:**
> "Using a requirements.txt ensures anyone can reproduce the exact environment. DVC complements Git by versioning large data files and ML pipelines that Git isn't designed to handle."

---

### STEP 3 — Generate Synthetic Training Data
**What we did:**
- Ran `python -m ml.data_generator`
- Generated 36 rows of monthly license usage data (12 months × 3 products)
- Saved to `backend/data/raw/license_usage.csv`

**Data shape:**
```
date        | usage | product
2023-01-01  |  541  | Jira
2023-01-01  |  653  | Slack
2023-01-01  |  427  | Zoom
... (12 months each)
```

**Tools Used:** pandas, numpy

**Why this matters in an interview:**
> "In real projects you'd connect to an ITSM or SaaS admin API. Here I used synthetic data with realistic trends — Jira growing, Zoom declining — to simulate a real business scenario without needing live data."

---

### STEP 4 — Start MLflow Tracking Server
**What we did:**
- Started MLflow server as a background process
- Configured it with a SQLite backend database and local artifact storage
- Accessible at `http://localhost:5000`

**Command:**
```bash
python -m mlflow server \
  --backend-store-uri sqlite:///./mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 --port 5000
```

**Tools Used:** MLflow, SQLite, uvicorn (internally)

**Why this matters in an interview:**
> "MLflow acts as the experiment tracker and model registry. Every training run — including its parameters, metrics, and model artifact — is logged automatically. This is how you go from 'it works on my machine' to a reproducible, auditable ML process."

---

### STEP 5 — Train Prophet Models (Hyperparameter Search)
**What we did:**
- Ran `python -m ml.train_prophet`
- For each product (Jira, Slack, Zoom), ran **16 hyperparameter combinations**:
  - Seasonality mode: `multiplicative` vs `additive`
  - Changepoint prior scale: `0.01`, `0.05`, `0.1`, `0.3`
  - Seasonality prior scale: `1.0`, `10.0`
- Each run was logged to MLflow with MAPE (Mean Absolute Percentage Error)
- Best model per product was automatically promoted to **Production** in the MLflow registry

**Best results:**
| Product | Best MAPE | Best Config |
|---------|-----------|-------------|
| Jira    | ~1%       | multiplicative, low changepoint |
| Slack   | 0.27%     | multiplicative, cp=0.1, sp=1.0 |
| Zoom    | 0.64%     | additive, cp=0.3, sp=1.0 |

**Tools Used:** Prophet (Facebook/Meta), MLflow, cmdstanpy

**Why this matters in an interview:**
> "Instead of manually tuning, I used a grid search over 16 combinations and let MLflow track every experiment. The best model is then promoted to Production — this mirrors how teams manage model versioning in real MLOps workflows."

---

### STEP 6 — Train ARIMA Models (Benchmark)
**What we did:**
- Ran `python -m ml.train_arima`
- Fitted SARIMA(1,1,1)(1,1,1,12) — a seasonal ARIMA model — for each product
- Logged metrics to MLflow for comparison against Prophet

**Results:**
| Product | MAPE   |
|---------|--------|
| Jira    | 8.36%  |
| Slack   | 16.03% |
| Zoom    | 9.59%  |

**Tools Used:** statsmodels, MLflow

**Why this matters in an interview:**
> "ARIMA is a classical statistical model — fast and interpretable. Comparing it against Prophet quantitatively (not just by feel) is good ML practice. Prophet won on this dataset because it handles trend shifts better than ARIMA with only 12 data points."

---

### STEP 7 — Run Model Evaluation
**What we did:**
- Ran `python -m ml.evaluate`
- Compared Prophet vs ARIMA across all products
- Saved results to `backend/ml/metrics/prophet_metrics.json` and `arima_metrics.json`

**Tools Used:** MLflow client API, pandas

---

### STEP 8 — Start FastAPI Backend
**What we did:**
- Ran `uvicorn app.main:app --reload --port 8000`
- On startup, `ModelLoader` singleton connects to MLflow and loads the **Production** Prophet model for each product
- Falls back to local `.pkl` files if MLflow is unavailable

**Key API Endpoints:**
```
GET  /health              → {"status":"healthy","models_loaded":{...}}
GET  /forecast/Jira?horizon_months=6
GET  /products
GET  /metrics
```

**Tools Used:** FastAPI, uvicorn, MLflow Python client, pandas

**Why this matters in an interview:**
> "The singleton pattern for model loading means we load the model once at startup, not on every request — which would be very slow. FastAPI's lifespan hooks are perfect for this pattern."

---

### STEP 9 — Start React Frontend
**What we did:**
- Ran `npm install` then `npm run dev`
- Vite dev server started at `http://localhost:3000`
- Frontend calls the backend via the `useForecast` hook
- Displays a Recharts line chart with historical + forecast data and recommendation cards

**Tools Used:** React, Vite, Recharts, Tailwind CSS, Node.js, npm

---

### STEP 10 — Run Tests
**What we did:**
- Ran `pytest tests/ -v`
- 29 tests across 3 test files

**Test coverage:**
| File | Tests |
|------|-------|
| test_data_generator.py | 8 tests — data shape, trends, columns |
| test_forecast_router.py | 10 tests — API endpoints, validation |
| test_forecaster_service.py | 11 tests — business logic, recommendations |

**Result: 29/29 passed** (after fixing one bug — see Challenges below)

**Tools Used:** pytest, FastAPI TestClient

---

### STEP 11 — Initialize DVC
**What we did:**
- Ran `dvc init` to set up Data Version Control
- DVC tracks the data pipeline stages defined in `dvc.yaml`
- Stages mirror the ML pipeline: generate → train_prophet → train_arima → evaluate

**Tools Used:** DVC

**Why this matters in an interview:**
> "Git tracks code. DVC tracks data and model artifacts. Together they give you full reproducibility — you can checkout any git commit and reproduce the exact data and models that were used."

---

### STEP 12 — Git Commit & Push to GitHub
**What we did:**
- Staged all project files (excluding `.venv`, data, models per `.gitignore`)
- Created initial commit
- Pushed to GitHub: `https://github.com/thakur9sunil/license-forecast`

**Tools Used:** Git, GitHub

---

## Challenges Faced & How They Were Solved

---

### Challenge 1 — MLflow Server Wouldn't Start in Background
**Problem:** Running `mlflow server &` in Git Bash on Windows killed the process when the shell exited. The server wasn't staying alive.

**Root Cause:** Git Bash background processes don't persist between shell sessions on Windows.

**Solution:** Used `python -m mlflow server` via a persistent background task runner, which kept the process alive independently.

**Lesson:** On Windows, background process management behaves differently than Linux/macOS. Always verify the process is running with a health check before proceeding.

---

### Challenge 2 — Prophet Wouldn't Initialize (CmdStan Compilation Failure)
**Problem:** Running `Prophet()` threw `AttributeError: 'Prophet' object has no attribute 'stan_backend'`. Prophet uses Stan (a statistical computing language) under the hood, which requires a C++ compiler to build.

**Root Cause (layered):**
1. The system's MinGW compiler (GCC 7.3, 32-bit) was too old — it didn't support C++14 features that CmdStan 2.38 requires (`std::mutex`, `std::thread`)
2. A newer cmdstanpy (1.3.0) validates the CmdStan path strictly by checking for a `makefile` — but Prophet's **own bundled pre-compiled CmdStan 2.33.1** (shipped inside the pip package) doesn't include one

**Solution:** Created a one-line dummy `makefile` in Prophet's bundled CmdStan directory:
```
CMDSTAN_VERSION = 2.33.1
```
This satisfied cmdstanpy's validation check, allowing it to accept the pre-compiled binary that Prophet already ships with — no C++ compilation needed at all.

**Lesson:** When a package ships pre-compiled binaries, you don't always need to rebuild from source. Reading the source code of the dependency (Prophet's `models.py`) revealed the simpler path.

---

### Challenge 3 — Windows Unicode Encoding Error
**Problem:** MLflow printed emoji characters (🏃, 🧪) in its output, which caused a `UnicodeEncodeError` on Windows because the default encoding is `cp1252`, not UTF-8.

**Solution:** Set `PYTHONIOENCODING=utf-8` as an environment variable before running the training scripts.

**Lesson:** Always set `PYTHONIOENCODING=utf-8` when running Python scripts on Windows that may print Unicode characters.

---

### Challenge 4 — Failing Test: `test_recommendation_hold`
**Problem:** The test passed `yhat_upper=540` and `current_licensed=550` and expected `"hold"`, but got `"buy_more"`.

**Root Cause:** The recommender applied a 10% safety buffer to `yhat_upper` (`540 × 1.10 = 594`) and then compared against `current_licensed × 1.05` (`550 × 1.05 = 577.5`). Since `594 > 577.5`, it triggered `buy_more` — even though the forecast was clearly below the license count.

**Solution:** Changed the threshold to `yhat_upper > current_licensed` directly. If the upper confidence bound of the forecast doesn't exceed your current license count, there's no reason to buy more. The buffer is still used to calculate *how many* to buy, just not to decide *whether* to buy.

**Lesson:** Safety buffers in business logic can compound in unexpected ways. Unit tests caught a real business logic bug — this is exactly what tests are for.

---

## Full Tools & Technologies Summary

| Category | Tool | Purpose |
|----------|------|---------|
| Language | Python 3.14 | Backend & ML |
| Language | JavaScript (ES6+) | Frontend |
| ML Model | Prophet (Meta) | Time-series forecasting |
| ML Model | SARIMA (statsmodels) | Benchmark forecasting |
| Stan Backend | cmdstanpy + CmdStan | Prophet's probabilistic engine |
| Experiment Tracking | MLflow | Log runs, register models, compare metrics |
| API Framework | FastAPI | REST API with auto docs |
| API Server | uvicorn | ASGI server for FastAPI |
| Frontend Framework | React + Vite | UI with fast HMR dev server |
| Charting | Recharts | Line charts for forecasts |
| Styling | Tailwind CSS | Utility-first CSS |
| Data Manipulation | pandas, numpy | Data processing |
| Testing | pytest + FastAPI TestClient | Unit & integration tests |
| Data Versioning | DVC | Version data files & pipeline stages |
| Version Control | Git + GitHub | Source code versioning |
| Package Manager (Python) | pip + venv | Dependency isolation |
| Package Manager (JS) | npm | Frontend dependencies |
| Database (MLflow) | SQLite | MLflow experiment storage |
| Containerization (defined) | Docker + docker-compose | Full stack deployment |

---

## How to Explain This Project in an Interview

---

### Part 1 — What the Project Does (Start Here)

"I built a full-stack machine learning application that helps a company manage their software license costs.

The problem it solves is simple: companies pay for software licenses — tools like Jira (project management), Slack (messaging), and Zoom (video calls). They often either overpay by buying too many licenses, or run short and can't onboard new employees. My application looks at past usage data, predicts how many licenses will be needed at the next renewal date, and tells you whether to buy more, reduce, or keep the same number.

The entire system runs as a web application — you open a browser, pick a product, choose how far ahead you want to forecast (3, 6, or 12 months), and it shows you a chart with the prediction and a recommendation."

---

### Part 2 — How the Data Works

"I started by generating synthetic (fake but realistic) monthly usage data for 12 months across 3 products — 36 rows total. In a real company this data would come from the admin dashboard of each SaaS tool.

The data has three columns: date, number of users, and product name. I built in realistic patterns — Jira usage grows over time (company hiring), Zoom usage declines (back to office), and Slack stays relatively stable. This makes the forecasting problem realistic even without real data."

---

### Part 3 — The Machine Learning (Explain the Models Simply)

"I trained two types of forecasting models and compared them:

**Model 1 — Prophet (built by Meta/Facebook)**
Prophet is a time-series forecasting tool designed for business data. Think of it as a smart trend detector. You give it historical data points and it learns three things:
- The overall trend (is usage going up or down over months?)
- Seasonality (does usage spike at certain times of year?)
- Holiday effects (optional — unusual spikes or dips)

Prophet works by fitting a mathematical curve to your data using a technique called **Stan** — which is a statistical modeling language. Stan works like this: instead of giving you one single prediction, it runs thousands of simulations of 'what could happen' based on your data and gives you a range of likely outcomes. That range is called the **confidence interval** — the lower and upper bounds of what usage might look like. We use the upper bound to be safe when recommending licenses.

**What is Stan exactly?**
Stan is not something you directly write — it's an engine that runs underneath Prophet. Prophet writes the statistical problem in Stan's language, and Stan solves it. Stan is written in C++ which is why it needs to be compiled (turned into machine-executable code) before it can run. This became one of our main technical challenges on Windows, which I'll explain shortly.

**Model 2 — SARIMA (classical statistics)**
SARIMA stands for Seasonal AutoRegressive Integrated Moving Average. It is the traditional way to forecast time-series data, used for decades in finance and economics. It looks at patterns in past data — specifically how each month's usage relates to previous months — and projects that pattern forward.

I trained SARIMA as a benchmark — a baseline to compare against Prophet. SARIMA got 8–16% error (MAPE) while Prophet got under 1% on this dataset. Prophet won because it handles trend changes better with limited data."

---

### Part 4 — MLflow: Experiment Tracking (Very Important to Explain Well)

"When training machine learning models, you have something called **hyperparameters** — settings you choose before training that affect how the model learns. For example, how sensitive should the model be to sudden trend changes? How strongly should it weight seasonal patterns?

Instead of guessing the best settings, I ran **16 combinations** of hyperparameters for each of the 3 products — that's 48 training runs in total. Each run produced a different model with different accuracy.

**MLflow** is the tool I used to track all of this. Think of MLflow like a spreadsheet that automatically records every experiment:
- What settings (hyperparameters) were used
- What accuracy (MAPE score) the model achieved
- The actual trained model file saved as an artifact

After all 48 runs completed, I used MLflow's **Model Registry** to promote the best-performing model for each product to 'Production' status. This means the live application always loads the best version automatically.

MLflow also provides a web UI at port 5000 where you can visually compare all runs, see charts of metrics, and manage which model version is active."

---

### Part 5 — The Backend API (FastAPI)

"The backend is built with **FastAPI** — a modern Python web framework. FastAPI is fast to develop with and automatically generates interactive API documentation.

When the server starts up, it uses a design pattern called a **Singleton** to load the ML models once into memory. A singleton means: create this object only one time and reuse it for every request. Loading an ML model takes a few seconds, so doing it once at startup (instead of on every request) makes the API fast.

The main endpoint is `GET /forecast/Jira?horizon_months=6`. When called:
1. It reads the historical CSV data
2. Uses the loaded Prophet model to generate future predictions
3. Passes the prediction to a recommendation engine
4. Returns historical points + forecast points + a buy/reduce/hold recommendation as JSON

The recommendation logic compares the upper confidence interval of the forecast against how many licenses the company currently has. If the upper bound exceeds current licenses, recommend buying more. If predicted usage drops more than 20% below current licenses, recommend reducing. Otherwise, hold."

---

### Part 6 — The Frontend (React)

"The frontend is a **React** application built with **Vite** as the build tool. Vite is much faster than the older Create React App setup — it starts in under a second.

The UI has:
- A product selector (Jira / Slack / Zoom)
- A horizon selector (3, 6, or 12 months ahead)
- A **Recharts** line chart showing the historical usage as a solid line and the forecast as a dashed line with a shaded confidence band
- Summary cards showing the recommendation, predicted count, and percentage change

The React hook `useForecast` handles all the API communication — when the user changes the product or horizon, it automatically fetches the new forecast from the backend and updates the chart."

---

### Part 7 — DVC: Data Version Control

"**DVC** (Data Version Control) works alongside Git. Git is great for tracking code changes, but it's not designed for large files like datasets or trained model files (which can be hundreds of megabytes).

DVC solves this by storing the actual data files separately (in local storage or cloud storage like S3) while Git only stores a small pointer file that says 'the data for this commit is at this location.' This way your Git repo stays small but you can always reproduce the exact data that was used for any commit.

I also used DVC to define the ML pipeline as stages: generate data → train Prophet → train ARIMA → evaluate. This means anyone can run `dvc repro` and get the exact same results."

---

### Part 8 — Technical Challenges (This Shows Problem-Solving Skills)

"The most interesting challenge was getting Prophet to work on Windows.

Prophet uses a statistical engine called Stan, which is written in C++ and needs to be compiled into machine code before it can run. On Linux and Mac, this happens automatically. On Windows, you need a C++ compiler like MinGW installed.

Our Windows machine had MinGW but it was version 7.3 from 2017 — too old to support the C++14 language features that Stan requires. We tried installing a newer compiler but ran into permission issues.

The breakthrough came from reading Prophet's source code directly. I discovered that Prophet actually ships its own pre-compiled Stan model binary inside the pip package — it's a `.bin` file that's already compiled and ready to run. The only reason it wasn't being used was that the newer version of cmdstanpy (the Python wrapper for Stan) was checking for a file called `makefile` in the bundled folder to validate the installation. That file was missing.

The fix was just one line — creating a dummy `makefile` with the version number. That satisfied the validation check and Prophet immediately started working using its own pre-compiled binary. No C++ compiler needed at all.

This kind of debugging — where you go from a confusing error message to reading library source code to finding a simple root cause — is something I find very satisfying."

---

### Part 9 — Testing

"I wrote 29 automated tests using **pytest**:
- Tests for the data generator (correct shape, no negative values, correct trends)
- Tests for the API endpoints (correct responses, error handling for invalid inputs)
- Tests for the business logic (does the recommendation engine give the right answer for buy/reduce/hold scenarios?)

One test actually caught a real bug: the recommendation engine was applying a 10% safety buffer to the forecast AND a 5% tolerance to the license count — both at the same time. This caused it to recommend 'buy more' even when the forecast was clearly below the license count. The fix was to simplify the logic: only recommend buying more if the forecast's upper bound actually exceeds the current license count.

This is exactly why you write tests — they catch logic bugs that look correct when you write them but fail on edge cases."

---

### Quick Reference: What Each Tool Does (For Follow-Up Questions)

| If asked about... | Say this... |
|-------------------|-------------|
| **Prophet** | "A forecasting library from Meta. It learns trend, seasonality, and holiday patterns from historical data and predicts future values with a confidence range." |
| **Stan / CmdStan** | "The statistical engine inside Prophet. It runs thousands of simulations to give a range of predictions rather than just one number. Written in C++ so it needs compilation." |
| **MLflow** | "An experiment tracking tool. It logs every training run's settings and accuracy, stores the model files, and lets you promote the best model to production." |
| **FastAPI** | "A Python web framework for building REST APIs. Fast to develop, automatically generates API docs, and handles async requests efficiently." |
| **DVC** | "Data Version Control. Like Git but for large data files and ML pipelines. Keeps the Git repo small while ensuring data and models are reproducible." |
| **Vite** | "A modern JavaScript build tool that starts a development server in under a second. Much faster than older tools like Webpack." |
| **SARIMA** | "A classical statistical forecasting model. It finds patterns between each time point and previous time points to project the future. Used here as a benchmark to compare against Prophet." |
| **Singleton pattern** | "A design pattern where you create an expensive object (like a loaded ML model) only once and reuse it. Avoids reloading the model on every API request." |
| **Confidence interval** | "The range of values the model thinks the true answer could fall within. The upper bound is the worst case — useful for license planning because you'd rather have too many than too few." |
| **Hyperparameter search** | "Testing multiple combinations of model settings to find which one gives the most accurate predictions. Like tuning a recipe by trying different amounts of each ingredient." |
