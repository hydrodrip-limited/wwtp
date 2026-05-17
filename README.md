# WWTP Effluent NH₄ Predictor

[![CI](https://github.com/your-org/wwtp-rnn/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/wwtp-rnn/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/your-org/wwtp-rnn/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/wwtp-rnn)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org)
[![MLflow 3.12](https://img.shields.io/badge/MLflow-3.12-orange)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Professional reproduction of **Wongburi & Park (2023)** — _"Prediction of Effluent Quality from a Wastewater Treatment Plant Using Machine Learning"_ — applied to the Tilburg WWTP online sensor dataset.

## Results

| Model                             | RMSE | MAE  | R²   | Train time |
| --------------------------------- | ---- | ---- | ---- | ---------- |
| RNN (50 epochs, batch 100, lb 8)  | 0.42 | 0.31 | 0.87 | ~45 s      |
| LSTM (50 epochs, batch 100, lb 8) | 0.38 | 0.28 | 0.89 | ~62 s      |

> All metrics on the **held-out test set** (last 20 % of the chronological time series).

---

## Quick start

### Prerequisites

- Python ≥ 3.11
- Docker + Docker Compose v2

### 1. Install in development mode

```bash
git clone https://github.com/hydrodrip-limited/wwtp.git
cd wwtp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

### 3. Run the Streamlit UI

#### With Docker Compose (recommended)

```bash
docker compose up
# UI → http://localhost:8502
```

## Development

### Run the test suite

```bash
# Unit tests only (fast, no GPU, no real model)
pytest tests/unit -v

# Unit tests with coverage gate (≥ 80 %)
pytest tests/unit --cov=wwtp --cov-report=term-missing

# Integration tests (trains for 2 epochs — ~30 s on CPU)
pytest tests/integration -m integration -v
```

### Code quality

```bash
ruff format src tests      # auto-format
ruff check src tests       # lint
mypy src/wwtp              # strict type check
```

### Pre-commit hooks (run automatically on git commit)

```bash
pre-commit install
```

---

## Configuration reference

All parameters can be set via environment variables (or in `.env`).

| Variable                 | Default                             | Description                                     |
| ------------------------ | ----------------------------------- | ----------------------------------------------- |
| `MLFLOW_TRACKING_URI`    | `file:./mlruns`                     | MLflow backend store                            |
| `MLFLOW_EXPERIMENT_NAME` | `wwtp_effluent_prediction`          | Experiment name                                 |
| `MODEL_NAME`             | `wwtp_effluent_predictor`           | Registered model name                           |
| `DATA__TARGET_COL`       | `NH4`                               | Prediction target (`NH4`, `DO_1`…`DO_3`, `NO3`) |
| `MODEL__LOOKBACK`        | `8`                                 | Lookback window (timesteps)                     |
| `MODEL__EPOCHS`          | `50`                                | Training epochs                                 |
| `MODEL__BATCH_SIZE`      | `100`                               | Mini-batch size                                 |
| `API_URL`                | `http://localhost:5000/invocations` | Streamlit → MLflow server URL                   |
| `SCALERS_PATH`           | _(see .env.example)_                | Path to `*_scalers.joblib`                      |
| `LOG_FORMAT`             | `human`                             | `human` (dev) or `json` (prod)                  |

Use `MODEL__LOOKBACK=16` syntax (double underscore) to override nested sub-models.

---

## License

MIT — see [LICENSE](LICENSE).
