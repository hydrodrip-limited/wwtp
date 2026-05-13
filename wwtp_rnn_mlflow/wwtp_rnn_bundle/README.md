# WWTP Effluent Prediction — RNN/LSTM with MLflow

Implementation of the Recurrent Neural Network architecture from
**Wongburi & Park (2023), _Prediction of Wastewater Treatment Plant Effluent
Water Quality Using Recurrent Neural Network (RNN) Models_** (Water, 15, 3325),
applied to the Tilburg WWTP `dynamic-modeling-of-wastewater-treatment-process`
dataset and wrapped with MLflow tracking, model registry, and deployment.

## What this implements

The paper's modeling section (2.4, Figures 4 and 5) lays out a five-step
pipeline. Each step lives in its own function in `src/model.py`:

| Paper step | Function |
|---|---|
| 1. Define Network | `define_network` |
| 2. Compile Network | `compile_network` |
| 3. Fit Network | `fit_network` |
| 4. Evaluate Network | `evaluate_network` |
| 5. Make Predictions | `make_predictions` |

Defaults match the paper's "optimal" hyperparameters: 50 epochs, batch size
100, Adam optimizer, MAE loss, no shuffle (time series). Both `SimpleRNN`
and `LSTM` variants are trained and compared in the same MLflow experiment
exactly as in Table 5 of the paper.

## Data adaptation

The Tilburg dataset comes with an explicit input/output contract in its data
dictionary. We follow it directly — the `train_val_test_online_dataset.csv`
INPUT/OUTPUT split is:

```
INPUTS  (controllable / influent):  Q_inf, Q_air_1..5, Temp       [7 features]
OUTPUTS (downstream responses):     DO_1, DO_2, DO_3, NO3, NH4
```

Default target is `NH4` (closest analog to the paper's "effluent NH3-N").
Any of the five outputs can be swapped in via `--target`; nothing else has
to change. Both `train_val_test_online_dataset.csv` (training) and
`evaluation_online_dataset.csv` (deployment scoring) share the input schema,
so the same model handles both.

The `train_val_test_lab_dataset.xlsx` file maps more directly onto the
paper (influent water quality → effluent water quality across COD, BOD5,
TKN, NH4, NOx, Ntot, TSS), but it has only 35 paired observations across
nine months. That's far too sparse for an RNN/LSTM; it's noted here as a
secondary modeling option for use with classical regressors or for
transfer-learning a model trained on the online data.

The `*_dynamic_influent.csv` files are synthetic ASM1 state fractionations
intended as input to a mechanistic model — not used here.

## Project layout

```
wwtp_rnn/
├── src/
│   ├── data_prep.py        # Section 2.2-2.3: scaling, windowing, split
│   ├── model.py            # Section 2.4: the 5-step network functions
│   ├── train_mlflow.py     # Orchestrator: trains both RNN/LSTM, logs to
│   │                       #   MLflow, registers best run
│   └── predict.py          # Loads from MLflow registry and runs inference
├── artifacts/              # Loss curves, prediction plots, scalers, CSVs
└── mlruns/                 # MLflow file-backend tracking store
```

## Running it

```bash
# 1. Train and register (default target = NH4)
python src/train_mlflow.py \
    --data-path /path/to/train_val_test_online_dataset.csv \
    --target NH4 \
    --lookback 8 \
    --epochs 50 \
    --batch-size 100 \
    --tracking-uri file:./mlruns

# 2. Train a different target (any of DO_1, DO_2, DO_3, NO3, NH4)
python src/train_mlflow.py --target DO_3 ... # same other flags

# 3. Inspect runs in the UI
mlflow ui --backend-store-uri file:./mlruns
# -> http://localhost:5000

# 4. Score the evaluation holdout via the registered model
python src/predict.py \
    --data-path /path/to/evaluation_online_dataset.csv \
    --model-name wwtp_effluent_predictor \
    --version latest \
    --target NH4 \
    --lookback 8 \
    --output evaluation_predictions.csv

# 5. Or serve as an HTTP endpoint (no code changes)
mlflow models serve -m "models:/wwtp_effluent_predictor/latest" -p 5000
```

## What MLflow captures per run

- **Params**: architecture, units, epochs, batch_size, optimizer, loss,
  lookback, n_features, target, train/test sizes, seed
- **Metrics**: per-epoch `train_loss` and `val_loss`, final `rmse`, `mae`,
  `r2`, and `train_seconds` — all RMSE/MAE in the target's **original units
  (mg/L)** after inverse-scaling, so they're directly interpretable
- **Artifacts**: model summary, feature names JSON, train/test loss curve
  PNG, prediction-vs-actual PNG, test-set predictions CSV, fitted scalers
- **Model**: TensorFlow model with signature + input example, ready for
  `mlflow.pyfunc.load_model` or `mlflow models serve`
- **Registry**: best run by RMSE (per training invocation) is auto-promoted
  to a new version of `wwtp_effluent_predictor`

## Included runs

Two targets trained at lookback=8 (≈ 2 hours of context at 15-min
intervals), epochs=50, batch=100:

| Target | Architecture | RMSE (mg/L) | MAE | R² | Train time |
|---|---|---|---|---|---|
| **NH4** | SimpleRNN ★ | 1.43 | 1.08 | 0.34 | 63 s |
| NH4 | LSTM | 1.76 | 1.27 | −0.01 | 67 s |
| **DO_3** | LSTM ★ | 0.41 | 0.32 | 0.11 | 67 s |
| DO_3 | SimpleRNN | 0.45 | 0.27 | −0.08 | 63 s |

★ = registered version. NH4 best is `wwtp_effluent_predictor` v1; DO_3
best is v2.

Inference against the actual evaluation file
(`evaluation_online_dataset.csv`, March–June 2022, no outputs provided)
yields 8,631 predictions for NH4 in `artifacts/evaluation_predictions.csv`.

These R² values are much lower than the paper's ~0.95+. Two honest reasons:

1. **Input set is harder.** The paper uses direct influent water quality
   (BOD5/TSS/TKN/NH3-N/TP) mechanistically tied to effluent. Here the
   inputs are operational controls (aeration rates, temperature) — the
   model has to learn the entire activated-sludge biological response
   from those controls alone. A single-layer RNN can track gross
   dynamics but won't capture spike magnitudes.
2. **Test-fold distribution shift.** The DO_3 prediction plot makes this
   visible: the later portion of the test fold has substantially higher
   DO_3 than anything in the training fold. R² collapses on regime
   change. This is the kind of issue you'd plan for in a production
   SCADA pipeline — drift detection, retraining triggers, mechanistic
   priors.

### Concrete next moves to lift accuracy (in order of expected impact)

1. **Add lagged target values as autoregressive features.** Strongest
   single change for this dataset — most of the next-step prediction
   problem is recovered by `NH4(t-1)`.
2. **Stack a second RNN/LSTM layer.** The paper uses a single layer; with
   25k+ samples, depth helps.
3. **Lengthen lookback** to 24-96 (6-24 hours). NH4 dynamics in biological
   nitrification have hour-scale time constants.
4. **Bring in the mechanistic ASM1 state from
   `train_val_test_dynamic_influent.csv`** as additional inputs — this is
   what the competition designers intended for closing the gap on rain
   events and seasonality not represented in the training fold.

## Production notes

- The file-backend MLflow store is fine for local development. MLflow 3.x
  emits deprecation warnings on it — for production switch to
  `sqlite:///mlflow.db` (single-node) or Postgres + S3/Azure Blob (team).
  The `--tracking-uri` flag is the only thing that changes.
- The fitted scalers must travel with the model. They are logged as a
  joblib artifact under the source run and re-downloaded by `predict.py`
  via the `run_id` captured from the registered model version. This keeps
  inference reproducible without baking scalers into the model graph.
- For Azure/Databricks deployment, point `--tracking-uri` at the
  workspace URI (`databricks` or `azureml://...`) and
  `mlflow.tensorflow.log_model` works unchanged. The same code runs
  locally, in Databricks Jobs, or in an Azure ML pipeline.
