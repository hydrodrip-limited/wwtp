"""
Training orchestrator — runs the paper's 5-step pipeline for both RNN and LSTM
architectures, logs every run to MLflow, and registers the best variant in the
MLflow Model Registry.

Changes vs the original ``train_mlflow.py``
-------------------------------------------
* Default ``lookback`` is **8** (was 1, causing silent shape mismatches).
* Uses a **separate validation set** (70/10/20 split) so test-set metrics
  reflect true held-out generalization.
* Scalers are **bundled** into the MLflow model artifact via
  :class:`~wwtp.predictor.WwtpPredictor` — no more fragile run_id lookups.
* All ``print`` calls replaced with structured logging.
* ``import joblib`` moved to module top level.

Usage
-----
    python -m wwtp.train \\
        --data-path train_val_test_online_dataset.csv \\
        --target NH4 \\
        --lookback 8 \\
        --epochs 50 \\
        --batch-size 100

After training, start the MLflow UI::

    mlflow ui --backend-store-uri file:./mlruns
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib  # noqa: F401 — kept for backward-compat artifact reading
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from wwtp.config import DEFAULT_INPUT_COLS, VALID_TARGETS, Settings
from wwtp.data_prep import PreparedData, prepare
from wwtp.logging_cfg import get_logger
from wwtp.model import (
    EvalMetrics,
    TrainConfig,
    compile_network,
    define_network,
    evaluate_network,
    fit_network,
    make_predictions,
)
from wwtp.predictor import log_wwtp_model

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Plotting helpers (paper figures 7-18)
# ---------------------------------------------------------------------------


def _plot_loss_curve(history: object, out_path: Path, title: str) -> None:
    """Save a train/val loss curve to *out_path*."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="train")  # type: ignore[attr-defined]
    ax.plot(history.history["val_loss"], label="val")  # type: ignore[attr-defined]
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MAE)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    target_name: str,
) -> None:
    """Save a prediction-vs-actual line plot to *out_path*."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Actual", marker="o", markersize=2, linewidth=0.7)
    ax.plot(y_pred, label="Predicted", linewidth=0.9)
    ax.set_xlabel("Measuring period")
    ax.set_ylabel(f"Effluent {target_name}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-architecture run
# ---------------------------------------------------------------------------


def run_one_architecture(
    cfg: TrainConfig,
    data: PreparedData,
    artifact_dir: Path,
    experiment_name: str,
) -> dict:
    """Execute the 5-step pipeline as one MLflow run.

    Args:
        cfg: Hyper-parameter configuration for this run.
        data: Pre-processed data (train / val / test splits + scalers).
        artifact_dir: Local directory for temporary plot / CSV files.
        experiment_name: MLflow experiment name (for logging context only).

    Returns:
        Dict summarising the run: ``run_id``, ``run_name``, ``architecture``,
        ``metrics``, and ``train_seconds``.
    """
    arch = cfg.architecture.upper()
    run_name = f"{arch}_e{cfg.epochs}_b{cfg.batch_size}_lb{data.lookback}"
    logger.info("Starting run: %s", run_name)

    with mlflow.start_run(run_name=run_name) as run:
        # ── Log hyper-parameters ─────────────────────────────────────────────
        mlflow.log_params(
            {
                "architecture": cfg.architecture,
                "units": cfg.units,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "optimizer": cfg.optimizer,
                "loss": cfg.loss,
                "lookback": data.lookback,
                "n_features": data.n_features,
                "target": data.target_name,
                "n_train": len(data.y_train),
                "n_val": len(data.y_val),
                "n_test": len(data.y_test),
                "seed": cfg.seed,
            }
        )
        mlflow.log_dict({"feature_names": data.feature_names}, "feature_names.json")

        # ===== Step 1: Define ================================================
        model = define_network(cfg, data.lookback, data.n_features)
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        mlflow.log_text("\n".join(summary_lines), "model_summary.txt")

        # ===== Step 2: Compile ===============================================
        compile_network(model, cfg)

        # ===== Step 3: Fit ===================================================
        t0 = time.perf_counter()
        history = fit_network(
            model,
            data.X_train,
            data.y_train,
            data.X_val,  # ← validation set (not test set)
            data.y_val,
            cfg,
            verbose=0,
        )
        train_seconds = time.perf_counter() - t0
        mlflow.log_metric("train_seconds", train_seconds)

        for epoch, (tr, vl) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
            mlflow.log_metric("train_loss", tr, step=epoch)
            mlflow.log_metric("val_loss", vl, step=epoch)

        loss_plot = artifact_dir / f"{run_name}_loss.png"
        _plot_loss_curve(history, loss_plot, f"{arch} model — train/val loss")
        mlflow.log_artifact(str(loss_plot))

        # ===== Step 4: Evaluate =============================================
        metrics, y_true, y_pred = evaluate_network(
            model, data.X_test, data.y_test, data.target_scaler
        )
        mlflow.log_metrics(metrics.as_dict())

        pred_plot = artifact_dir / f"{run_name}_prediction.png"
        _plot_predictions(
            y_true,
            y_pred,
            pred_plot,
            f"Prediction of Effluent {data.target_name} ({arch})",
            data.target_name,
        )
        mlflow.log_artifact(str(pred_plot))

        # ===== Step 5: Log model + scalers (bundled) ========================
        preds = make_predictions(model, data.X_test, data.target_scaler)

        preds_csv = artifact_dir / f"{run_name}_test_predictions.csv"
        pd.DataFrame(
            {
                "timestamp": data.test_index,
                "y_true": y_true,
                "y_pred": preds,
            }
        ).to_csv(preds_csv, index=False)
        mlflow.log_artifact(str(preds_csv))

        signature = infer_signature(data.X_test[:5], model.predict(data.X_test[:5], verbose=0))

        # Scalers are bundled inside the pyfunc artifact — no separate joblib
        log_wwtp_model(
            keras_model=model,
            scalers={
                "feature_scaler": data.feature_scaler,
                "target_scaler": data.target_scaler,
            },
            artifact_path="model",
            signature=signature,
            input_example=data.X_test[:1],
        )

        logger.info(
            "[%s] RMSE=%.4f  MAE=%.4f  R²=%.4f  time=%.1fs",
            run_name,
            metrics.rmse,
            metrics.mae,
            metrics.r2,
            train_seconds,
        )

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "architecture": cfg.architecture,
            "metrics": metrics.as_dict(),
            "train_seconds": train_seconds,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and orchestrate training for both architectures."""
    settings = Settings()

    parser = argparse.ArgumentParser(
        description="Train WWTP effluent RNN/LSTM and register the best model in MLflow."
    )
    parser.add_argument("--data-path", required=True, help="Path to the online dataset CSV")
    parser.add_argument(
        "--target",
        default=settings.data.target_col,
        choices=VALID_TARGETS,
        help="Which output column to predict",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=settings.model.lookback,  # ← default is 8, not 1
        help="Number of past timesteps per input window",
    )
    parser.add_argument("--epochs", type=int, default=settings.model.epochs)
    parser.add_argument("--batch-size", type=int, default=settings.model.batch_size)
    parser.add_argument("--experiment", default=settings.experiment_name)
    parser.add_argument("--registered-model", default=settings.registered_model)
    parser.add_argument("--tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--artifact-dir", default=str(settings.artifact_dir))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare data once, reuse for both architectures ──────────────────────
    data = prepare(
        csv_path=args.data_path,
        input_cols=list(DEFAULT_INPUT_COLS),
        target_col=args.target,
        lookback=args.lookback,
        test_size=settings.data.test_size,
        val_size=settings.data.val_size,
    )
    logger.info(
        "Target: %s   X_train=%s  X_val=%s  X_test=%s",
        args.target,
        data.X_train.shape,
        data.X_val.shape,
        data.X_test.shape,
    )

    results = []
    for arch in ("rnn", "lstm"):
        cfg = TrainConfig(
            architecture=arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append(run_one_architecture(cfg, data, artifact_dir, args.experiment))

    # ── Register the best run by RMSE ────────────────────────────────────────
    best = min(results, key=lambda r: r["metrics"]["rmse"])
    logger.info("Best model: %s (RMSE=%.4f)", best["run_name"], best["metrics"]["rmse"])

    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=args.registered_model)
    logger.info("Registered '%s' version %s", args.registered_model, registered.version)

    summary = {
        "results": results,
        "best_run_id": best["run_id"],
        "best_run_name": best["run_name"],
        "registered_model": args.registered_model,
        "registered_version": registered.version,
    }
    summary_path = artifact_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
