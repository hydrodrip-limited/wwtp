"""
Train and deploy WWTP effluent prediction model with MLflow.

This script runs the paper's 5-step pipeline (define -> compile -> fit ->
evaluate -> predict) for both Simple RNN and LSTM architectures, logs each run
to MLflow with metrics/artifacts/model, and registers the best variant
into the MLflow Model Registry.

Usage:
    python train_mlflow.py --data-path ../train_val_test_online_dataset.csv

After it completes, inspect runs with:
    mlflow ui --backend-store-uri file:./mlruns
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from data_prep import DEFAULT_INPUT_COLS, DEFAULT_TARGET_COL, VALID_TARGETS, prepare
from model import (
    TrainConfig,
    compile_network,
    define_network,
    evaluate_network,
    fit_network,
    make_predictions,
)


def plot_loss_curve(history, out_path: Path, title: str) -> None:
    """Recreate Figures 17/18 from the paper."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MAE)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    target_name: str,
) -> None:
    """Recreate Figures 7-16 from the paper."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Original Value", marker="o", markersize=2, linewidth=0.7)
    ax.plot(y_pred, label="Predicted value", linewidth=0.9)
    ax.set_xlabel("Measuring Period")
    ax.set_ylabel(f"Effluent {target_name}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_one_architecture(
    cfg: TrainConfig,
    data,
    artifact_dir: Path,
    experiment_name: str,
) -> dict:
    """Execute the 5-step pipeline as one MLflow run. Returns a result summary."""
    arch = cfg.architecture.upper()
    run_name = f"{arch}_e{cfg.epochs}_b{cfg.batch_size}_lb{data.lookback}"

    with mlflow.start_run(run_name=run_name) as run:
        # ---- Log config + dataset shape ----
        mlflow.log_params({
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
            "n_test": len(data.y_test),
            "seed": cfg.seed,
        })
        mlflow.log_dict({"feature_names": data.feature_names}, "feature_names.json")

        # ===== Step 1: Define Network =====
        model = define_network(cfg, data.lookback, data.n_features)
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        mlflow.log_text("\n".join(summary_lines), "model_summary.txt")

        # ===== Step 2: Compile Network =====
        compile_network(model, cfg)

        # ===== Step 3: Fit Network =====
        t0 = time.time()
        history = fit_network(
            model, data.X_train, data.y_train, data.X_test, data.y_test, cfg, verbose=0
        )
        train_seconds = time.time() - t0
        mlflow.log_metric("train_seconds", train_seconds)

        # Log per-epoch loss curves (so MLflow plots them)
        for epoch, (tr, vl) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
            mlflow.log_metric("train_loss", tr, step=epoch)
            mlflow.log_metric("val_loss", vl, step=epoch)

        loss_plot = artifact_dir / f"{run_name}_loss.png"
        plot_loss_curve(history, loss_plot, f"{arch} model — train/test loss")
        mlflow.log_artifact(str(loss_plot))

        # ===== Step 4: Evaluate Network =====
        metrics, y_true, y_pred = evaluate_network(
            model, data.X_test, data.y_test, data.target_scaler
        )
        mlflow.log_metrics(metrics.as_dict())

        pred_plot = artifact_dir / f"{run_name}_prediction.png"
        plot_predictions(
            y_true, y_pred, pred_plot,
            f"Prediction of Effluent {data.target_name} ({arch})",
            data.target_name,
        )
        mlflow.log_artifact(str(pred_plot))

        # ===== Step 5: Make Predictions (and log the model) =====
        preds = make_predictions(model, data.X_test, data.target_scaler)

        # Persist predictions as a CSV artifact for downstream consumers
        preds_csv = artifact_dir / f"{run_name}_test_predictions.csv"
        pd.DataFrame({
            "timestamp": data.test_index,
            "y_true": y_true,
            "y_pred": preds,
        }).to_csv(preds_csv, index=False)
        mlflow.log_artifact(str(preds_csv))

        # Build a signature from a real input/output sample — required for clean deploy
        signature = infer_signature(data.X_test[:5], model.predict(data.X_test[:5], verbose=0))

        # Log model with signature + input example so serving works out of the box
        mlflow.tensorflow.log_model(
            model,
            name="model",
            signature=signature,
            input_example=data.X_test[:1],
        )

        # Also log the fitted scalers — needed at inference time
        import joblib
        scaler_path = artifact_dir / f"{run_name}_scalers.joblib"
        joblib.dump(
            {"feature_scaler": data.feature_scaler, "target_scaler": data.target_scaler},
            scaler_path,
        )
        mlflow.log_artifact(str(scaler_path))

        print(f"[{run_name}] RMSE={metrics.rmse:.4f}  MAE={metrics.mae:.4f}  "
              f"R2={metrics.r2:.4f}  train_time={train_seconds:.1f}s")

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "architecture": cfg.architecture,
            "metrics": metrics.as_dict(),
            "train_seconds": train_seconds,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the online dataset CSV")
    parser.add_argument("--target", default=DEFAULT_TARGET_COL, choices=VALID_TARGETS,
                        help="Which output column to predict (per the dataset's data dictionary)")
    parser.add_argument("--lookback", type=int, default=1,
                        help="Number of past timesteps per input window (paper uses t-1)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--experiment", default="wwtp_effluent_prediction")
    parser.add_argument("--registered-model", default="wwtp_effluent_predictor")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--artifact-dir", default="./artifacts")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Prepare data once, reuse for both architectures ----------
    data = prepare(
        csv_path=args.data_path,
        input_cols=DEFAULT_INPUT_COLS,
        target_col=args.target,
        lookback=args.lookback,
        test_size=0.2,
    )
    print(f"Target: {args.target}   X_train={data.X_train.shape}  X_test={data.X_test.shape}")

    results = []
    for arch in ("rnn", "lstm"):
        cfg = TrainConfig(
            architecture=arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append(run_one_architecture(cfg, data, artifact_dir, args.experiment))

    # ---------- Select the best run by RMSE and register it ----------
    best = min(results, key=lambda r: r["metrics"]["rmse"])
    print(f"\nBest model: {best['run_name']} (RMSE={best['metrics']['rmse']:.4f})")

    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=args.registered_model)
    print(f"Registered '{args.registered_model}' version {registered.version}")

    # ---------- Write a compact summary for downstream automation ----------
    summary_path = artifact_dir / "training_summary.json"
    summary_path.write_text(json.dumps({
        "results": results,
        "best_run_id": best["run_id"],
        "best_architecture": best["architecture"],
        "registered_model": args.registered_model,
        "registered_version": registered.version,
        "model_uri": model_uri,
    }, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
