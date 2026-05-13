"""
WWTP NH4 Prediction UI

Three modes:
  Tab 1 — Dummy Data    : auto-generated realistic sensor windows
  Tab 2 — Manual Payload: editable 8×7 table of raw sensor readings
  Tab 3 — PeePyPoo      : run a mechanistic WWTP simulation, roll prediction
                           windows across the output, compare vs ground truth
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Config ──────────────────────────────────────────────────────────────────
SCALERS_PATH = Path(
    os.getenv(
        "SCALERS_PATH",
        "wwtp_rnn_mlflow/wwtp_rnn_bundle/mlruns/"
        "248838729491991513/1ea0da5dd20549dea78db1770a219acb/"
        "artifacts/RNN_e50_b100_lb8_scalers.joblib",
    )
)

INPUT_COLS = ["Q_inf", "Q_air_1", "Q_air_2", "Q_air_3", "Q_air_4", "Q_air_5", "Temp"]
LOOKBACK = 8

FEATURE_DEFAULTS = {
    "Q_inf":   3200.0,
    "Q_air_1": 1400.0,
    "Q_air_2": 1400.0,
    "Q_air_3": 1400.0,
    "Q_air_4": 1400.0,
    "Q_air_5": 1400.0,
    "Temp":    14.0,
}
FEATURE_RANGES = {
    "Q_inf":   (1000, 6000),
    "Q_air_1": (400,  3000),
    "Q_air_2": (400,  3000),
    "Q_air_3": (400,  3000),
    "Q_air_4": (400,  3000),
    "Q_air_5": (400,  3000),
    "Temp":    (8,    25),
}


# ─── Shared helpers ───────────────────────────────────────────────────────────
@st.cache_resource
def load_scalers() -> dict:
    return joblib.load(SCALERS_PATH)


def api_ping(url: str) -> bool:
    try:
        r = requests.get(url.replace("/invocations", "/ping"), timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_api(scaled_batch: np.ndarray, url: str) -> np.ndarray:
    resp = requests.post(url, json={"inputs": scaled_batch.tolist()}, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    return np.array(result.get("predictions", result), dtype="float32").ravel()


def predict_nh4(raw_windows: np.ndarray, scalers: dict, api_url: str) -> np.ndarray:
    """(n, lookback, n_features) raw sensor values → NH4 mg/L (n,)."""
    n, lb, nf = raw_windows.shape
    scaled = scalers["feature_scaler"].transform(
        raw_windows.reshape(-1, nf)
    ).reshape(n, lb, nf).astype("float32")
    raw_preds = call_api(scaled, api_url)
    return scalers["target_scaler"].inverse_transform(
        raw_preds.reshape(-1, 1)
    ).ravel()


def windows_from_df(df: pd.DataFrame, lookback: int = LOOKBACK) -> np.ndarray:
    """Build (n, lookback, 7) windows from a DataFrame with INPUT_COLS."""
    arr = df[INPUT_COLS].values.astype("float32")
    n = len(arr)
    if n < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {n}")
    return np.stack([arr[i - lookback : i] for i in range(lookback, n)])


# ─── Page layout ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="WWTP NH₄ Predictor", layout="wide")
st.title("WWTP Effluent NH₄ — RNN Prediction")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input(
        "Model API URL",
        value=os.getenv("API_URL", "http://localhost:5000/invocations"),
    )
    online = api_ping(api_url)
    st.markdown(f"**API status:** {'🟢 online' if online else '🔴 offline'}")
    if not online:
        st.warning("Start the container: `docker compose up`")
    st.divider()
    st.caption("Model · wwtp_effluent_predictor")
    st.caption("Target · NH₄   Lookback · 8 × 15 min")

if not SCALERS_PATH.exists():
    st.error(
        f"Scalers not found at `{SCALERS_PATH}`.  "
        "Set the `SCALERS_PATH` environment variable to the correct `.joblib` path."
    )
    st.stop()

scalers = load_scalers()

tab1, tab2, tab3 = st.tabs(
    [" Data", "Manual Payload", " PeePyPoo Simulation"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 —  DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Auto-generate synthetic sensor windows and predict NH₄")
    st.caption(
        "Produces a smooth random walk around typical Tilburg WWTP operating values, "
        "builds rolling 8-step windows, and calls the prediction API for each."
    )

    ctrl, chart_area = st.columns([1, 2])

    with ctrl:
        n_preds   = st.slider("Number of predictions", 1, 100, 20)
        noise_pct = st.slider("Noise (% of operating range)", 0, 40, 8)
        seed      = st.number_input("Random seed", value=42, step=1)

        if st.button("Generate & Predict", type="primary", disabled=not online):
            rng     = np.random.default_rng(int(seed))
            noise   = noise_pct / 100.0
            lo      = np.array([FEATURE_RANGES[c][0] for c in INPUT_COLS])
            hi      = np.array([FEATURE_RANGES[c][1] for c in INPUT_COLS])
            span    = hi - lo
            base    = np.array([FEATURE_DEFAULTS[c] for c in INPUT_COLS])

            # Smooth random walk: each prediction is one forward step
            windows = []
            for _ in range(n_preds):
                base += rng.normal(0, span * noise * 0.15, 7)
                base  = np.clip(base, lo, hi)
                # 8-step lookback window around current base
                win = base + rng.normal(0, span * noise * 0.05, (8, 7))
                win = np.clip(win, lo, hi)
                windows.append(win.astype("float32"))

            try:
                preds = predict_nh4(np.stack(windows), scalers, api_url)
                st.session_state["dummy_df"] = pd.DataFrame(
                    {"Step": range(1, n_preds + 1), "NH₄ (mg/L)": preds.round(4)}
                )
            except Exception as exc:
                st.error(f"API error: {exc}")

    if "dummy_df" in st.session_state:
        df = st.session_state["dummy_df"]
        with chart_area:
            fig = go.Figure(go.Scatter(
                x=df["Step"], y=df["NH₄ (mg/L)"],
                mode="lines+markers", line_color="#1f77b4",
            ))
            fig.update_layout(
                xaxis_title="Prediction step",
                yaxis_title="Predicted NH₄ (mg/L)",
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL PAYLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Enter 8 timesteps of raw sensor readings manually")
    st.caption(
        "Each row = one 15-minute reading (lookback window = 8 rows). "
        "Enter real physical units — MinMax scaling is applied automatically before the API call."
    )

    st.markdown(
        "| Column | Unit | Typical range |\n"
        "|---|---|---|\n"
        "| Q_inf | m³/h influent flow | 1 000 – 6 000 |\n"
        "| Q_air_1 … Q_air_5 | m³/h aeration per zone | 400 – 3 000 |\n"
        "| Temp | °C end-of-zone temperature | 8 – 25 |"
    )

    default_df = pd.DataFrame(
        {col: [FEATURE_DEFAULTS[col]] * 8 for col in INPUT_COLS}
    )

    edited = st.data_editor(
        default_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.2f")
            for col in INPUT_COLS
        },
    )

    if st.button("Predict NH₄", type="primary", disabled=not online, key="manual_predict"):
        raw = edited[INPUT_COLS].values.astype("float32")[np.newaxis]  # (1, 8, 7)
        try:
            pred = predict_nh4(raw, scalers, api_url)
            st.metric("Predicted NH₄", f"{pred[0]:.4f} mg/L")

            with st.expander("Raw API payload (scaled [0,1] values sent to /invocations)"):
                scaled_display = scalers["feature_scaler"].transform(
                    raw[0].reshape(-1, 7)
                ).reshape(1, 8, 7)
                st.json({"inputs": scaled_display.tolist()})
        except Exception as exc:
            st.error(f"API error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PEEPYPOO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Mechanistic WWTP simulation → rolling RNN predictions")
    st.caption(
        "Runs a WWTP simulation that produces Q_inf, Q_air, and Temp time-series. "
        "The RNN model predicts NH₄ for each rolling 8-step window. "
        "Simulated NH₄ from the ODE is shown as ground truth."
    )

    try:
        import peepypoo  # noqa: F401 — only succeeds if Julia+peepypoo are installed
        from peepypoo_sim import run_simulation
        sim_backend = "peepypoo"
    except ImportError:
        from peepypoo_sim import run_synthetic_simulation as run_simulation
        sim_backend = "synthetic"

    if sim_backend == "synthetic":
        st.info(
            "**PeePyPoo / Julia not detected** — the built-in 3-state nitrification ODE "
            "is being used as a fallback. To enable the full ASM1 simulation:\n"
            "```\n"
            "pip install peepypoo\n"
            "python -c 'import juliapkg; juliapkg.resolve()'\n"
            "```\n"
            "Then restart this app."
        )
    else:
        st.success("PeePyPoo (Julia/ASM1) available.")

    load_tab, run_tab = st.tabs(["📂 Load exported CSV", "▶️ Configure & Run"])

    with load_tab:
        st.markdown(
            "Upload a CSV produced by `python peepypoo_sim.py --output sim.csv`, "
            "or any CSV/TSV with columns `Q_inf, Q_air_1..5, Temp` "
            "(optional `timestamp` or `time_h` column)."
        )
        uploaded = st.file_uploader("Upload sensor CSV", type=["csv", "tsv"])
        if uploaded:
            sep = "\t" if uploaded.name.endswith(".tsv") else ","
            udf = pd.read_csv(uploaded, sep=sep)
            missing = [c for c in INPUT_COLS if c not in udf.columns]
            if missing:
                st.error(f"Missing columns in upload: {missing}")
            else:
                st.session_state["sim_df"] = udf
                st.success(f"Loaded {len(udf)} rows.")

    with run_tab:
        c1, c2 = st.columns(2)
        with c1:
            duration_h  = st.slider("Duration (h)", 2, 48, 12)
            q_inf_base  = st.slider("Base Q_inf (m³/h)", 1000, 6000, 3200, step=100)
            q_air_base  = st.slider("Base Q_air per zone (m³/h)", 400, 3000, 1400, step=100)
        with c2:
            temp_c      = st.slider("Temperature (°C)", 8.0, 25.0, 14.0, step=0.5)
            q_var_pct   = st.slider("Flow variation (%)", 0, 50, 20)
            n_sim_steps = st.slider("Simulation timesteps", 20, 200, 60)

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running WWTP simulation…"):
                try:
                    sim_df = run_simulation(
                        duration_h=duration_h,
                        q_inf_base=q_inf_base,
                        q_air_base=q_air_base,
                        temp_c=temp_c,
                        q_variation_pct=q_var_pct,
                        n_steps=n_sim_steps,
                    )
                    st.session_state["sim_df"] = sim_df
                    st.success(f"Simulation done — {len(sim_df)} timesteps.")
                except Exception as exc:
                    st.error(f"Simulation failed: {exc}")

    # ── Results panel ──────────────────────────────────────────────────────
    if "sim_df" in st.session_state:
        sim_df = st.session_state["sim_df"]
        st.divider()
        st.subheader("Simulation outputs")

        x_axis = sim_df.get("time_h", pd.Series(range(len(sim_df)))).tolist()

        with st.expander("Input time-series (Q_inf, Q_air zones, Temp)", expanded=True):
            fig_in = go.Figure()
            for col in ["Q_inf", "Q_air_1", "Q_air_2", "Q_air_3", "Q_air_4", "Q_air_5"]:
                if col in sim_df.columns:
                    fig_in.add_trace(go.Scatter(x=x_axis, y=sim_df[col], mode="lines", name=col))
            fig_in.update_layout(
                xaxis_title="Time (h)" if "time_h" in sim_df.columns else "Step",
                yaxis_title="Flow (m³/h)",
                margin=dict(t=10, b=40),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_in, use_container_width=True)

        st.divider()
        st.subheader("RNN predictions on simulation data")

        if len(sim_df) < LOOKBACK + 1:
            st.warning(f"Need at least {LOOKBACK + 1} rows — simulation only has {len(sim_df)}.")
        elif not online:
            st.warning("API is offline — start the container first.")
        else:
            if st.button("Predict NH₄ across simulation", type="primary"):
                try:
                    windows = windows_from_df(sim_df)
                    preds   = predict_nh4(windows, scalers, api_url)
                    x_pred  = x_axis[LOOKBACK:]

                    fig_nh4 = go.Figure()
                    fig_nh4.add_trace(go.Scatter(
                        x=x_pred, y=preds,
                        mode="lines+markers", name="RNN predicted NH₄",
                        line_color="#1f77b4",
                    ))
                    if "NH4_simulated" in sim_df.columns:
                        fig_nh4.add_trace(go.Scatter(
                            x=x_pred,
                            y=sim_df["NH4_simulated"].iloc[LOOKBACK:].values,
                            mode="lines", name="Simulated NH₄ (ODE ground truth)",
                            line=dict(color="#ff7f0e", dash="dash"),
                        ))
                    fig_nh4.update_layout(
                        xaxis_title="Time (h)" if "time_h" in sim_df.columns else "Step",
                        yaxis_title="NH₄ (mg/L)",
                        margin=dict(t=10, b=40),
                        legend=dict(x=0, y=1),
                    )
                    st.plotly_chart(fig_nh4, use_container_width=True)

                    result_df = pd.DataFrame({
                        "time": x_pred,
                        "predicted_NH4_mg_per_l": preds.round(4),
                    })
                    if "NH4_simulated" in sim_df.columns:
                        result_df["simulated_NH4_mg_per_l"] = (
                            sim_df["NH4_simulated"].iloc[LOOKBACK:].values.round(4)
                        )
                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode()
                    st.download_button(
                        "Download predictions CSV",
                        data=csv_bytes,
                        file_name="wwtp_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Prediction error: {exc}")
