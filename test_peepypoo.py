"""
Standalone PeePyPoo integration test.

Run this BEFORE starting the full docker stack to verify peepypoo is installed
correctly and the simulation pipeline produces sensible output.

Usage
-----
    # Step 1 — install peepypoo and resolve Julia packages (one-time, ~5-15 min)
    pip install peepypoo
    python -c "import juliapkg; juliapkg.resolve()"

    # Step 2 — run this test
    python test_peepypoo.py

    # Step 3 — optional: also test the synthetic fallback
    python test_peepypoo.py --backend synthetic

What it checks
--------------
1. peepypoo imports without error
2. A short simulation (2 h, 20 steps) completes successfully
3. NH4 (S_NH) state is accessible in the solution
4. Output DataFrame has the correct columns and value ranges
5. Prints a sample of the output so you can sanity-check the numbers
"""
from __future__ import annotations

import argparse
import sys


def check_import() -> str:
    """Returns backend name or exits with a clear error."""
    try:
        import peepypoo  # noqa: F401
        return "peepypoo"
    except ImportError:
        print(
            "\n[FAIL] peepypoo is not installed.\n"
            "Install it with:\n"
            "    pip install git+https://gitlab.com/datinfo/PeePyPoo\n"
            "    python -c 'import juliapkg; juliapkg.resolve()'\n"
            "Or just use Docker: docker compose --profile peepypoo up\n"
        )
        return "missing"


def run_test(backend: str) -> None:
    print(f"\n{'='*60}")
    print(f"  PeePyPoo integration test  (backend={backend})")
    print(f"{'='*60}\n")

    # ── 1. Import check ──────────────────────────────────────────────────────
    if backend == "peepypoo":
        detected = check_import()
        if detected == "missing":
            sys.exit(1)
        print("[OK] peepypoo imported successfully")
        from peepypoo_sim import run_simulation as sim_fn
    else:
        from peepypoo_sim import run_synthetic_simulation as sim_fn
        print("[OK] using synthetic scipy ODE backend")

    # ── 2. Run a short simulation ────────────────────────────────────────────
    print("\nRunning 2-hour simulation (20 steps)…")
    try:
        df = sim_fn(
            duration_h=2.0,
            q_inf_base=3200.0,
            q_air_base=1400.0,
            temp_c=14.0,
            q_variation_pct=20.0,
            n_steps=20,
            seed=0,
        )
    except Exception as exc:
        print(f"\n[FAIL] Simulation raised an exception:\n  {exc}")
        sys.exit(1)

    print(f"[OK] Simulation completed — {len(df)} rows returned")

    # ── 3. Column check ──────────────────────────────────────────────────────
    required = ["time_h", "Q_inf", "Q_air_1", "Q_air_2", "Q_air_3",
                "Q_air_4", "Q_air_5", "Temp", "NH4_simulated"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"[FAIL] Missing columns: {missing_cols}")
        sys.exit(1)
    print(f"[OK] All required columns present: {required}")

    # ── 4. Value range checks ────────────────────────────────────────────────
    checks = {
        "time_h        in [0, 2.1]":  df["time_h"].between(0, 2.1).all(),
        "Q_inf         in [500, 8000]": df["Q_inf"].between(500, 8000).all(),
        "Q_air_1       in [200, 10000]": df["Q_air_1"].between(200, 10000).all(),
        "Temp          == 14.0":        (df["Temp"] == 14.0).all(),
        "NH4_simulated >= 0":           (df["NH4_simulated"] >= 0).all(),
        "NH4_simulated <= 200":         (df["NH4_simulated"] <= 200).all(),
    }
    all_passed = True
    for label, passed in checks.items():
        status = "[OK]  " if passed else "[WARN]"
        print(f"  {status} {label}")
        if not passed:
            all_passed = False

    # ── 5. Print sample output ───────────────────────────────────────────────
    print("\nFirst 5 rows of simulation output:")
    print(df[["time_h", "Q_inf", "Q_air_1", "Temp", "NH4_simulated"]].head(5).to_string(index=False))

    print(f"\nNH4_simulated stats:")
    print(f"  min={df['NH4_simulated'].min():.3f}  "
          f"max={df['NH4_simulated'].max():.3f}  "
          f"mean={df['NH4_simulated'].mean():.3f}  mg/L")

    # ── 6. Rolling-window check (mimics what the Streamlit app does) ─────────
    import numpy as np
    LOOKBACK = 8
    if len(df) >= LOOKBACK + 1:
        INPUT_COLS = ["Q_inf", "Q_air_1", "Q_air_2", "Q_air_3", "Q_air_4", "Q_air_5", "Temp"]
        arr = df[INPUT_COLS].values.astype("float32")
        n_windows = len(arr) - LOOKBACK
        windows = np.stack([arr[i: i + LOOKBACK] for i in range(n_windows)])
        print(f"\n[OK] Rolling-window construction: {n_windows} windows of shape {windows.shape[1:]}")
        print(      "      Ready to feed into RNN model via /invocations API")
    else:
        print(f"\n[WARN] Only {len(df)} rows — need at least {LOOKBACK + 1} for windowing. "
              "Increase --steps or --duration.")

    print(f"\n{'='*60}")
    print("  All checks passed." if all_passed else "  Some range warnings — review output above.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PeePyPoo integration test")
    p.add_argument(
        "--backend",
        choices=["peepypoo", "synthetic"],
        default="peepypoo",
        help="Which simulation backend to test (default: peepypoo)",
    )
    args = p.parse_args()
    run_test(args.backend)