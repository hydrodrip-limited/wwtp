"""
WWTP simulation for feeding the RNN prediction API.

Two backends
------------
run_simulation()           — full PeePyPoo / Julia ASM1 (requires `pip install peepypoo`)
run_synthetic_simulation() — pure-Python scipy fallback (always available, no Julia)

Both return a DataFrame with columns:
    time_h, Q_inf, Q_air_1, Q_air_2, Q_air_3, Q_air_4, Q_air_5, Temp, NH4_simulated

The Q_inf / Q_air / Temp columns plug directly into the RNN model (INPUT_COLS).
NH4_simulated is the ODE ground truth for visual comparison.

Standalone usage
----------------
    # auto-detect backend
    python peepypoo_sim.py --output sim.csv --duration 12 --q-inf 3200 --q-air 1400 --temp 14

    # force backends
    python peepypoo_sim.py --backend synthetic --output sim.csv
    python peepypoo_sim.py --backend peepypoo  --output sim.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


# ─── Synthetic fallback ───────────────────────────────────────────────────────

def run_synthetic_simulation(
    duration_h: float = 12.0,
    q_inf_base: float = 3200.0,
    q_air_base: float = 1400.0,
    temp_c: float = 14.0,
    q_variation_pct: float = 20.0,
    n_steps: int = 50,
    V_m3: float = 50_000.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simplified 3-state nitrification ODE — no Julia required.

    States
    ------
    S_NH : ammonium nitrogen [g N/m³]
    S_O  : dissolved oxygen  [g O2/m³]
    X_BA : autotrophic biomass [g COD/m³]

    Forcing
    -------
    Q_inf and Q_air follow a diurnal sinusoid + random walk.
    Temp is constant (temperature correction via Arrhenius θ).
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    rng    = np.random.default_rng(seed)
    t_eval = np.linspace(0.0, duration_h, n_steps)
    noise  = q_variation_pct / 100.0

    # Diurnal-ish profiles
    q_inf_vals = (
        q_inf_base * (1 + noise * np.sin(2 * np.pi * t_eval / 24))
        + rng.normal(0, q_inf_base * noise * 0.08, n_steps)
    )
    q_air_vals = (
        q_air_base * (1 + noise * np.sin(2 * np.pi * t_eval / 12 + 1.0))
        + rng.normal(0, q_air_base * noise * 0.08, n_steps)
    )
    q_inf_vals = np.clip(q_inf_vals, 500.0,  8000.0)
    q_air_vals = np.clip(q_air_vals, 200.0, 10000.0)

    q_inf_fn = interp1d(t_eval, q_inf_vals, kind="linear", fill_value="extrapolate")
    q_air_fn = interp1d(t_eval, q_air_vals, kind="linear", fill_value="extrapolate")

    # ASM1-inspired kinetic parameters at 20 °C
    mu_A_max  = 0.8 / 24   # /h
    K_NH      = 1.0        # g N/m³
    K_O_A     = 0.4        # g O2/m³
    Y_A       = 0.24       # g COD/g N
    b_A       = 0.05 / 24  # /h
    S_O_sat   = 10.0       # g O2/m³
    alpha     = 0.003      # k_La / (Q_air_total / V)  →  /h per (m³/h / m³)
    theta_T   = 1.07 ** (temp_c - 20.0)

    def ode(t: float, y: list) -> list:
        S_NH = max(y[0], 0.0)
        S_O  = max(y[1], 0.0)
        X_BA = max(y[2], 0.0)

        Q    = float(q_inf_fn(t))
        Qair = float(q_air_fn(t)) * 5  # five zones summed
        k_La = alpha * Qair / V_m3

        mu_A = mu_A_max * theta_T * (S_NH / (K_NH + S_NH)) * (S_O / (K_O_A + S_O))
        r_A  = mu_A * X_BA

        dS_NH = (Q / V_m3) * (30.0 - S_NH) - r_A / Y_A
        dS_O  = (Q / V_m3) * (0.5 - S_O) + k_La * (S_O_sat - S_O) \
                - (4.57 - Y_A) / Y_A * r_A
        dX_BA = r_A - b_A * X_BA - (Q / V_m3) * X_BA
        return [dS_NH, dS_O, dX_BA]

    sol = solve_ivp(
        ode,
        (0.0, duration_h),
        [10.0, 2.0, 200.0],
        t_eval=t_eval,
        method="RK45",
        max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Split total Q_air across 5 zones with slight variation
    zf = np.array([0.19, 0.20, 0.21, 0.20, 0.20])
    mu = zf.mean()

    return pd.DataFrame({
        "time_h":        sol.t,
        "Q_inf":         q_inf_vals,
        "Q_air_1":       q_air_vals * zf[0] / mu,
        "Q_air_2":       q_air_vals * zf[1] / mu,
        "Q_air_3":       q_air_vals * zf[2] / mu,
        "Q_air_4":       q_air_vals * zf[3] / mu,
        "Q_air_5":       q_air_vals * zf[4] / mu,
        "Temp":          temp_c,
        "NH4_simulated": sol.y[0],
    })


# ─── PeePyPoo / Julia backend ─────────────────────────────────────────────────

def run_simulation(
    duration_h: float = 12.0,
    q_inf_base: float = 3200.0,
    q_air_base: float = 1400.0,
    temp_c: float = 14.0,
    q_variation_pct: float = 20.0,
    n_steps: int = 50,
    V_m3: float = 50_000.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Full PeePyPoo ASM1 simulation.

    Prerequisites
    -------------
    pip install peepypoo
    python -c 'import juliapkg; juliapkg.resolve()'   # downloads Julia + packages (~5 min once)

    What it does
    ------------
    1. Builds Q_inf and Q_air time-series (diurnal sinusoid + noise).
    2. Creates a single CSTR with ASM1 kinetics and time-varying Aeration.
    3. Solves the ODE over [0, duration_h] hours using Tsit5.
    4. Returns Q_inf, Q_air_1..5, Temp, and NH4 (= S_NH state) as a DataFrame.

    Note on state names
    -------------------
    PeePyPoo uses Julia ModelingToolkit namespacing: `reactor₊S_NH` (₊ = U+208A).
    If your PeePyPoo version uses a different separator (e.g. `.`), adjust the
    sol["reactor₊S_NH"] lines below accordingly.
    """
    import numpy as _np
    from peepypoo.ProcessElements import CSTR, ASM1, Aeration, Influent
    from peepypoo import ODESystem, ODEProblem, Solvers
    from peepypoo.Interpolations import Interpolation, LinearInterpolation

    rng   = _np.random.default_rng(seed)
    t_s   = _np.linspace(0.0, duration_h * 3600.0, n_steps)
    noise = q_variation_pct / 100.0

    q_inf_vals = (
        q_inf_base * (1 + noise * _np.sin(2 * _np.pi * t_s / 86400))
        + rng.normal(0, q_inf_base * noise * 0.08, n_steps)
    )
    q_air_vals = (
        q_air_base * (1 + noise * _np.sin(2 * _np.pi * t_s / 43200 + 1.0))
        + rng.normal(0, q_air_base * noise * 0.08, n_steps)
    )
    q_inf_vals = _np.clip(q_inf_vals, 500.0,  8000.0)
    q_air_vals = _np.clip(q_air_vals, 200.0, 10000.0)

    # k_La from Q_air: linear proxy (alpha scales by reactor size)
    alpha     = 0.003 / (V_m3 / 1000.0)
    kLa_vals  = (q_air_vals * 5 * alpha).tolist()

    q_inf_src = Interpolation(
        LinearInterpolation(q_inf_vals.tolist(), t_s.tolist()), name="q_inf_src"
    )
    kLa_src   = Interpolation(
        LinearInterpolation(kLa_vals, t_s.tolist()), name="kLa_src"
    )

    influent = Influent(sources=q_inf_src, name="influent")
    aeration = Aeration(k_La=kLa_src, name="aeration")
    reactor  = CSTR(
        V=V_m3 * 1000,  # PeePyPoo uses litres internally
        processes=[ASM1(temperature=temp_c), aeration],
        name="reactor",
    )

    sys     = ODESystem([influent, reactor])
    simp    = sys.structural_simplify()
    problem = ODEProblem(simp, tspan=(0.0, duration_h * 3600.0))
    sol     = problem.solve(
        algorithm=Solvers.Tsit5(),
        saveat=duration_h * 3600.0 / n_steps,
        abstol=1e-6,
        reltol=1e-3,
    )

    if not sol.successful:
        raise RuntimeError("PeePyPoo ODE solve did not converge.")

    t_out  = _np.array(sol.t)
    nh4    = _np.array(sol["reactor₊S_NH"])

    # Interpolate Q_inf / Q_air back onto the saved time points
    from scipy.interpolate import interp1d
    qi_fn = interp1d(t_s, q_inf_vals, fill_value="extrapolate")
    qa_fn = interp1d(t_s, q_air_vals, fill_value="extrapolate")
    qi_out = qi_fn(t_out)
    qa_out = qa_fn(t_out)

    zf = _np.array([0.19, 0.20, 0.21, 0.20, 0.20])
    mu = zf.mean()

    return pd.DataFrame({
        "time_h":        t_out / 3600.0,
        "Q_inf":         qi_out,
        "Q_air_1":       qa_out * zf[0] / mu,
        "Q_air_2":       qa_out * zf[1] / mu,
        "Q_air_3":       qa_out * zf[2] / mu,
        "Q_air_4":       qa_out * zf[3] / mu,
        "Q_air_5":       qa_out * zf[4] / mu,
        "Temp":          temp_c,
        "NH4_simulated": nh4,
    })


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Run a WWTP simulation and export CSV.")
    p.add_argument("--output",    default="sim_data.csv", help="Output CSV path")
    p.add_argument("--duration",  type=float, default=12.0,   metavar="H",    help="Simulation hours")
    p.add_argument("--q-inf",     type=float, default=3200.0, metavar="M3H",  help="Base Q_inf m³/h")
    p.add_argument("--q-air",     type=float, default=1400.0, metavar="M3H",  help="Base Q_air per zone m³/h")
    p.add_argument("--temp",      type=float, default=14.0,   metavar="C",    help="Temperature °C")
    p.add_argument("--variation", type=float, default=20.0,   metavar="PCT",  help="Flow variation %%")
    p.add_argument("--steps",     type=int,   default=50,                     help="Output timesteps")
    p.add_argument("--backend",   choices=["auto", "peepypoo", "synthetic"],  default="auto")
    args = p.parse_args()

    if args.backend == "peepypoo":
        fn = run_simulation
    elif args.backend == "synthetic":
        fn = run_synthetic_simulation
    else:
        try:
            import peepypoo  # noqa: F401
            fn = run_simulation
            print("Backend: PeePyPoo (Julia/ASM1)")
        except ImportError:
            fn = run_synthetic_simulation
            print("Backend: synthetic (scipy ODE — install peepypoo for full ASM1)")

    df = fn(
        duration_h=args.duration,
        q_inf_base=args.q_inf,
        q_air_base=args.q_air,
        temp_c=args.temp,
        q_variation_pct=args.variation,
        n_steps=args.steps,
    )
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows → {args.output}")
    print(df[["time_h", "Q_inf", "Q_air_1", "Temp", "NH4_simulated"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
