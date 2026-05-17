"""Pure-Python scipy ODE simulation — always available, no Julia required."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from wwtp.simulation.base import SimulationBackend
from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)


class SyntheticSimulation(SimulationBackend):
    """Simplified 3-state nitrification ODE — no Julia required.

    States
    ------
    S_NH : ammonium nitrogen [g N/m³]
    S_O  : dissolved oxygen  [g O₂/m³]
    X_BA : autotrophic biomass [g COD/m³]

    Forcing
    -------
    Q_inf and Q_air follow a diurnal sinusoid plus a random walk.
    Temperature is constant; its effect is captured via the Arrhenius
    correction ``θ^(T - 20)``.

    The five aeration zones receive slightly different fractions of the total
    Q_air, matching the observed asymmetry in the Tilburg dataset.
    """

    # ASM1-inspired kinetic parameters at 20 °C
    _MU_A_MAX: float = 0.8 / 24  # /h — maximum autotrophic growth rate
    _K_NH: float = 1.0  # g N/m³ — half-saturation for NH4
    _K_O_A: float = 0.4  # g O₂/m³ — half-saturation for O₂
    _Y_A: float = 0.24  # g COD/g N — autotrophic yield
    _B_A: float = 0.05 / 24  # /h — autotrophic decay rate
    _S_O_SAT: float = 10.0  # g O₂/m³ — O₂ saturation concentration
    _ALPHA: float = 0.003  # k_La / (Q_air_total / V)  →  /h per m³/h/m³
    _ZONE_FRACTIONS: np.ndarray = np.array([0.19, 0.20, 0.21, 0.20, 0.20])

    def run(
        self,
        *,
        duration_h: float = 12.0,
        q_inf_base: float = 3200.0,
        q_air_base: float = 1400.0,
        temp_c: float = 14.0,
        q_variation_pct: float = 20.0,
        n_steps: int = 50,
        v_m3: float = 50_000.0,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Run the 3-state ODE simulation.

        Args:
            duration_h: Total simulation duration in hours.
            q_inf_base: Baseline influent flow rate (m³/h).
            q_air_base: Baseline aeration flow rate per zone (m³/h).
            temp_c: Constant wastewater temperature (°C).
            q_variation_pct: Amplitude of diurnal variation as % of baseline.
            n_steps: Number of output time points.
            v_m3: Reactor volume in m³.
            seed: Random seed for the noise terms.

        Returns:
            DataFrame with columns defined in
            :attr:`~wwtp.simulation.base.SimulationBackend.REQUIRED_COLUMNS`.

        Raises:
            RuntimeError: If the ODE solver fails to converge.
        """
        logger.info("Running synthetic simulation (duration=%.1fh, steps=%d)", duration_h, n_steps)
        rng = np.random.default_rng(seed)
        t_eval = np.linspace(0.0, duration_h, n_steps)
        noise = q_variation_pct / 100.0
        theta_t = 1.07 ** (temp_c - 20.0)

        # ── Build diurnal forcing profiles ──────────────────────────────────
        q_inf_vals = q_inf_base * (1 + noise * np.sin(2 * np.pi * t_eval / 24)) + rng.normal(
            0, q_inf_base * noise * 0.08, n_steps
        )
        q_air_vals = q_air_base * (1 + noise * np.sin(2 * np.pi * t_eval / 12 + 1.0)) + rng.normal(
            0, q_air_base * noise * 0.08, n_steps
        )
        q_inf_vals = np.clip(q_inf_vals, 500.0, 8000.0)
        q_air_vals = np.clip(q_air_vals, 200.0, 10_000.0)

        q_inf_fn = interp1d(t_eval, q_inf_vals, kind="linear", fill_value="extrapolate")
        q_air_fn = interp1d(t_eval, q_air_vals, kind="linear", fill_value="extrapolate")

        def ode(t: float, y: list[float]) -> list[float]:
            s_nh = max(y[0], 0.0)
            s_o = max(y[1], 0.0)
            x_ba = max(y[2], 0.0)

            q = float(q_inf_fn(t))
            q_air = float(q_air_fn(t)) * 5  # sum of five zones
            k_la = self._ALPHA * q_air / v_m3

            mu_a = (
                self._MU_A_MAX
                * theta_t
                * (s_nh / (self._K_NH + s_nh))
                * (s_o / (self._K_O_A + s_o))
            )
            r_a = mu_a * x_ba

            d_s_nh = (q / v_m3) * (30.0 - s_nh) - r_a / self._Y_A
            d_s_o = (
                (q / v_m3) * (0.5 - s_o)
                + k_la * (self._S_O_SAT - s_o)
                - (4.57 - self._Y_A) / self._Y_A * r_a
            )
            d_x_ba = r_a - self._B_A * x_ba - (q / v_m3) * x_ba
            return [d_s_nh, d_s_o, d_x_ba]

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

        # ── Split total Q_air across 5 zones ─────────────────────────────────
        zf = self._ZONE_FRACTIONS
        mu = float(zf.mean())

        df = pd.DataFrame(
            {
                "time_h": sol.t,
                "Q_inf": q_inf_vals,
                "Q_air_1": q_air_vals * zf[0] / mu,
                "Q_air_2": q_air_vals * zf[1] / mu,
                "Q_air_3": q_air_vals * zf[2] / mu,
                "Q_air_4": q_air_vals * zf[3] / mu,
                "Q_air_5": q_air_vals * zf[4] / mu,
                "Temp": float(temp_c),
                "NH4_simulated": sol.y[0],
            }
        )
        self.validate_output(df)
        logger.info("Synthetic simulation complete — %d rows", len(df))
        return df
