"""Full PeePyPoo / Julia ASM1 simulation backend.

Prerequisites
-------------
    pip install git+https://gitlab.com/datinfo/PeePyPoo@main
    python -c 'import juliapkg; juliapkg.resolve()'   # first run ~5-40 min

If ``peepypoo`` is not installed this module can still be imported safely;
the :class:`PeePyPooSimulation` class raises :class:`ImportError` at
instantiation time, which the factory in
:mod:`wwtp.simulation` converts into a graceful fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from wwtp.simulation.base import SimulationBackend
from wwtp.logging_cfg import get_logger

logger = get_logger(__name__)

_ZONE_FRACTIONS = np.array([0.19, 0.20, 0.21, 0.20, 0.20])


class PeePyPooSimulation(SimulationBackend):
    """Full ASM1 simulation via the PeePyPoo Julia package.

    Args:
        Raises ImportError at instantiation if ``peepypoo`` is not installed.

    Note on state names
    -------------------
    PeePyPoo uses Julia ModelingToolkit namespacing: ``reactor₊S_NH``
    (subscript-plus, U+208A).  If a newer version uses a different separator
    (e.g. ``.``), adjust the ``sol["reactor₊S_NH"]`` line accordingly.
    """

    def __init__(self) -> None:
        try:
            import peepypoo  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PeePyPoo is not installed.  Install it with:\n"
                "    pip install git+https://gitlab.com/datinfo/PeePyPoo@main\n"
                "    python -c 'import juliapkg; juliapkg.resolve()'\n"
                "Or use the synthetic fallback (SyntheticSimulation) instead."
            ) from exc

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
        """Run the full ASM1 simulation via PeePyPoo.

        Args:
            duration_h: Total simulation duration in hours.
            q_inf_base: Baseline influent flow rate (m³/h).
            q_air_base: Baseline aeration flow rate per zone (m³/h).
            temp_c: Wastewater temperature (°C).
            q_variation_pct: Amplitude of diurnal variation as % of baseline.
            n_steps: Number of output time points.
            v_m3: Reactor volume in m³.
            seed: Random seed for noise terms.

        Returns:
            DataFrame with columns defined in
            :attr:`~wwtp.simulation.base.SimulationBackend.REQUIRED_COLUMNS`.

        Raises:
            RuntimeError: If the Julia ODE solver fails to converge.
        """
        from peepypoo.ProcessElements import CSTR, ASM1, Aeration, Influent
        from peepypoo import ODESystem, ODEProblem, Solvers
        from peepypoo.Interpolations import Interpolation, LinearInterpolation

        logger.info(
            "Running PeePyPoo ASM1 simulation (duration=%.1fh, steps=%d)",
            duration_h,
            n_steps,
        )

        rng = np.random.default_rng(seed)
        t_s = np.linspace(0.0, duration_h * 3600.0, n_steps)
        noise = q_variation_pct / 100.0

        q_inf_vals = q_inf_base * (1 + noise * np.sin(2 * np.pi * t_s / 86400)) + rng.normal(
            0, q_inf_base * noise * 0.08, n_steps
        )
        q_air_vals = q_air_base * (1 + noise * np.sin(2 * np.pi * t_s / 43200 + 1.0)) + rng.normal(
            0, q_air_base * noise * 0.08, n_steps
        )
        q_inf_vals = np.clip(q_inf_vals, 500.0, 8000.0)
        q_air_vals = np.clip(q_air_vals, 200.0, 10_000.0)

        # k_La from Q_air via linear proxy; alpha scales by reactor volume.
        alpha = 0.003 / (v_m3 / 1_000.0)
        kla_vals = (q_air_vals * 5 * alpha).tolist()

        q_inf_src = Interpolation(
            LinearInterpolation(q_inf_vals.tolist(), t_s.tolist()), name="q_inf_src"
        )
        kla_src = Interpolation(LinearInterpolation(kla_vals, t_s.tolist()), name="kLa_src")

        influent = Influent(sources=q_inf_src, name="influent")
        aeration = Aeration(k_La=kla_src, name="aeration")
        reactor = CSTR(
            V=v_m3 * 1_000,  # PeePyPoo uses litres internally
            processes=[ASM1(temperature=temp_c), aeration],
            name="reactor",
        )

        sys = ODESystem([influent, reactor])
        simp = sys.structural_simplify()
        problem = ODEProblem(simp, tspan=(0.0, duration_h * 3600.0))
        sol = problem.solve(
            algorithm=Solvers.Tsit5(),
            saveat=duration_h * 3600.0 / n_steps,
            abstol=1e-6,
            reltol=1e-3,
        )

        if not sol.successful:
            raise RuntimeError("PeePyPoo ODE solve did not converge.")

        t_out = np.array(sol.t)
        nh4 = np.array(sol["reactor₊S_NH"])

        qi_fn = interp1d(t_s, q_inf_vals, fill_value="extrapolate")
        qa_fn = interp1d(t_s, q_air_vals, fill_value="extrapolate")
        qi_out = qi_fn(t_out)
        qa_out = qa_fn(t_out)

        zf = _ZONE_FRACTIONS
        mu = float(zf.mean())

        df = pd.DataFrame(
            {
                "time_h": t_out / 3600.0,
                "Q_inf": qi_out,
                "Q_air_1": qa_out * zf[0] / mu,
                "Q_air_2": qa_out * zf[1] / mu,
                "Q_air_3": qa_out * zf[2] / mu,
                "Q_air_4": qa_out * zf[3] / mu,
                "Q_air_5": qa_out * zf[4] / mu,
                "Temp": float(temp_c),
                "NH4_simulated": nh4,
            }
        )
        self.validate_output(df)
        logger.info("PeePyPoo simulation complete — %d rows", len(df))
        return df
