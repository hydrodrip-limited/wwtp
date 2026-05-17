"""Abstract base class for WWTP simulation backends (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SimulationBackend(ABC):
    """Common interface for all WWTP simulation implementations.

    Concrete sub-classes must implement :meth:`run` and return a
    :class:`pandas.DataFrame` with the columns listed in
    :attr:`REQUIRED_COLUMNS`.

    This pattern (Strategy) lets the calling code — the Streamlit app, CLI
    scripts, and tests — switch between the full ASM1 Julia simulation
    (:class:`~wwtp.simulation.peepypoo.PeePyPooSimulation`) and the
    pure-Python scipy fallback
    (:class:`~wwtp.simulation.synthetic.SyntheticSimulation`) without any
    ``if/else`` branches at the call site.
    """

    #: Columns that every backend must return in the output DataFrame.
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "time_h",
        "Q_inf",
        "Q_air_1",
        "Q_air_2",
        "Q_air_3",
        "Q_air_4",
        "Q_air_5",
        "Temp",
        "NH4_simulated",
    )

    @abstractmethod
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
        """Run the simulation and return a tidy DataFrame.

        Args:
            duration_h: Total simulation duration in hours.
            q_inf_base: Baseline influent flow rate (m³/h).
            q_air_base: Baseline aeration flow rate per zone (m³/h).
            temp_c: Wastewater temperature (°C).
            q_variation_pct: Amplitude of diurnal flow variation as a
                percentage of the baseline (0–100).
            n_steps: Number of output time points.
            v_m3: Reactor volume in m³.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with columns :attr:`REQUIRED_COLUMNS`.
        """

    @classmethod
    def validate_output(cls, df: pd.DataFrame) -> None:
        """Assert that *df* contains all required columns.

        Args:
            df: Simulation output to validate.

        Raises:
            ValueError: If any required column is missing.
        """
        missing = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Simulation output is missing required columns: {missing}")
