"""Integration tests for WWTP simulation backends.

Both backends must produce a DataFrame with the correct column contract.
PeePyPoo tests are skipped when Julia is not installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from wwtp.simulation.base import SimulationBackend
from wwtp.simulation.synthetic import SyntheticSimulation

pytestmark = pytest.mark.integration


# ─── Shared run parameters ────────────────────────────────────────────────────

RUN_KWARGS = {
    "duration_h": 2,
    "q_inf_base": 3200.0,
    "q_air_base": 1400.0,
    "temp_c": 14.0,
    "q_variation_pct": 20,
    "n_steps": 24,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _skip_peepypoo() -> bool:
    """Return True when PeePyPoo / Julia are not available."""
    try:
        importlib.import_module("peepypoo")
        return False
    except ImportError:
        return True


def _assert_output_contract(df: pd.DataFrame, backend_name: str) -> None:
    """Assert that the output DataFrame respects the SimulationBackend contract."""
    missing = [c for c in SimulationBackend.REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"{backend_name} output is missing required columns: {missing}"
    assert len(df) > 0, f"{backend_name} returned an empty DataFrame"
    # No NaN/Inf in any required column
    for col in SimulationBackend.REQUIRED_COLUMNS:
        assert df[col].notna().all(), f"{backend_name}: NaN in column '{col}'"
        assert np.isfinite(df[col].values).all(), f"{backend_name}: Inf in column '{col}'"


# ─── SyntheticSimulation ─────────────────────────────────────────────────────


class TestSyntheticSimulation:
    def test_output_has_required_columns(self) -> None:
        sim = SyntheticSimulation()
        df = sim.run(**RUN_KWARGS)
        _assert_output_contract(df, "SyntheticSimulation")

    def test_output_length_matches_n_steps(self) -> None:
        sim = SyntheticSimulation()
        df = sim.run(**RUN_KWARGS)
        assert len(df) == RUN_KWARGS["n_steps"]

    def test_nh4_simulated_column_present(self) -> None:
        """Synthetic backend should expose NH4_simulated for ground truth comparison."""
        sim = SyntheticSimulation()
        df = sim.run(**RUN_KWARGS)
        assert "NH4_simulated" in df.columns

    def test_q_inf_values_within_physical_range(self) -> None:
        sim = SyntheticSimulation()
        df = sim.run(**RUN_KWARGS)
        assert df["Q_inf"].between(0, 20000).all()

    def test_temperature_column_equals_input(self) -> None:
        sim = SyntheticSimulation()
        df = sim.run(**RUN_KWARGS)
        assert (df["Temp"] == RUN_KWARGS["temp_c"]).all()


# ─── PeePyPoo backend (skipped when Julia absent) ────────────────────────────


@pytest.mark.skipif(
    _skip_peepypoo(), reason="PeePyPoo / Julia not installed — skipping ASM1 tests."
)
class TestPeePyPooSimulation:
    def test_output_has_required_columns(self) -> None:
        from wwtp.simulation.peepypoo import PeePyPooSimulation

        sim = PeePyPooSimulation()
        df = sim.run(**RUN_KWARGS)
        _assert_output_contract(df, "PeePyPooSimulation")

    def test_nh4_values_are_non_negative(self) -> None:
        from wwtp.simulation.peepypoo import PeePyPooSimulation

        sim = PeePyPooSimulation()
        df = sim.run(**RUN_KWARGS)
        assert (df["NH4_simulated"] >= 0).all()


# ─── get_backend factory ─────────────────────────────────────────────────────


def test_get_backend_auto_returns_simulation_backend() -> None:
    from wwtp.simulation import get_backend

    b = get_backend("auto")
    assert isinstance(b, SimulationBackend)


def test_get_backend_synthetic_returns_correct_type() -> None:
    from wwtp.simulation import get_backend

    b = get_backend("synthetic")
    assert isinstance(b, SyntheticSimulation)


def test_get_backend_unknown_name_raises() -> None:
    from wwtp.simulation import get_backend

    with pytest.raises(ValueError, match="Unknown simulation backend"):
        get_backend("unknown_xyz")
