"""Unit tests for ``wwtp.simulation`` — base, synthetic backend, and factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wwtp.simulation.base import SimulationBackend
from wwtp.simulation.synthetic import SyntheticSimulation

# Minimal args shared across tests
_RUN = dict(
    duration_h=1,
    q_inf_base=3200.0,
    q_air_base=1400.0,
    temp_c=14.0,
    q_variation_pct=20.0,
    n_steps=12,
    seed=0,
)


# ─── SimulationBackend base class ────────────────────────────────────────────


class TestSimulationBackendBase:
    def test_required_columns_is_non_empty_tuple(self) -> None:
        assert isinstance(SimulationBackend.REQUIRED_COLUMNS, tuple)
        assert len(SimulationBackend.REQUIRED_COLUMNS) > 0

    def test_validate_output_passes_for_correct_df(self) -> None:
        df = pd.DataFrame({col: [1.0] for col in SimulationBackend.REQUIRED_COLUMNS})
        SimulationBackend.validate_output(df)  # must not raise

    def test_validate_output_raises_on_missing_column(self) -> None:
        cols = list(SimulationBackend.REQUIRED_COLUMNS)
        df = pd.DataFrame({c: [1.0] for c in cols[:-1]})  # drop last column
        with pytest.raises(ValueError, match="missing required columns"):
            SimulationBackend.validate_output(df)


# ─── SyntheticSimulation ─────────────────────────────────────────────────────


class TestSyntheticSimulationShape:
    def test_output_has_all_required_columns(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        missing = [c for c in SimulationBackend.REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_output_row_count_matches_n_steps(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        assert len(df) == _RUN["n_steps"]

    def test_time_column_starts_near_zero(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        assert df["time_h"].iloc[0] >= 0.0

    def test_time_column_ends_near_duration(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        assert df["time_h"].iloc[-1] <= _RUN["duration_h"] * 1.01


class TestSyntheticSimulationValues:
    def test_nh4_simulated_is_non_negative(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        assert (df["NH4_simulated"] >= 0).all(), "NH4 must be ≥ 0 mg/L"

    def test_temperature_column_matches_input(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        assert (df["Temp"] == _RUN["temp_c"]).all()

    def test_q_inf_within_physical_range(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        assert df["Q_inf"].between(0, 20_000).all()

    def test_q_air_zones_are_positive(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        for zone in ["Q_air_1", "Q_air_2", "Q_air_3", "Q_air_4", "Q_air_5"]:
            assert (df[zone] > 0).all(), f"{zone} must be > 0"

    def test_no_nan_or_inf_in_output(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        for col in SimulationBackend.REQUIRED_COLUMNS:
            assert df[col].notna().all(), f"NaN in {col}"
            assert np.isfinite(df[col].values).all(), f"Inf in {col}"

    def test_seeded_runs_are_reproducible(self) -> None:
        df1 = SyntheticSimulation().run(**_RUN)
        df2 = SyntheticSimulation().run(**_RUN)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_q_inf(self) -> None:
        df1 = SyntheticSimulation().run(**{**_RUN, "seed": 0})
        df2 = SyntheticSimulation().run(**{**_RUN, "seed": 99})
        assert not df1["Q_inf"].equals(df2["Q_inf"])

    def test_validate_output_does_not_raise(self) -> None:
        df = SyntheticSimulation().run(**_RUN)
        SimulationBackend.validate_output(df)  # must not raise


# ─── get_backend factory ─────────────────────────────────────────────────────


class TestGetBackendFactory:
    def test_auto_returns_simulation_backend(self) -> None:
        from wwtp.simulation import get_backend

        b = get_backend("auto")
        assert isinstance(b, SimulationBackend)

    def test_synthetic_returns_synthetic_simulation(self) -> None:
        from wwtp.simulation import get_backend

        b = get_backend("synthetic")
        assert isinstance(b, SyntheticSimulation)

    def test_unknown_name_raises_value_error(self) -> None:
        from wwtp.simulation import get_backend

        with pytest.raises(ValueError, match="Unknown simulation backend"):
            get_backend("not_a_backend")

    def test_peepypoo_raises_import_error_when_absent(self) -> None:
        """When PeePyPoo is not installed get_backend('peepypoo') must raise."""
        from wwtp.simulation import get_backend

        try:
            import peepypoo  # noqa: F401

            pytest.skip("PeePyPoo is installed — cannot test the ImportError path")
        except ImportError:
            with pytest.raises(ImportError):
                get_backend("peepypoo")
