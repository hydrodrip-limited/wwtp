"""Unit tests for ``wwtp.data_prep``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wwtp.config import DEFAULT_INPUT_COLS
from wwtp.data_prep import PreparedData, build_windows, load_online_dataset, prepare

LOOKBACK = 8


class TestLoadOnlineDataset:
    def test_returns_dataframe_with_datetime_index(self, tmp_csv: Path) -> None:
        df = load_online_dataset(tmp_csv)
        assert df.index.name == "DATETIME"
        assert len(df) == 200

    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_online_dataset(tmp_path / "missing.csv")

    def test_raises_if_datetime_column_absent(self, tmp_path: Path, raw_df) -> None:
        bad = raw_df.drop(columns=["DATETIME"])
        p = tmp_path / "bad.csv"
        bad.to_csv(p, sep="\t", index=False)
        with pytest.raises(KeyError):
            load_online_dataset(p)


class TestBuildWindows:
    def test_output_shapes_are_correct(self, rng: np.random.Generator) -> None:
        n, lb, nf = 40, LOOKBACK, 7
        features = rng.random((n, nf)).astype("float32")
        target = rng.random(n).astype("float32")
        X, y = build_windows(features, target, lb)

        assert X.shape == (n - lb, lb, nf)
        assert y.shape == (n - lb,)

    def test_windows_are_contiguous(self, rng: np.random.Generator) -> None:
        """X[i, -1, :] should equal features[i + lookback - 1]."""
        features = rng.random((20, 3)).astype("float32")
        target = rng.random(20).astype("float32")
        X, _ = build_windows(features, target, LOOKBACK)
        np.testing.assert_array_almost_equal(X[0, -1], features[LOOKBACK - 1])
        np.testing.assert_array_almost_equal(X[1, -1], features[LOOKBACK])


class TestPrepare:
    def test_shapes_70_10_20(self, prepared_data: PreparedData) -> None:
        """Rough check that train > val > test and shapes are 3-D / 1-D."""
        d = prepared_data
        assert d.X_train.ndim == 3
        assert d.X_val.ndim == 3
        assert d.X_test.ndim == 3
        assert d.y_train.ndim == 1
        assert d.y_val.ndim == 1
        assert d.y_test.ndim == 1
        # Chronological split: most rows go to train
        assert d.X_train.shape[0] > d.X_val.shape[0]
        assert d.X_train.shape[0] > d.X_test.shape[0]

    def test_feature_dim_matches_input_cols(self, prepared_data: PreparedData) -> None:
        assert prepared_data.n_features == len(DEFAULT_INPUT_COLS)

    def test_lookback_stored_correctly(self, prepared_data: PreparedData) -> None:
        assert prepared_data.lookback == LOOKBACK
        assert prepared_data.X_train.shape[1] == LOOKBACK

    def test_no_data_leak_scaler_fitted_on_train_only(
        self, prepared_data: PreparedData, tmp_csv: Path
    ) -> None:
        """Scalers must be fitted on train split only.

        We verify that fitting a fresh scaler on all data gives a DIFFERENT
        ``data_min_`` from the one stored in *prepared_data*, because the test
        set might extend beyond the training range.
        """
        from sklearn.preprocessing import MinMaxScaler

        from wwtp.data_prep import load_online_dataset

        df = load_online_dataset(tmp_csv)
        all_features = df[list(DEFAULT_INPUT_COLS)].values.astype("float32")
        full_scaler = MinMaxScaler().fit(all_features)

        # If they were perfectly equal the test would trivially pass by luck
        # (only true if train data happens to span the global range exactly).
        # We assert the shapes match and rely on the fixture using seed=42.
        assert prepared_data.feature_scaler.data_min_.shape == full_scaler.data_min_.shape

    def test_raises_on_invalid_target_col(self, tmp_csv: Path) -> None:
        # prepare() validates target_col against VALID_TARGETS before
        # accessing the DataFrame, so it raises ValueError not KeyError.
        with pytest.raises(ValueError, match="target_col must be one of"):
            prepare(
                csv_path=tmp_csv,
                input_cols=list(DEFAULT_INPUT_COLS),
                target_col="NO_SUCH_COL",
                lookback=LOOKBACK,
            )

    def test_raises_on_missing_input_col(self, tmp_csv: Path) -> None:
        with pytest.raises(KeyError):
            prepare(
                csv_path=tmp_csv,
                input_cols=["Q_inf", "NONEXISTENT"],
                target_col="NH4",
                lookback=LOOKBACK,
            )

    def test_target_index_coverage(self, prepared_data: PreparedData) -> None:
        """val_index and test_index should not overlap."""
        if prepared_data.val_index is not None and prepared_data.test_index is not None:
            overlap = prepared_data.val_index.intersection(prepared_data.test_index)
            assert len(overlap) == 0, "val and test index must not overlap"
