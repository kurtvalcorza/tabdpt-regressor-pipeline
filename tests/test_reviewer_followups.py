import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tabdpt_regressor_pipeline.dimer_runtime import (
    DimerRuntimeConfig,
    _dataset_limits,
    prepare_dimer_frames,
    run_dimer_job,
)
from tabdpt_regressor_pipeline.pipeline import _resolve_use_flash


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400"])
def test_dataset_limits_reject_non_finite_compression_ratio(monkeypatch, value):
    monkeypatch.setenv("DIMER_MAX_COMPRESSION_RATIO", value)
    with pytest.raises(ValueError, match="must be finite"):
        _dataset_limits()


def test_dataset_limits_preserve_numeric_and_positive_validation(monkeypatch):
    monkeypatch.setenv("DIMER_MAX_ARCHIVE_BYTES", "abc")
    with pytest.raises(ValueError, match="must be numeric"):
        _dataset_limits()
    monkeypatch.delenv("DIMER_MAX_ARCHIVE_BYTES")

    monkeypatch.setenv("DIMER_MAX_DATASET_FILES", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _dataset_limits()
    monkeypatch.delenv("DIMER_MAX_DATASET_FILES")

    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "-5")
    with pytest.raises(ValueError, match="must be positive"):
        _dataset_limits()


def _fake_torch(available: bool, capability: tuple[int, int]):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda device=None: capability,
    )
    return SimpleNamespace(cuda=cuda)


def test_flash_auto_detection_disables_t4_and_enables_ampere(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, (7, 5)))
    assert _resolve_use_flash(None, None) is False

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, (8, 0)))
    assert _resolve_use_flash(None, "cuda") is True


def test_flash_auto_detection_fails_safe_and_respects_explicit_override(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, (0, 0)))
    assert _resolve_use_flash(None, None) is False
    assert _resolve_use_flash(None, "cpu") is False
    assert _resolve_use_flash(True, "cpu") is True
    assert _resolve_use_flash(False, "cuda") is False


def test_constant_supplied_validation_target_fails_before_model_work():
    train = pd.DataFrame({"x": range(1000), "target": [float(i) for i in range(1000)]})
    val = pd.DataFrame({"x": [1, 2], "target": [5.0, 5.0]})
    with pytest.raises(ValueError, match="validation target must not be constant"):
        prepare_dimer_frames(train, val, DimerRuntimeConfig())


def test_post_split_frames_are_revalidated_for_constant_target():
    frame = pd.DataFrame({"x": range(1000), "target": [1.0] * 999 + [2.0]})
    config = DimerRuntimeConfig(validation_split=0.2, seed=3)
    with pytest.raises(ValueError, match="target must not be constant"):
        prepare_dimer_frames(frame, None, config)


def test_artifact_drop_columns_use_fitted_preprocessing_state(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    dataset.mkdir()
    pd.DataFrame({"id": [1, 2, 3, 4], "x": [1, 2, 3, 4], "target": [1.0, 2.0, 3.0, 4.0]}).to_csv(
        dataset / "train.csv", index=False
    )
    pd.DataFrame({"id": [5, 6], "x": [5, 6], "target": [5.0, 6.0]}).to_csv(
        dataset / "val.csv", index=False
    )

    class FakePipeline:
        def __init__(self, **kwargs):
            self.drop_columns_ = []

        def fit(self, frame, target_column="target", drop_columns=None):
            self.drop_columns_ = [c for c in (drop_columns or []) if c != target_column]
            return self

        def evaluate(self, frame, **kwargs):
            return {"mae": 0.0, "rmse": 0.0, "r2": 1.0}

        def export_preprocessing_state(self):
            return {
                "schemaVersion": 1,
                "targetColumn": "target",
                "dropColumns": list(self.drop_columns_),
                "encoder": {"schemaVersion": 1, "featureColumns": ["x"], "numericColumns": ["x"], "categoryMaps": {}, "categoricalEncoding": {}},
            }

    import tabdpt_regressor_pipeline.dimer_runtime as runtime

    monkeypatch.setattr(runtime, "TabDPTRegressionPipeline", FakePipeline)
    monkeypatch.setenv("DIMER_DATASET_DIR", str(dataset))
    monkeypatch.setenv("DIMER_OUTPUT_DIR", str(output))
    monkeypatch.setenv(
        "DIMER_PREPROCESSING_ARGS_JSON",
        json.dumps({"target_column": "target", "drop_columns": "id,target"}),
    )

    run_dimer_job()
    artifact = json.loads((output / "artifacts" / "artifact.json").read_text())
    assert artifact["dropColumns"] == ["id"]
    assert artifact["preprocessing"]["dropColumns"] == ["id"]
