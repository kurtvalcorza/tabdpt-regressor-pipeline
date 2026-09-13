"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21).

No weights, no model: the checks are the same private functions `fit` and `predict` route through,
exercised on small in-memory tables.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tabdpt_regressor_pipeline import (
    INPUT_SCHEMA,
    METRIC_IDS,
    MIN_DISTINCT_TARGETS,
    MODEL_ID,
    MODEL_REVISION,
    TabDPTRegressionPipeline,
    evaluation_report,
    training_mean_baseline,
    validate_inputs,
)


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [31, 45, np.nan, 52, 23, 38],
            "city": ["a", "b", None, "a", "c", "b"],
            "target": [1.5, 2.0, 0.5, 3.0, 1.0, 2.5],
        }
    )


def test_validate_inputs_fit_mode_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_table(), target_column="target", names=["sample"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["distinct_targets"] == [MIN_DISTINCT_TARGETS, None]
    assert manifest["target_column"] == "target"
    assert manifest["drop_columns"] == []
    (entry,) = manifest["inputs"]
    assert entry["id"] == "sample"
    assert entry["mode"] == "fit"
    assert entry["rows"] == 6
    assert entry["feature_columns"] == ["age", "city"]
    assert entry["numeric_columns"] == ["age"]
    assert entry["categorical_columns"] == ["city"]
    assert entry["missing_value_columns"] == {"age": 1, "city": 1}
    assert entry["target_summary"] == {"min": 0.5, "max": 3.0, "mean": 1.75, "distinct": 6}
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_drop_columns() -> None:
    manifest = validate_inputs(_table(), "target", ["city", "target", "city"])
    assert manifest["inputs"][0]["id"] == "table-0"
    assert manifest["drop_columns"] == ["city"]
    assert manifest["inputs"][0]["feature_columns"] == ["age"]


def test_validate_inputs_rejects_like_fit() -> None:
    pipe = TabDPTRegressionPipeline()
    duplicated = pd.concat([_table(), _table()[["age"]]], axis=1)
    for bad, message in (
        (duplicated, "Duplicate column names"),
        (_table().rename(columns={"target": "label"}), "Target column 'target' not found"),
        (_table().assign(target=["1", "x", "2", "3", "4", "5"]), "non-numeric values"),
        (_table().assign(target=[1.0, None, 2.0, 3.0, 4.0, 5.0]), "finite and non-missing"),
        (_table().assign(target=[1.0, np.inf, 2.0, 3.0, 4.0, 5.0]), "finite and non-missing"),
        (_table().assign(target=7.0), "must not be constant"),
        (_table()[["target"]], "At least one feature column"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inputs(bad, target_column="target")
        with pytest.raises(ValueError, match=message):
            pipe.fit(bad, target_column="target")
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_table(), names=["a", "b"])


def test_validate_inputs_inference_mode_rejects_like_predict() -> None:
    fitted = ["age", "city"]
    manifest = validate_inputs(_table().drop(columns=["target"]), None, feature_columns=fitted, names=["new"])
    (entry,) = manifest["inputs"]
    assert entry == {
        "id": "new",
        "mode": "inference",
        "rows": 6,
        "feature_columns": fitted,
        "missing_value_columns": {"age": 1, "city": 1},
    }
    assert manifest["target_column"] is None
    pipe = TabDPTRegressionPipeline()
    pipe.feature_encoder.feature_columns = fitted
    pipe.feature_encoder.is_fitted = True
    pipe.feature_encoder.numeric_columns = {"age"}
    pipe.feature_encoder.category_maps = {"city": {"a": 0, "b": 1, "c": 2}}
    wrong = _table().drop(columns=["target", "city"]).assign(extra=1)
    with pytest.raises(ValueError, match=r"missing=\['city'\], extra=\['extra'\]"):
        validate_inputs(wrong, None, feature_columns=fitted)
    with pytest.raises(ValueError, match=r"missing=\['city'\], extra=\['extra'\]"):
        pipe._feature_frame(wrong)
    with pytest.raises(ValueError, match="feature_columns is required"):
        validate_inputs(wrong, None)


def test_training_mean_baseline_matches_the_evaluate_metric_ids() -> None:
    baseline = training_mean_baseline([1.0, 2.0, 3.0], [2.0, 4.0])
    assert set(baseline) == set(METRIC_IDS)
    # constant prediction 2.0 vs [2, 4]: mae = 1, rmse = sqrt(2), r2 = 1 - 4/2 = -1
    assert baseline["mae"] == 1.0
    assert math.isclose(baseline["rmse"], math.sqrt(2.0))
    assert math.isclose(baseline["r2"], -1.0)
    with pytest.raises(ValueError, match="must not be constant"):
        training_mean_baseline([1.0, 2.0], [5.0, 5.0])
    with pytest.raises(ValueError, match="numeric and finite"):
        training_mean_baseline([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="numeric and finite"):
        training_mean_baseline([1.0, 2.0], ["x", 2.0])


def test_evaluation_report_not_measurable_without_metrics() -> None:
    report = evaluation_report(None, n_holdout=0, target_column="target", sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert "labelled holdout" in report["needs"] and "training_mean_baseline" in report["needs"]
    assert report["target_column"] == "target"
    assert report["sample_kind"] == "BYOD"
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_metrics_and_baseline() -> None:
    metrics = {"mae": 40.0, "rmse": 55.0, "r2": 0.45}
    baseline = {"mae": 65.0, "rmse": 78.0, "r2": -0.01}
    report = evaluation_report(metrics, baseline=baseline, n_holdout=10, sample_kind="sample", estimation="e")
    assert report["verdict"] == "sample-sanity"
    assert [m["id"] for m in report["metrics"]] == list(METRIC_IDS)
    assert {m["id"]: m["value"] for m in report["metrics"]} == metrics
    higher = {m["id"]: m["higher_is_better"] for m in report["metrics"]}
    assert higher == {"mae": False, "rmse": False, "r2": True}
    assert all(m["estimation"] == "e" for m in report["metrics"])
    assert report["metrics"][0]["units"] == "target units"
    assert report["baselines"] == [
        {"id": "training_mean", "metrics": [{"id": k, "value": baseline[k]} for k in METRIC_IDS]}
    ]
    assert report["n_holdout"] == 10
    assert "10 labelled holdout row(s)" in report["reason"]
    with pytest.raises(ValueError, match="unknown metric ids"):
        evaluation_report({"mape": 0.5})
