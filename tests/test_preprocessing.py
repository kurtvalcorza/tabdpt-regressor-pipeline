import json

import numpy as np
import pandas as pd
import pytest

from tabdpt_regressor_pipeline import TabDPTRegressionPipeline, TabularFeatureEncoder


def test_train_fitted_mixed_encoder_handles_unknown_and_missing():
    train = pd.DataFrame({"age": [20, 30, 40], "region": ["north", "south", None]})
    encoder = TabularFeatureEncoder().fit(train)
    transformed = encoder.transform(pd.DataFrame({"age": [50, None], "region": ["east", None]}))
    assert transformed.shape == (2, 2)
    assert transformed[0, 1] == 2.0
    assert transformed[1, 1] == 3.0
    assert np.isnan(transformed[1, 0])


def test_encoder_rejects_schema_drift():
    encoder = TabularFeatureEncoder().fit(pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(ValueError, match="schema mismatch"):
        encoder.transform(pd.DataFrame({"b": [1, 2]}))


def test_encoder_state_round_trip_preserves_numeric_like_categorical_semantics(tmp_path):
    train = pd.DataFrame({
        "code": pd.Series(["1", "2", "3"], dtype="object"),
        "value": [1.5, 2.5, 3.5],
    })
    encoder = TabularFeatureEncoder().fit(train)
    assert encoder.numeric_columns == {"value"}
    assert "code" in encoder.category_maps

    state = json.loads(json.dumps(encoder.to_state()))
    csv_path = tmp_path / "context.csv"
    train.to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path)
    assert pd.api.types.is_numeric_dtype(reloaded["code"])

    restored = TabularFeatureEncoder.from_state(state)
    np.testing.assert_allclose(encoder.transform(train), restored.transform(reloaded))
    assert restored.to_state() == state


def test_pipeline_exports_reloadable_preprocessing_state():
    pipeline = TabDPTRegressionPipeline()
    pipeline.target_column = "target"
    pipeline.drop_columns_ = ["id"]
    pipeline.feature_encoder.fit(pd.DataFrame({"code": pd.Series(["1", "2"], dtype="object")}))
    state = pipeline.export_preprocessing_state()
    assert state["targetColumn"] == "target"
    assert state["dropColumns"] == ["id"]
    restored = TabularFeatureEncoder.from_state(state["encoder"])
    assert restored.category_maps == pipeline.feature_encoder.category_maps


def test_pipeline_rejects_unconfigured_extras_but_allows_configured_drop_columns():
    pipeline = TabDPTRegressionPipeline()
    pipeline.drop_columns_ = ["id"]
    pipeline.feature_encoder.fit(pd.DataFrame({"a": [1, 2]}))
    accepted = pipeline._feature_frame(pd.DataFrame({"id": [9], "a": [3]}))
    assert list(accepted.columns) == ["a"]
    with pytest.raises(ValueError, match=r"extra=\['rogue'\]"):
        pipeline._feature_frame(pd.DataFrame({"a": [3], "rogue": [1]}))


def test_evaluate_rejects_constant_target_before_model_execution():
    pipeline = TabDPTRegressionPipeline()
    pipeline.estimator = object()
    pipeline.target_column = "target"
    with pytest.raises(ValueError, match="must not be constant"):
        pipeline.evaluate(pd.DataFrame({"x": [1, 2], "target": [5.0, 5.0]}))
