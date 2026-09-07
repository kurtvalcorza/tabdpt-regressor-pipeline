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


def test_pipeline_rejects_unconfigured_extras_but_allows_configured_drop_columns():
    pipeline = TabDPTRegressionPipeline()
    pipeline.drop_columns_ = ["id"]
    pipeline.feature_encoder.fit(pd.DataFrame({"a": [1, 2]}))
    accepted = pipeline._feature_frame(pd.DataFrame({"id": [9], "a": [3]}))
    assert list(accepted.columns) == ["a"]
    with pytest.raises(ValueError, match="extra=\['rogue'\]"):
        pipeline._feature_frame(pd.DataFrame({"a": [3], "rogue": [1]}))
