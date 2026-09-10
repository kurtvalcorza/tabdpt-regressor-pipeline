import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabdpt_regressor_pipeline import TabDPTRegressionPipeline
from tabdpt_regressor_pipeline.artifact import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    EXPECTED_BASE_MODEL,
    export_artifact_bundle,
    load_verified_artifact,
    validate_artifact_bundle,
)


def _export_without_model_download(tmp_path):
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "category": ["01", "02", "03"],
            "target": [10.0, 20.0, 30.0],
        }
    )
    pipe = TabDPTRegressionPipeline(use_flash=False)
    pipe.target_column = "target"
    pipe.drop_columns_ = []
    encoded = pipe.feature_encoder.fit_transform(frame.drop(columns=["target"]))
    pipe.estimator = SimpleNamespace(
        V=None,
        imputer=SimpleNamespace(statistics_=np.nanmean(encoded, axis=0)),
        scaler=SimpleNamespace(mean_=np.nanmean(encoded, axis=0), scale_=np.ones(encoded.shape[1])),
    )
    return export_artifact_bundle(pipe, frame, tmp_path / "artifact")


def test_exported_artifact_validates_without_loading_model(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest, context_path = validate_artifact_bundle(manifest_path)

    assert manifest["format"] == ARTIFACT_FORMAT
    assert manifest["formatVersion"] == ARTIFACT_FORMAT_VERSION
    assert manifest["baseModel"] == EXPECTED_BASE_MODEL
    assert manifest["trainingContext"]["path"] == "training_context.parquet"
    assert manifest["trainingContext"]["size"] == context_path.stat().st_size
    assert context_path.is_file()


def test_established_v3_runtime_manifest_remains_accepted(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("formatVersion")
    manifest.pop("artifactSemantics")
    manifest["trainingContext"].pop("size")
    manifest_path.write_text(json.dumps(manifest))

    validated, _ = validate_artifact_bundle(manifest_path)
    assert validated["format"] == ARTIFACT_FORMAT
    assert validated["baseModel"] == EXPECTED_BASE_MODEL


def test_explicit_artifact_requires_fitted_upstream_preprocessing_state(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["preprocessing"].pop("upstream")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="missing fitted upstream preprocessing state"):
        validate_artifact_bundle(manifest_path)


def test_verified_reload_restores_upstream_state_without_fit(tmp_path, monkeypatch):
    manifest_path = _export_without_model_download(tmp_path)

    class NoRefitEstimator:
        def __init__(self, *args, **kwargs):
            self.device = kwargs.get("device") or "cpu"
            self.missing_indicators = False
            self.normalizer = "standard"
            self.feature_reduction = "pca"
            self.max_features = 128
            self.V = None

        def fit(self, X, y):
            raise AssertionError("verified artifact reload must not call estimator.fit")

        def predict(self, X, **kwargs):
            transformed = self.scaler.transform(self.imputer.transform(X))
            return transformed.sum(axis=1)

    import tabdpt_regressor_pipeline.artifact as artifact_mod

    monkeypatch.setitem(sys.modules, "tabdpt", SimpleNamespace(TabDPTRegressor=NoRefitEstimator))
    monkeypatch.setattr(artifact_mod, "resolve_tabdpt_weights", lambda *args, **kwargs: Path("fake.safetensors"))

    restored = load_verified_artifact(manifest_path, compile_model=False, use_flash=False)
    assert restored.preprocessing_restored_ is True
    np.testing.assert_allclose(restored.estimator.imputer.statistics_, [2.0, 1.0])
    np.testing.assert_allclose(restored.estimator.scaler.mean_, [2.0, 1.0])
    np.testing.assert_allclose(restored.estimator.scaler.scale_, [1.0, 1.0])

    query = pd.DataFrame({"numeric": [4.0], "category": ["02"]})
    prediction = restored.predict(query, n_ensembles=1, context_size=128, batch_size=1, seed=42)
    assert prediction.shape == (1,)


def test_artifact_rejects_base_model_revision_mismatch(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["baseModel"]["revision"] = "mutable-or-wrong-revision"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"baseModel\.revision mismatch"):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_unsupported_format_version(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["formatVersion"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="formatVersion"):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_context_path_traversal(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["trainingContext"]["path"] = "../training_context.parquet"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsafe artifact member path"):
        validate_artifact_bundle(manifest_path)


def test_artifact_rejects_context_size_or_digest_mismatch(tmp_path):
    manifest_path = _export_without_model_download(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["trainingContext"]["size"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Training context size mismatch"):
        validate_artifact_bundle(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    context_path = manifest_path.parent / "training_context.parquet"
    manifest["trainingContext"]["size"] = context_path.stat().st_size
    manifest["trainingContext"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Training context SHA-256 mismatch"):
        validate_artifact_bundle(manifest_path)
