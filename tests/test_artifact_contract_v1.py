import json

import pandas as pd
import pytest

from tabdpt_regressor_pipeline import TabDPTRegressionPipeline
from tabdpt_regressor_pipeline.artifact import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    EXPECTED_BASE_MODEL,
    export_artifact_bundle,
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
    pipe.feature_encoder.fit(frame.drop(columns=["target"]))
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
