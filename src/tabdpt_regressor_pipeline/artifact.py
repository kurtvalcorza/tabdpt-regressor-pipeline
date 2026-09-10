from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from .pipeline import (
    TABDPT_HF_REPO,
    TABDPT_HF_REVISION,
    TABDPT_UPSTREAM_CODE_COMMIT,
    TABDPT_WEIGHT_FILENAME,
    TABDPT_WEIGHT_SHA256,
    TabDPTRegressionPipeline,
)

ARTIFACT_FORMAT = "tabdpt-dimer-context-v3"
ARTIFACT_FORMAT_VERSION = 3
ARTIFACT_SEMANTICS = "support-context-plus-pinned-base-model"

EXPECTED_BASE_MODEL = {
    "repo": TABDPT_HF_REPO,
    "revision": TABDPT_HF_REVISION,
    "filename": TABDPT_WEIGHT_FILENAME,
    "sha256": TABDPT_WEIGHT_SHA256,
    "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_bundle_member(manifest_path: Path, relative_path: str) -> Path:
    if not relative_path or "\\" in relative_path:
        raise ValueError("Artifact member path must be a non-empty POSIX relative path")
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe artifact member path: {relative_path!r}")
    root = manifest_path.parent.resolve()
    resolved = (root / Path(*posix_path.parts)).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Artifact member escapes bundle root: {relative_path!r}")
    return resolved


def export_artifact_bundle(
    pipeline: TabDPTRegressionPipeline,
    training_context: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Export the reusable DIMER serving state for an in-context TabDPT regressor."""
    if pipeline.target_column is None or not pipeline.feature_encoder.is_fitted:
        raise RuntimeError("Pipeline must be conditioned before artifact export")
    if training_context.columns.duplicated().any():
        raise ValueError("Training context contains duplicate column names")
    if pipeline.target_column not in training_context.columns:
        raise ValueError(f"Training context is missing target column {pipeline.target_column!r}")

    expected_features = list(pipeline.feature_encoder.feature_columns)
    feature_frame = training_context.drop(
        columns=[pipeline.target_column, *pipeline.drop_columns_], errors="ignore"
    )
    if list(feature_frame.columns) != expected_features:
        raise ValueError(
            "Training context schema does not match the fitted serving schema; "
            f"expected={expected_features}, got={list(feature_frame.columns)}"
        )
    pipeline.feature_encoder.transform(feature_frame)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "training_context.parquet"
    training_context.to_parquet(context_path, index=False)

    preprocessing = pipeline.export_preprocessing_state()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "formatVersion": ARTIFACT_FORMAT_VERSION,
        "taskType": "tabular_regression",
        "artifactSemantics": ARTIFACT_SEMANTICS,
        "targetColumn": pipeline.target_column,
        "dropColumns": list(pipeline.drop_columns_),
        "preprocessing": preprocessing,
        "baseModel": dict(EXPECTED_BASE_MODEL),
        "trainingContext": {
            "path": context_path.name,
            "size": context_path.stat().st_size,
            "sha256": _sha256(context_path),
        },
    }
    manifest_path = output / "artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def validate_artifact_bundle(artifact_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Validate artifact identity and context integrity before model-state reconstruction.

    Existing DIMER v3 manifests encode their version in ``format`` and may omit the newer
    explicit ``formatVersion``, ``artifactSemantics``, and context ``size`` fields. When those
    additive fields are present they are validated; their absence does not invalidate a v3
    artifact produced by the repository's established runtime contract.
    """
    manifest_path = Path(artifact_path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"Artifact manifest must be a regular file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Artifact manifest must contain a JSON object")
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"Unsupported artifact format: {manifest.get('format')!r}")

    explicit_version = manifest.get("formatVersion")
    if explicit_version is not None and explicit_version != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported artifact formatVersion: {explicit_version!r}; expected {ARTIFACT_FORMAT_VERSION}"
        )
    explicit_semantics = manifest.get("artifactSemantics")
    if explicit_semantics is not None and explicit_semantics != ARTIFACT_SEMANTICS:
        raise ValueError(f"Unsupported artifact semantics: {explicit_semantics!r}")
    if manifest.get("taskType") != "tabular_regression":
        raise ValueError(f"Artifact taskType mismatch: {manifest.get('taskType')!r}")

    base_model = manifest.get("baseModel")
    if not isinstance(base_model, dict):
        raise ValueError("Artifact manifest is missing baseModel provenance")
    for key, expected in EXPECTED_BASE_MODEL.items():
        actual = base_model.get(key)
        if actual != expected:
            raise ValueError(
                f"Artifact baseModel.{key} mismatch: expected {expected!r}, got {actual!r}"
            )

    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Artifact manifest is missing preprocessing state")
    if preprocessing.get("targetColumn") != manifest.get("targetColumn"):
        raise ValueError("Artifact targetColumn disagrees with preprocessing state")
    if list(preprocessing.get("dropColumns", [])) != list(manifest.get("dropColumns", [])):
        raise ValueError("Artifact dropColumns disagree with preprocessing state")

    context = manifest.get("trainingContext")
    if not isinstance(context, dict):
        raise ValueError("Artifact manifest is missing trainingContext")
    context_path = _safe_bundle_member(manifest_path, context.get("path", ""))
    if not context_path.is_file() or context_path.is_symlink():
        raise ValueError(f"Training context must be a regular file: {context_path}")

    expected_size = context.get("size")
    if expected_size is not None:
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError("Training context size is invalid")
        if expected_size != context_path.stat().st_size:
            raise ValueError(
                f"Training context size mismatch: expected {expected_size}, got {context_path.stat().st_size}"
            )
    expected_sha = context.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("Training context SHA-256 is missing or malformed")
    actual_sha = _sha256(context_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"Training context SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return manifest, context_path


def load_verified_artifact(
    artifact_path: str | Path,
    *,
    model_weight_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    use_flash: bool = False,
    compile_model: bool = False,
    verbose: bool = False,
    seed: int | None = None,
) -> TabDPTRegressionPipeline:
    """Validate a DIMER artifact and reconstruct its serving pipeline without preprocessing refit."""
    manifest_path = Path(artifact_path)
    _, context_path = validate_artifact_bundle(manifest_path)
    return TabDPTRegressionPipeline.load_artifact(
        manifest_path,
        context_path=context_path,
        model_weight_path=model_weight_path,
        cache_dir=cache_dir,
        device=device,
        use_flash=use_flash,
        compile_model=compile_model,
        verbose=verbose,
        seed=seed,
    )
