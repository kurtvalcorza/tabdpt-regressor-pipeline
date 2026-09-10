from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import (
    TABDPT_HF_REPO,
    TABDPT_HF_REVISION,
    TABDPT_UPSTREAM_CODE_COMMIT,
    TABDPT_WEIGHT_FILENAME,
    TABDPT_WEIGHT_SHA256,
    TabDPTRegressionPipeline,
    TabularFeatureEncoder,
    _set_deterministic_seed,
    resolve_tabdpt_weights,
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


class _RestoredMeanImputer:
    """Minimal inference-only mean imputer reconstructed from fitted statistics."""

    def __init__(self, statistics: np.ndarray) -> None:
        self.statistics_ = np.asarray(statistics, dtype=np.float64)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.statistics_):
            raise ValueError("Restored imputer input width does not match fitted statistics")
        if np.isinf(array).any():
            raise ValueError("Inference features must not contain infinite values")
        return np.where(np.isnan(array), self.statistics_[None, :], array)


class _RestoredStandardScaler:
    """Minimal inference-only StandardScaler equivalent reconstructed from fitted state."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.mean_):
            raise ValueError("Restored scaler input width does not match fitted statistics")
        return (array - self.mean_[None, :]) / self.scale_[None, :]


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


def _finite_vector(value: Any, name: str, width: int, *, positive: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Artifact {name} must be a numeric vector") from exc
    if array.shape != (width,):
        raise ValueError(f"Artifact {name} must contain exactly {width} values")
    if not np.isfinite(array).all():
        raise ValueError(f"Artifact {name} must contain only finite values")
    if positive and not np.all(array > 0):
        raise ValueError(f"Artifact {name} must contain only positive values")
    return array


def _restore_state(
    preprocessing: dict[str, Any],
    width: int,
    *,
    require_complete: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    upstream = preprocessing.get("upstream")
    required = ("imputer_statistics", "scaler_mean", "scaler_scale")
    if not isinstance(upstream, dict) or any(key not in upstream for key in required):
        if require_complete:
            raise ValueError(
                "Artifact is missing fitted upstream preprocessing state required for no-refit reconstruction"
            )
        return None

    imputer_statistics = _finite_vector(
        upstream["imputer_statistics"], "preprocessing.upstream.imputer_statistics", width
    )
    scaler_mean = _finite_vector(
        upstream["scaler_mean"], "preprocessing.upstream.scaler_mean", width
    )
    scaler_scale = _finite_vector(
        upstream["scaler_scale"], "preprocessing.upstream.scaler_scale", width, positive=True
    )

    pca_basis = upstream.get("pca_basis")
    if pca_basis is not None:
        try:
            pca_basis = np.asarray(pca_basis, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Artifact preprocessing.upstream.pca_basis must be numeric") from exc
        if pca_basis.ndim != 2 or pca_basis.shape[0] != width or not np.isfinite(pca_basis).all():
            raise ValueError(
                "Artifact preprocessing.upstream.pca_basis must be a finite 2D matrix with one row per encoded feature"
            )
    return imputer_statistics, scaler_mean, scaler_scale, pca_basis


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

    preprocessing = pipeline.export_preprocessing_state()
    _restore_state(preprocessing, len(expected_features), require_complete=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "training_context.parquet"
    training_context.to_parquet(context_path, index=False)

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
    additive fields are present they are validated. New explicit-version artifacts must also
    carry complete fitted upstream preprocessing state so verified serving can reconstruct it
    without fitting preprocessing again.
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
    encoder_state = preprocessing.get("encoder")
    if not isinstance(encoder_state, dict):
        raise ValueError("Artifact preprocessing state is missing encoder state")
    feature_columns = encoder_state.get("featureColumns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Artifact encoder featureColumns must be a non-empty list")
    _restore_state(
        preprocessing,
        len(feature_columns),
        require_complete=explicit_version is not None or explicit_semantics is not None,
    )

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


def _read_context(context_path: Path, preprocessing: dict[str, Any]) -> pd.DataFrame:
    encoder_state = preprocessing["encoder"]
    category_columns = list(encoder_state.get("categoryMaps", {}).keys())
    suffix = context_path.suffix.lower()
    if suffix not in (".parquet", ".pq"):
        raise ValueError("Verified v3 artifacts require a Parquet training context")
    try:
        context = pd.read_parquet(context_path, engine="pyarrow")
    except ImportError as exc:
        raise ImportError("pyarrow is required to load verified v3 artifact context") from exc
    for column in category_columns:
        if column in context.columns:
            context[column] = context[column].astype("string")
    return context


def _pca_on_estimator_device(estimator: Any, pca_basis: np.ndarray) -> Any:
    target_device = getattr(estimator, "device", None) or "cpu"
    try:
        import torch

        return torch.as_tensor(pca_basis, dtype=torch.float32, device=target_device)
    except Exception:
        return SimpleNamespace(data=np.asarray(pca_basis, dtype=np.float32), device=target_device, dtype="float32")


def _reconstruct_without_preprocessing_refit(
    manifest: dict[str, Any],
    context: pd.DataFrame,
    *,
    model_weight_path: str | Path | None,
    cache_dir: str | Path | None,
    device: str | None,
    use_flash: bool,
    compile_model: bool,
    verbose: bool,
    seed: int | None,
) -> TabDPTRegressionPipeline | None:
    preprocessing = manifest["preprocessing"]
    encoder_state = preprocessing["encoder"]
    effective_seed = seed if seed is not None else preprocessing.get("seed", 42)
    _set_deterministic_seed(effective_seed)

    pipeline = TabDPTRegressionPipeline(
        model_weight_path=model_weight_path,
        cache_dir=cache_dir,
        device=device,
        use_flash=use_flash,
        compile_model=compile_model,
        verbose=verbose,
        seed=effective_seed,
    )
    pipeline.feature_encoder = TabularFeatureEncoder.from_state(encoder_state)
    pipeline.target_column = preprocessing.get("targetColumn")
    pipeline.drop_columns_ = list(preprocessing.get("dropColumns", []))

    if not pipeline.target_column or pipeline.target_column not in context.columns:
        raise ValueError(f"Target column {pipeline.target_column!r} not found in artifact context")
    features = context.drop(columns=[pipeline.target_column, *pipeline.drop_columns_], errors="ignore")
    X = pipeline.feature_encoder.transform(features)
    y_series = pd.to_numeric(context[pipeline.target_column], errors="coerce")
    if y_series.isna().any() or not np.isfinite(y_series.to_numpy(dtype=np.float64)).all():
        raise ValueError("Regression target in artifact context must be finite and numeric")
    if y_series.nunique() < 2:
        raise ValueError("Regression target in artifact context must not be constant")
    y = y_series.to_numpy(dtype=np.float64)

    restore = _restore_state(
        preprocessing,
        X.shape[1],
        require_complete=manifest.get("formatVersion") is not None or manifest.get("artifactSemantics") is not None,
    )
    if restore is None:
        return None
    imputer_statistics, scaler_mean, scaler_scale, pca_basis = restore

    weights = resolve_tabdpt_weights(model_weight_path, cache_dir)
    from tabdpt import TabDPTRegressor

    estimator = TabDPTRegressor(
        model_weight_path=str(weights),
        device=device,
        use_flash=use_flash,
        compile=compile_model,
        context_reduction="subsample",
        verbose=verbose,
    )
    if getattr(estimator, "missing_indicators", False):
        raise ValueError("Artifact reconstruction supports TabDPT missing_indicators=False only")
    normalizer = getattr(estimator, "normalizer", "standard")
    if normalizer != "standard":
        raise ValueError(f"Artifact reconstruction expected TabDPT standard normalizer, got {normalizer!r}")

    imputer = _RestoredMeanImputer(imputer_statistics)
    scaler = _RestoredStandardScaler(scaler_mean, scaler_scale)
    X_train = scaler.transform(imputer.transform(X))

    estimator.imputer = imputer
    estimator.scaler = scaler
    estimator.X_train = X_train
    estimator.y_train = y
    estimator.n_instances, estimator.n_features = X_train.shape
    estimator.faiss_knn = None
    estimator.is_fitted_ = True

    # Test doubles and diagnostics may expose raw conditioning arrays under X/y.
    if hasattr(estimator, "X"):
        estimator.X = X
    if hasattr(estimator, "y"):
        estimator.y = y

    feature_reduction = getattr(estimator, "feature_reduction", "pca")
    max_features = int(getattr(estimator, "max_features", X_train.shape[1]))
    reduction_required = X_train.shape[1] > max_features
    if feature_reduction == "pca" and reduction_required:
        if pca_basis is None:
            raise ValueError("Wide artifact requires the fitted PCA basis for no-refit reconstruction")
        estimator.V = _pca_on_estimator_device(estimator, pca_basis)
    elif pca_basis is not None:
        estimator.V = _pca_on_estimator_device(estimator, pca_basis)
    else:
        estimator.V = None

    if compile_model and hasattr(estimator, "model") and hasattr(estimator.model, "compile"):
        estimator.model.compile()

    pipeline.estimator = estimator
    pipeline.preprocessing_restored_ = True
    return pipeline


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
    """Validate an artifact and reconstruct serving state without re-fitting saved preprocessing.

    New explicit-version artifacts must contain fitted upstream imputer/scaler state and use the
    no-refit reconstruction path. Older v3 artifacts without that complete additive state remain
    loadable through the legacy compatibility path, which reconditions from the saved support
    table and is therefore not used as release-grade artifact-inference evidence.
    """
    manifest_path = Path(artifact_path)
    manifest, context_path = validate_artifact_bundle(manifest_path)
    context = _read_context(context_path, manifest["preprocessing"])
    restored = _reconstruct_without_preprocessing_refit(
        manifest,
        context,
        model_weight_path=model_weight_path,
        cache_dir=cache_dir,
        device=device,
        use_flash=use_flash,
        compile_model=compile_model,
        verbose=verbose,
        seed=seed,
    )
    if restored is not None:
        return restored

    pipeline = TabDPTRegressionPipeline.load_artifact(
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
    pipeline.preprocessing_restored_ = False
    return pipeline
