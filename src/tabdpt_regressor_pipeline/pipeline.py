from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TABDPT_PACKAGE_VERSION = "1.2.0"
TABDPT_UPSTREAM_CODE_COMMIT = "9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246"

# Fleet snapshot identity (DIMER Notebook Specification 1.1, ST3/MOD13). The pinned upstream model is
# unchanged; these are the fleet-standard names for the same repository, revision, license and snapshot
# key. The TABDPT_* spellings below stay as the package's published names and alias these constants.
MODEL_ID = "Layer6/TabDPT"
MODEL_REVISION = "4462ffbd1d8dea25d4862d30beed4b70cd596ae5"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "tabdpt-1.2"
MANIFEST_NAME = "dimer-base-manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY

TABDPT_HF_REPO = MODEL_ID
TABDPT_HF_REVISION = MODEL_REVISION
TABDPT_WEIGHT_FILENAME = "tabdpt1_2.safetensors"
TABDPT_WEIGHT_SHA256 = "06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"

MIN_DISTINCT_TARGETS = 2  # `fit` refuses a constant target (R² would be undefined)
METRIC_IDS = ("mae", "rmse", "r2")  # the ids `evaluate` reports (target units, target units, unitless)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_tabdpt_weights(model_weight_path: str | Path | None = None, cache_dir: str | Path | None = None) -> Path:
    if model_weight_path is None:
        model_weight_path = hf_hub_download(
            repo_id=TABDPT_HF_REPO,
            filename=TABDPT_WEIGHT_FILENAME,
            revision=TABDPT_HF_REVISION,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    path = Path(model_weight_path)
    if not path.is_file():
        raise FileNotFoundError(f"TabDPT model weight not found: {path}")
    actual = sha256_file(path)
    if actual != TABDPT_WEIGHT_SHA256:
        raise RuntimeError(
            f"TabDPT weight SHA-256 mismatch: expected {TABDPT_WEIGHT_SHA256}, got {actual}"
        )
    return path


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local pinned snapshot against its manifest; raise naming the first mismatch.

    The manifest is the parity anchor the standalone tutorial carries inline (NOTEBOOK_SPEC 1.1 ST3).
    The package's own ``TABDPT_WEIGHT_SHA256`` is not replaced by it: the manifest entry for
    ``TABDPT_WEIGHT_FILENAME`` must equal that constant, so the two can never diverge silently.
    """
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    entries = manifest.get("files", [])
    declared = {entry["path"]: entry["sha256"] for entry in entries}
    if declared.get(TABDPT_WEIGHT_FILENAME) != TABDPT_WEIGHT_SHA256:
        raise ValueError(
            f"manifest {TABDPT_WEIGHT_FILENAME} sha256 {declared.get(TABDPT_WEIGHT_FILENAME)!r} "
            f"!= TABDPT_WEIGHT_SHA256 {TABDPT_WEIGHT_SHA256!r}"
        )
    for entry in entries:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    hf_hub_download(
        repo_id=MODEL_ID,
        filename=relative_path,
        revision=MODEL_REVISION,
        local_dir=str(root),
    )


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a clone commits the manifest but
    git-ignores the checkpoint). Returns the relative paths fetched; ``verify_snapshot`` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _resolve_use_flash(requested: bool | None, device: str | None) -> bool:
    """Enable FlashAttention by default only on CUDA devices with compute capability >= 8.0."""
    if requested is not None:
        return requested
    if device is not None and not str(device).startswith("cuda"):
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        target = None if device in (None, "cuda") else device
        major, _ = torch.cuda.get_device_capability(target)
        return major >= 8
    except Exception:
        return False


class TabularFeatureEncoder:
    """Train-fitted mixed-table encoder with explicit missing/unknown categorical codes."""

    STATE_SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.numeric_columns: set[str] = set()
        self.category_maps: dict[str, dict[str, int]] = {}
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame) -> TabularFeatureEncoder:
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate feature column names are not supported")
        if frame.shape[1] == 0:
            raise ValueError("At least one feature column is required")
        self.feature_columns = list(frame.columns)
        self.numeric_columns = {
            col for col in self.feature_columns if pd.api.types.is_numeric_dtype(frame[col])
        }
        self.category_maps = {}
        for col in self.feature_columns:
            if col in self.numeric_columns:
                continue
            values = sorted({str(v) for v in frame[col].dropna().tolist()})
            self.category_maps[col] = {value: idx for idx, value in enumerate(values)}
        self.is_fitted = True
        return self

    def to_state(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Feature encoder is not fitted")
        return {
            "schemaVersion": self.STATE_SCHEMA_VERSION,
            "featureColumns": list(self.feature_columns),
            "numericColumns": [col for col in self.feature_columns if col in self.numeric_columns],
            "categoryMaps": {
                col: dict(self.category_maps[col])
                for col in self.feature_columns
                if col in self.category_maps
            },
            "categoricalEncoding": {
                "valueNormalization": "str",
                "unknownCode": "len(categoryMap)",
                "missingCode": "len(categoryMap)+1",
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TabularFeatureEncoder:
        if not isinstance(state, dict) or state.get("schemaVersion") != cls.STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported feature-encoder state schema")
        feature_columns = state.get("featureColumns")
        numeric_columns = state.get("numericColumns")
        category_maps = state.get("categoryMaps")
        if not isinstance(feature_columns, list) or not feature_columns or not all(isinstance(v, str) for v in feature_columns):
            raise ValueError("featureColumns must be a non-empty list of strings")
        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError("featureColumns contains duplicates")
        if not isinstance(numeric_columns, list) or not all(isinstance(v, str) for v in numeric_columns):
            raise ValueError("numericColumns must be a list of strings")
        if not set(numeric_columns).issubset(feature_columns):
            raise ValueError("numericColumns must be a subset of featureColumns")
        if not isinstance(category_maps, dict):
            raise ValueError("categoryMaps must be an object")
        expected_categorical = set(feature_columns) - set(numeric_columns)
        if set(category_maps) != expected_categorical:
            raise ValueError("categoryMaps must exactly cover non-numeric feature columns")
        normalized_maps: dict[str, dict[str, int]] = {}
        for col in feature_columns:
            if col in numeric_columns:
                continue
            mapping = category_maps[col]
            if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in mapping.items()):
                raise ValueError(f"Invalid category map for {col!r}")
            codes = sorted(mapping.values())
            if codes != list(range(len(codes))):
                raise ValueError(f"Category codes for {col!r} must be contiguous from zero")
            normalized_maps[col] = dict(mapping)
        encoder = cls()
        encoder.feature_columns = list(feature_columns)
        encoder.numeric_columns = set(numeric_columns)
        encoder.category_maps = normalized_maps
        encoder.is_fitted = True
        return encoder

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Feature encoder is not fitted")
        missing = [col for col in self.feature_columns if col not in frame.columns]
        extra = [col for col in frame.columns if col not in self.feature_columns]
        if missing or extra:
            raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
        out = np.empty((len(frame), len(self.feature_columns)), dtype=np.float64)
        for idx, col in enumerate(self.feature_columns):
            series = frame[col]
            if col in self.numeric_columns:
                out[:, idx] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
                continue
            mapping = self.category_maps[col]
            unknown_code = float(len(mapping))
            missing_code = float(len(mapping) + 1)
            encoded = []
            for value in series.tolist():
                if pd.isna(value):
                    encoded.append(missing_code)
                else:
                    encoded.append(float(mapping.get(str(value), unknown_code)))
            out[:, idx] = encoded
        return out

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


def _set_deterministic_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, AttributeError):
        pass


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "pandas.DataFrame, one row per example; feature columns of any dtype plus, for fit/evaluate, "
        "a numeric target column"
    ),
    "columns": "unique column names; `drop_columns` are removed before encoding",
    "target": "numeric, finite, no missing values, at least MIN_DISTINCT_TARGETS distinct values",
    "distinct_targets": [MIN_DISTINCT_TARGETS, None],
    "features": [1, None],
    "inference_input": (
        "exactly the fitted feature columns (after `drop_columns`), no target or output columns"
    ),
    "preprocessing": (
        "numeric columns are kept (NaN passes to TabDPT's support-fitted mean imputer); other columns "
        "are mapped to fitted integer codes with dedicated missing and unknown codes; upstream "
        "standardisation and any PCA basis are fitted on the support rows and reused at inference"
    ),
}


def _check_fit_inputs(
    frame: pd.DataFrame, target_column: str, drop_columns: Sequence[str] | None
) -> tuple[list[str], pd.DataFrame, pd.Series]:
    """The checks `fit` applies, in `fit`'s order, raising `fit`'s errors; returns what `fit` derives."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported")
    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} not found")
    drops = list(dict.fromkeys(c for c in (drop_columns or []) if c != target_column))
    raw_target = frame[target_column]
    target = pd.to_numeric(raw_target, errors="coerce")
    invalid = raw_target.notna() & target.isna()
    if invalid.any():
        examples = raw_target[invalid].astype(str).head(5).tolist()
        raise ValueError(f"Regression target contains non-numeric values: {examples}")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=np.float64)).all():
        raise ValueError("Regression target must be finite and non-missing")
    if target.nunique() < MIN_DISTINCT_TARGETS:
        raise ValueError("Regression target must not be constant")
    features = frame.drop(columns=[target_column, *drops], errors="ignore")
    if features.shape[1] == 0:
        raise ValueError("At least one feature column is required")
    return drops, features, target


def _check_inference_inputs(
    frame: pd.DataFrame, required: Sequence[str], drop_columns: Sequence[str]
) -> pd.DataFrame:
    """The schema check `predict` applies to an inference table; returns the ordered feature frame."""
    effective = frame.drop(columns=list(drop_columns), errors="ignore")
    missing = [col for col in required if col not in effective.columns]
    extra = [col for col in effective.columns if col not in required]
    if missing or extra:
        raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
    return effective.loc[:, list(required)]


def validate_inputs(
    frame: pd.DataFrame,
    target_column: str | None = "target",
    drop_columns: Sequence[str] | None = None,
    *,
    feature_columns: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observed table properties, verdict).

    With a ``target_column`` the table is checked exactly as ``fit`` checks it; with
    ``target_column=None`` it is an inference table checked against ``feature_columns`` (the fitted
    schema) exactly as ``predict`` checks it. Rejection is reported by raising the same error the
    core method raises; a caller that wants the finding recorded catches it and stores ``str(exc)``.
    """
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (the table's id)")
    table_id = names[0] if names else "table-0"
    if target_column is None:
        if feature_columns is None:
            raise ValueError("feature_columns is required to validate an inference table")
        drops = list(drop_columns or [])
        checked = _check_inference_inputs(frame, list(feature_columns), drops)
        missing_counts = checked.isna().sum()
        entry: dict[str, Any] = {
            "id": table_id,
            "mode": "inference",
            "rows": len(checked),
            "feature_columns": list(checked.columns),
            "missing_value_columns": {str(col): int(n) for col, n in missing_counts.items() if n > 0},
        }
    else:
        drops, features, target = _check_fit_inputs(frame, target_column, drop_columns)
        numeric = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
        missing_counts = features.isna().sum()
        values = target.to_numpy(dtype=np.float64)
        entry = {
            "id": table_id,
            "mode": "fit",
            "rows": len(frame),
            "feature_columns": list(features.columns),
            "numeric_columns": numeric,
            "categorical_columns": [col for col in features.columns if col not in numeric],
            "missing_value_columns": {str(col): int(n) for col, n in missing_counts.items() if n > 0},
            "target_summary": {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "distinct": int(target.nunique()),
            },
        }
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [entry],
        "target_column": target_column,
        "drop_columns": drops,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def training_mean_baseline(
    support_targets: Sequence[Any], holdout_targets: Sequence[Any]
) -> dict[str, float]:
    """The trivial baseline `evaluate` is compared against: always predict the support mean.

    The metric ids and their definitions are the ones ``evaluate`` reports (MAE and RMSE in target
    units, R² relative to the holdout's own mean, so the baseline's R² is at most 0).
    """
    support = pd.to_numeric(pd.Series(list(support_targets)), errors="coerce")
    holdout = pd.to_numeric(pd.Series(list(holdout_targets)), errors="coerce")
    for name, series in (("support_targets", support), ("holdout_targets", holdout)):
        if series.empty or series.isna().any() or not np.isfinite(series.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} must be non-empty, numeric and finite")
    if holdout.nunique() < MIN_DISTINCT_TARGETS:
        raise ValueError("Regression evaluation target must not be constant because R² is undefined")
    y_true = holdout.to_numpy(dtype=np.float64)
    constant = np.full(len(y_true), float(support.mean()))
    return {
        "mae": float(mean_absolute_error(y_true, constant)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, constant))),
        "r2": float(r2_score(y_true, constant)),
    }


def evaluation_report(
    metrics: Mapping[str, float] | None,
    *,
    baseline: Mapping[str, float] | None = None,
    n_holdout: int | None = None,
    target_column: str | None = None,
    sample_kind: str = "sample",
    estimation: str = "single seeded random holdout; no dispersion estimate",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``metrics`` is the dict ``evaluate`` returns (ids ``mae``, ``rmse``, ``r2``) and ``baseline`` the
    dict ``training_mean_baseline`` returns; the report is ``sample-sanity`` evidence. Without metrics
    (no labelled holdout) the verdict is ``not-measurable`` and the report says what labelled data would
    make the task measurable.
    """
    base: dict[str, Any] = {
        "task": "tabular regression by in-context conditioning on labelled support rows",
        "score_semantics": (
            "continuous point predictions in target units; no per-prediction uncertainty interval is produced"
        ),
        "sample_kind": sample_kind,
        "n_holdout": n_holdout,
        "target_column": target_column,
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if metrics is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no labelled holdout rows were supplied for the scored table",
            "needs": (
                "a labelled holdout table with a finite, non-constant numeric target column, scored with "
                "`evaluate` (mae, rmse, r2) against `training_mean_baseline`; an independent test set from "
                "the deployment domain for any generalisable claim, and calibration data before any "
                "prediction interval is attached"
            ),
        }
    unknown = sorted(set(metrics) - set(METRIC_IDS))
    if unknown:
        raise ValueError(f"unknown metric ids {unknown}; `evaluate` reports {list(METRIC_IDS)}")
    units = {"mae": "target units", "rmse": "target units", "r2": "unitless (1 - SSE/SST)"}
    reported = [
        {
            "id": metric_id,
            "value": float(metrics[metric_id]),
            "units": units[metric_id],
            "higher_is_better": metric_id == "r2",
            "estimation": estimation,
        }
        for metric_id in METRIC_IDS
        if metric_id in metrics
    ]
    baselines = []
    if baseline is not None:
        baselines.append(
            {
                "id": "training_mean",
                "metrics": [
                    {"id": metric_id, "value": float(baseline[metric_id])}
                    for metric_id in METRIC_IDS
                    if metric_id in baseline
                ],
            }
        )
    rows = "an unstated number of" if n_holdout is None else str(n_holdout)
    return {
        **base,
        "metrics": reported,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": f"{rows} labelled holdout row(s) from one seeded split; tutorial evidence, not a benchmark",
        "needs": (
            "an independent, domain-representative labelled test set for any generalisable quality "
            "claim; the point predictions carry no uncertainty interval"
        ),
    }


class TabDPTRegressionPipeline:
    def __init__(
        self,
        model_weight_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        use_flash: bool | None = None,
        compile_model: bool = False,
        verbose: bool = False,
        seed: int | None = 42,
    ) -> None:
        self.model_weight_path = model_weight_path
        self.cache_dir = cache_dir
        self.device = device
        self.use_flash = _resolve_use_flash(use_flash, device)
        self.compile_model = compile_model
        self.verbose = verbose
        self.seed = seed
        self.feature_encoder = TabularFeatureEncoder()
        self.target_column: str | None = None
        self.drop_columns_: list[str] = []
        self.estimator: Any | None = None
        self.source: str = "local-snapshot" if model_weight_path is not None else "hf-cache"

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        **kwargs: Any,
    ) -> TabDPTRegressionPipeline:
        """Build a pipeline whose base checkpoint is the digest-verified snapshot in ``weights_dir``.

        Stages only the manifest entries that are absent (at ``MODEL_REVISION``), re-hashes every entry
        against the manifest, and then pins ``model_weight_path`` to the verified file, so the in-context
        ``fit`` that follows can load nothing else. No model is loaded here.
        """
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        weight_path = root / TABDPT_WEIGHT_FILENAME
        pipeline = cls(model_weight_path=weight_path, **kwargs)
        pipeline.source = "local-snapshot"
        return pipeline

    def fit(
        self,
        frame: pd.DataFrame,
        target_column: str = "target",
        drop_columns: list[str] | None = None,
        seed: int | None = None,
    ):
        # The same checks `validate_inputs` applies (one shared function, so they cannot diverge).
        drops, features, target = _check_fit_inputs(frame, target_column, drop_columns)
        if seed is not None:
            self.seed = seed
        _set_deterministic_seed(self.seed)
        self.drop_columns_ = drops
        X = self.feature_encoder.fit_transform(features)
        y = target.to_numpy(dtype=np.float64)
        weights = resolve_tabdpt_weights(self.model_weight_path, self.cache_dir)
        from tabdpt import TabDPTRegressor

        self.estimator = TabDPTRegressor(
            model_weight_path=str(weights),
            device=self.device,
            use_flash=self.use_flash,
            compile=self.compile_model,
            context_reduction="subsample",
            verbose=self.verbose,
        )
        self.estimator.fit(X, y)
        self.target_column = target_column
        return self

    def export_preprocessing_state(self) -> dict[str, Any]:
        if not self.feature_encoder.is_fitted or self.target_column is None:
            raise RuntimeError("Pipeline preprocessing state is not fitted")
        state: dict[str, Any] = {
            "schemaVersion": 1,
            "targetColumn": self.target_column,
            "dropColumns": list(self.drop_columns_),
            "seed": self.seed,
            "encoder": self.feature_encoder.to_state(),
        }
        if self.estimator is not None:
            upstream: dict[str, Any] = {"seed": self.seed}
            V = getattr(self.estimator, "V", None)
            if V is not None:
                if hasattr(V, "detach"):
                    upstream["pca_basis"] = V.detach().cpu().numpy().tolist()
                elif isinstance(V, np.ndarray):
                    upstream["pca_basis"] = V.tolist()
                elif isinstance(V, list):
                    upstream["pca_basis"] = V
                elif hasattr(V, "data"):
                    v_data = V.data
                    if hasattr(v_data, "detach"):
                        upstream["pca_basis"] = v_data.detach().cpu().numpy().tolist()
                    elif isinstance(v_data, np.ndarray):
                        upstream["pca_basis"] = v_data.tolist()
                    elif isinstance(v_data, list):
                        upstream["pca_basis"] = v_data
                    else:
                        upstream["pca_basis"] = np.asarray(v_data).tolist()
            imputer = getattr(self.estimator, "imputer", None)
            if imputer is not None and hasattr(imputer, "statistics_") and imputer.statistics_ is not None:
                upstream["imputer_statistics"] = np.asarray(imputer.statistics_).tolist()
            scaler = getattr(self.estimator, "scaler", None)
            if scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None:
                upstream["scaler_mean"] = np.asarray(scaler.mean_).tolist()
                upstream["scaler_scale"] = np.asarray(scaler.scale_).tolist()
            state["upstream"] = upstream
        return state

    def condition_on_context(
        self,
        context_frame: pd.DataFrame,
        preprocessing_state: dict[str, Any],
        seed: int | None = None,
    ) -> TabDPTRegressionPipeline:
        """Condition the pipeline on an in-context support table using restored preprocessing state without refitting."""
        if not isinstance(preprocessing_state, dict) or preprocessing_state.get("schemaVersion") != 1:
            raise ValueError("Unsupported preprocessing state schemaVersion")
        encoder_state = preprocessing_state.get("encoder")
        if not isinstance(encoder_state, dict):
            raise ValueError("Invalid preprocessing state: missing 'encoder'")
        target_column = preprocessing_state.get("targetColumn")
        if not target_column or target_column not in context_frame.columns:
            raise ValueError(f"Target column {target_column!r} not found in context frame")

        effective_seed = seed if seed is not None else preprocessing_state.get("seed", self.seed)
        _set_deterministic_seed(effective_seed)
        self.seed = effective_seed

        self.feature_encoder = TabularFeatureEncoder.from_state(encoder_state)
        self.target_column = target_column
        self.drop_columns_ = list(preprocessing_state.get("dropColumns", []))

        features = context_frame.drop(columns=[target_column, *self.drop_columns_], errors="ignore")
        X = self.feature_encoder.transform(features)

        raw_target = pd.to_numeric(context_frame[target_column], errors="coerce")
        if raw_target.isna().any() or not np.isfinite(raw_target.to_numpy(dtype=np.float64)).all():
            raise ValueError("Regression target in context frame must be finite and non-missing")
        if raw_target.nunique() < 2:
            raise ValueError("Regression target in context frame must not be constant")
        y = raw_target.to_numpy(dtype=np.float64)

        weights = resolve_tabdpt_weights(self.model_weight_path, self.cache_dir)
        from tabdpt import TabDPTRegressor

        self.estimator = TabDPTRegressor(
            model_weight_path=str(weights),
            device=self.device,
            use_flash=self.use_flash,
            compile=self.compile_model,
            context_reduction="subsample",
            verbose=self.verbose,
        )
        self.estimator.fit(X, y)

        upstream = preprocessing_state.get("upstream")
        if isinstance(upstream, dict):
            pca_basis = upstream.get("pca_basis")
            if pca_basis is not None and hasattr(self.estimator, "V"):
                existing_v = getattr(self.estimator, "V", None)
                target_device = (
                    getattr(existing_v, "device", None)
                    or getattr(self.estimator, "device", None)
                    or self.device
                    or "cpu"
                )
                try:
                    import torch

                    target_dtype = getattr(existing_v, "dtype", torch.float32)
                    self.estimator.V = torch.as_tensor(
                        pca_basis,
                        dtype=target_dtype,
                        device=target_device,
                    )
                except Exception:
                    if hasattr(existing_v, "device") or (target_device and str(target_device) != "cpu"):
                        from types import SimpleNamespace

                        self.estimator.V = SimpleNamespace(
                            data=np.array(pca_basis, dtype=np.float32),
                            device=target_device,
                            dtype=getattr(existing_v, "dtype", "float32"),
                        )
                    else:
                        self.estimator.V = np.array(pca_basis, dtype=np.float32)
        return self

    @classmethod
    def load_artifact(
        cls,
        artifact_path: str | Path,
        context_path: str | Path | None = None,
        model_weight_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        use_flash: bool | None = None,
        compile_model: bool = False,
        verbose: bool = False,
        seed: int | None = None,
    ) -> TabDPTRegressionPipeline:
        """Load a DIMER serving artifact bundle, restoring preprocessing from manifest without refitting."""
        artifact_file = Path(artifact_path)
        if not artifact_file.is_file():
            raise FileNotFoundError(f"Artifact manifest not found: {artifact_file}")
        manifest = json.loads(artifact_file.read_text(encoding="utf-8"))
        fmt = manifest.get("format")
        if fmt not in ("tabdpt-dimer-context-v3", "tabdpt-dimer-context-v2"):
            raise ValueError(f"Unsupported artifact format: {fmt!r}")
        if manifest.get("taskType") != "tabular_regression":
            raise ValueError(
                f"Artifact taskType mismatch: expected 'tabular_regression', got {manifest.get('taskType')!r}"
            )
        preprocessing = manifest.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise ValueError("Artifact manifest missing 'preprocessing' state")

        if context_path is None:
            default_name = "training_context.parquet" if fmt == "tabdpt-dimer-context-v3" else "training_context.csv"
            context_rel = manifest.get("trainingContext", {}).get("path", default_name)
            context_file = artifact_file.parent / context_rel
        else:
            context_file = Path(context_path)

        if not context_file.is_file():
            raise FileNotFoundError(f"Training context table not found: {context_file}")

        expected_sha = manifest.get("trainingContext", {}).get("sha256")
        if expected_sha:
            actual_sha = sha256_file(context_file)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Training context digest mismatch: expected {expected_sha}, got {actual_sha}"
                )

        encoder_state = preprocessing.get("encoder", {})
        category_cols = list(encoder_state.get("categoryMaps", {}).keys())

        suffix = context_file.suffix.lower()
        if suffix in (".parquet", ".pq"):
            try:
                context_df = pd.read_parquet(context_file, engine="pyarrow")
            except ImportError as err:
                raise ImportError(
                    "pyarrow is required to load parquet serving context in 'tabdpt-dimer-context-v3'. "
                    "Install it with 'pip install pyarrow'."
                ) from err
            for col in category_cols:
                if col in context_df.columns:
                    context_df[col] = context_df[col].astype("string")
        else:
            dtype_spec = {col: "string" for col in category_cols}
            context_df = pd.read_csv(context_file, dtype=dtype_spec)
        effective_seed = seed if seed is not None else preprocessing.get("seed", 42)
        pipeline = cls(
            model_weight_path=model_weight_path,
            cache_dir=cache_dir,
            device=device,
            use_flash=use_flash,
            compile_model=compile_model,
            verbose=verbose,
            seed=effective_seed,
        )
        pipeline.condition_on_context(context_df, preprocessing, seed=effective_seed)
        return pipeline

    def _require_fitted(self):
        if self.estimator is None or self.target_column is None:
            raise RuntimeError("Pipeline is not fitted")

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return _check_inference_inputs(frame, self.feature_encoder.feature_columns, self.drop_columns_)

    def predict(
        self,
        frame: pd.DataFrame,
        n_ensembles: int = 4,
        context_size: int | None = 2048,
        batch_size: int | None = 4096,
        seed: int = 42,
    ) -> pd.Series:
        self._require_fitted()
        X = self.feature_encoder.transform(self._feature_frame(frame))
        pred = self.estimator.predict(
            X,
            n_ensembles=n_ensembles,
            context_size=context_size,
            batch_size=batch_size,
            seed=seed,
        )
        return pd.Series(np.asarray(pred, dtype=np.float64), index=frame.index, name="prediction")

    def evaluate(self, frame: pd.DataFrame, **kwargs) -> dict[str, float]:
        self._require_fitted()
        if self.target_column not in frame.columns:
            raise ValueError(f"Evaluation target {self.target_column!r} not found")
        y_true = pd.to_numeric(frame[self.target_column], errors="coerce")
        if y_true.isna().any() or not np.isfinite(y_true.to_numpy(dtype=np.float64)).all():
            raise ValueError("Evaluation target must be finite and numeric")
        if y_true.nunique() < 2:
            raise ValueError("Regression evaluation target must not be constant because R² is undefined")
        features = frame.drop(columns=[self.target_column])
        pred = self.predict(features, **kwargs)
        mse = mean_squared_error(y_true, pred)
        return {
            "mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, pred)),
        }
