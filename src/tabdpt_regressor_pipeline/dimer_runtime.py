from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .pipeline import (
    TABDPT_HF_REPO,
    TABDPT_HF_REVISION,
    TABDPT_UPSTREAM_CODE_COMMIT,
    TABDPT_WEIGHT_FILENAME,
    TABDPT_WEIGHT_SHA256,
    TabDPTRegressionPipeline,
)

SUPPORTED_PREPROCESSING_KEYS = frozenset(
    {"target_column", "drop_columns", "max_train_rows", "validation_split"}
)
SUPPORTED_HYPERPARAMETER_KEYS = frozenset(
    {"fine_tune", "n_ensembles", "context_size", "batch_size", "seed"}
)
TRANSPORT_HYPERPARAMETER_KEYS = frozenset({"model_id"})


def _json_object(value: str | None, name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return parsed


def _reject_unknown(
    payload: dict[str, Any],
    supported: frozenset[str],
    name: str,
    transport_keys: frozenset[str] = frozenset(),
) -> None:
    unknown = sorted(set(payload) - supported - transport_keys)
    if unknown:
        raise ValueError(f"Unsupported {name} keys: {unknown}")


def _integer(payload: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _number(payload: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _drop_columns(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = [item.strip() for item in value]
    else:
        raise ValueError("drop_columns must be a comma-separated string or list of strings")
    return list(dict.fromkeys(item for item in items if item))


@dataclass(frozen=True)
class DimerRuntimeConfig:
    target_column: str = "target"
    drop_columns: tuple[str, ...] = ()
    max_train_rows: int = 10000
    validation_split: float = 0.2
    fine_tune: bool = False
    n_ensembles: int = 4
    context_size: int = 2048
    batch_size: int = 4096
    seed: int = 42

    @classmethod
    def from_payloads(
        cls, preprocessing: dict[str, Any] | None = None, hyperparameters: dict[str, Any] | None = None
    ) -> "DimerRuntimeConfig":
        pre = dict(preprocessing or {})
        hp = dict(hyperparameters or {})
        _reject_unknown(pre, SUPPORTED_PREPROCESSING_KEYS, "preprocessing")
        _reject_unknown(
            hp,
            SUPPORTED_HYPERPARAMETER_KEYS,
            "hyperparameter",
            transport_keys=TRANSPORT_HYPERPARAMETER_KEYS,
        )
        target_column = pre.get("target_column", "target")
        if not isinstance(target_column, str) or not target_column.strip() or len(target_column) > 128:
            raise ValueError("target_column must be a non-empty string of at most 128 characters")
        fine_tune = hp.get("fine_tune", False)
        if not isinstance(fine_tune, bool):
            raise ValueError("fine_tune must be boolean")
        if fine_tune:
            raise ValueError("TabDPT v1.2 does not support gradient fine-tuning; fine_tune must be false")
        return cls(
            target_column=target_column.strip(),
            drop_columns=tuple(_drop_columns(pre.get("drop_columns", ""))),
            max_train_rows=_integer(pre, "max_train_rows", 10000, 200, 50000),
            validation_split=_number(pre, "validation_split", 0.2, 0.05, 0.4),
            fine_tune=False,
            n_ensembles=_integer(hp, "n_ensembles", 4, 1, 16),
            context_size=_integer(hp, "context_size", 2048, 128, 16384),
            batch_size=_integer(hp, "batch_size", 4096, 1, 131072),
            seed=_integer(hp, "seed", 42, 0, 2147483647),
        )

    @classmethod
    def from_environment(cls) -> "DimerRuntimeConfig":
        return cls.from_payloads(
            _json_object(os.getenv("DIMER_PREPROCESSING_ARGS_JSON"), "DIMER_PREPROCESSING_ARGS_JSON"),
            _json_object(os.getenv("DIMER_HYPERPARAMETERS_JSON"), "DIMER_HYPERPARAMETERS_JSON"),
        )

    def inference_kwargs(self) -> dict[str, Any]:
        return {
            "n_ensembles": self.n_ensembles,
            "context_size": self.context_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


def _dataset_limits() -> tuple[int, int, int, float, int]:
    try:
        limits = (
            int(os.getenv("DIMER_MAX_ARCHIVE_BYTES", str(1024**3))),
            int(os.getenv("DIMER_MAX_UNCOMPRESSED_BYTES", str(2 * 1024**3))),
            int(os.getenv("DIMER_MAX_MEMBER_BYTES", str(512 * 1024**2))),
            float(os.getenv("DIMER_MAX_COMPRESSION_RATIO", "200")),
            int(os.getenv("DIMER_MAX_DATASET_FILES", "200")),
        )
    except ValueError as exc:
        raise ValueError("DIMER dataset safety limits must be numeric") from exc
    if any(value <= 0 for value in limits):
        raise ValueError("DIMER dataset safety limits must be positive")
    return limits


def _normalize_member(name: str) -> str | None:
    if not name or name.endswith("/"):
        return None
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0].endswith(":")):
        raise ValueError(f"Unsafe dataset archive path: {name!r}")
    return path.as_posix() or None


def _validate_archive(path: Path) -> list[str]:
    max_archive, max_uncompressed, max_member, max_ratio, max_files = _dataset_limits()
    if path.is_symlink():
        raise ValueError("Dataset ZIP must not be a symlink")
    if path.stat().st_size > max_archive:
        raise ValueError("Dataset ZIP exceeds DIMER_MAX_ARCHIVE_BYTES")
    members: list[str] = []
    seen: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                normalized = _normalize_member(info.filename)
                if normalized is None:
                    continue
                if info.file_size > max_member:
                    raise ValueError(f"Archive member {normalized!r} exceeds DIMER_MAX_MEMBER_BYTES")
                total += info.file_size
                if total > max_uncompressed:
                    raise ValueError("Dataset ZIP exceeds DIMER_MAX_UNCOMPRESSED_BYTES")
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > max_ratio:
                    raise ValueError(
                        f"Archive member {normalized!r} exceeds DIMER_MAX_COMPRESSION_RATIO"
                    )
                if normalized in seen:
                    raise ValueError(f"Duplicate normalized archive path: {normalized!r}")
                seen.add(normalized)
                members.append(info.filename)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {exc}") from exc
    if len(members) > max_files:
        raise ValueError(
            f"Dataset ZIP contains {len(members)} files; DIMER_MAX_DATASET_FILES={max_files}"
        )
    return members


def _validate_direct_dataset(root: Path) -> None:
    _, max_uncompressed, max_member, _, max_files = _dataset_limits()
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Dataset directory must not contain symlinks: {path.relative_to(root)}")
        if path.is_file():
            files.append(path)
    if len(files) > max_files:
        raise ValueError(
            f"Dataset directory contains {len(files)} files; DIMER_MAX_DATASET_FILES={max_files}"
        )
    total = 0
    for path in files:
        size = path.stat().st_size
        if size > max_member:
            raise ValueError(f"Dataset file {path.name!r} exceeds DIMER_MAX_MEMBER_BYTES")
        total += size
        if total > max_uncompressed:
            raise ValueError("Dataset directory exceeds DIMER_MAX_UNCOMPRESSED_BYTES")


def _find_csv(root: Path, stem: str, required: bool) -> Path | None:
    matches = sorted(path for path in root.rglob("*.csv") if path.stem.lower() == stem.lower())
    if len(matches) > 1:
        raise ValueError(f"Multiple {stem}.csv files found")
    if not matches:
        if required:
            raise ValueError(f"Dataset must contain {stem}.csv")
        return None
    return matches[0]


def _zip_member(members: list[str], stem: str, required: bool) -> str | None:
    matches = sorted(
        name
        for name in members
        if Path(name).suffix.lower() == ".csv" and Path(name).stem.lower() == stem.lower()
    )
    if len(matches) > 1:
        raise ValueError(f"Archive contains multiple {stem}.csv files")
    if not matches:
        if required:
            raise ValueError(f"Archive must contain {stem}.csv")
        return None
    return matches[0]


def load_dimer_tables(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    root = Path(dataset_dir)
    if root.is_symlink():
        raise ValueError("DIMER_DATASET_DIR must not be a symlink")
    if not root.is_dir():
        raise ValueError(f"DIMER_DATASET_DIR is not a directory: {root}")
    archives = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".zip")
    direct_csv = list(root.rglob("*.csv"))
    if archives:
        if len(archives) != 1 or direct_csv:
            raise ValueError("Dataset directory must contain either CSV files or exactly one ZIP archive")
        members = _validate_archive(archives[0])
        with zipfile.ZipFile(archives[0]) as archive:
            train_name = _zip_member(members, "train", required=True)
            val_name = _zip_member(members, "val", required=False)
            if train_name is None:
                raise ValueError("Archive must contain train.csv")
            with archive.open(train_name) as handle:
                train = pd.read_csv(handle)
            val = None
            if val_name is not None:
                with archive.open(val_name) as handle:
                    val = pd.read_csv(handle)
            return train, val
    _validate_direct_dataset(root)
    train_path = _find_csv(root, "train", required=True)
    val_path = _find_csv(root, "val", required=False)
    if train_path is None:
        raise ValueError("Dataset must contain train.csv")
    return pd.read_csv(train_path), pd.read_csv(val_path) if val_path is not None else None


def _validate_target(frame: pd.DataFrame, config: DimerRuntimeConfig, name: str) -> None:
    if config.target_column not in frame.columns:
        raise ValueError(f"{name}.csv is missing target column {config.target_column!r}")
    raw = frame[config.target_column]
    target = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & target.isna()
    if invalid.any() or target.isna().any() or not np.isfinite(target.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name}.csv target must be finite and numeric")
    if name == "train" and target.nunique() < 2:
        raise ValueError("Regression training target must not be constant")


def prepare_dimer_frames(
    train: pd.DataFrame, val: pd.DataFrame | None, config: DimerRuntimeConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_target(train, config, "train")
    if val is None:
        train, val = train_test_split(
            train,
            test_size=config.validation_split,
            random_state=config.seed,
        )
    else:
        _validate_target(val, config, "val")
    if len(train) > config.max_train_rows:
        train = train.sample(n=config.max_train_rows, random_state=config.seed)
    if val is None:
        raise RuntimeError("Validation split was not created")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def run_dimer_job() -> dict[str, Any]:
    config = DimerRuntimeConfig.from_environment()
    dataset_dir = os.getenv("DIMER_DATASET_DIR", "/data/dataset")
    output_dir = Path(os.getenv("DIMER_OUTPUT_DIR", "/data/output"))
    result_path = Path(os.getenv("DIMER_RESULT_PATH", str(output_dir / "result.json")))
    train, val = load_dimer_tables(dataset_dir)
    train, val = prepare_dimer_frames(train, val, config)

    weight_path = os.getenv("DIMER_BASE_MODEL_PATH", "").strip() or None
    pipeline = TabDPTRegressionPipeline(
        model_weight_path=weight_path,
        device=os.getenv("DIMER_DEVICE", "").strip() or None,
        compile_model=False,
        verbose=False,
    )
    pipeline.fit(train, target_column=config.target_column, drop_columns=list(config.drop_columns))
    metrics = pipeline.evaluate(val, **config.inference_kwargs())

    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    context_path = artifact_dir / "training_context.csv"
    train.to_csv(context_path, index=False)
    manifest_path = artifact_dir / "artifact.json"
    manifest = {
        "format": "tabdpt-dimer-context-v2",
        "taskType": "tabular_regression",
        "targetColumn": config.target_column,
        "dropColumns": list(config.drop_columns),
        "runtimeConfig": asdict(config),
        "preprocessing": pipeline.export_preprocessing_state(),
        "baseModel": {
            "repo": TABDPT_HF_REPO,
            "revision": TABDPT_HF_REVISION,
            "filename": TABDPT_WEIGHT_FILENAME,
            "sha256": TABDPT_WEIGHT_SHA256,
            "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
        },
        "trainingContext": {"path": context_path.name, "sha256": _sha256(context_path)},
    }
    _write_json(manifest_path, manifest)
    result = {
        "contractVersion": 1,
        "successful": True,
        "metadata": {
            "taskType": "tabular_regression",
            "runId": os.getenv("DIMER_RUN_ID", ""),
            "trainRows": len(train),
            "validationRows": len(val),
            "artifactFormat": manifest["format"],
        },
        "metrics": metrics,
        "artifacts": {
            "trainingContext": {"path": str(context_path), "sha256": _sha256(context_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
        "provenance": {"baseModelSha256": TABDPT_WEIGHT_SHA256, "baseModelRevision": TABDPT_HF_REVISION},
    }
    _write_json(result_path, result)
    return result


def main() -> int:
    try:
        run_dimer_job()
        return 0
    except Exception as exc:
        output_dir = Path(os.getenv("DIMER_OUTPUT_DIR", "/data/output"))
        result_path = Path(os.getenv("DIMER_RESULT_PATH", str(output_dir / "result.json")))
        _write_json(
            result_path,
            {
                "contractVersion": 1,
                "successful": False,
                "metadata": {"taskType": "tabular_regression", "runId": os.getenv("DIMER_RUN_ID", "")},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
