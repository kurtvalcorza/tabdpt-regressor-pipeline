from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TABDPT_PACKAGE_VERSION = "1.2.0"
TABDPT_UPSTREAM_CODE_COMMIT = "9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246"
TABDPT_HF_REPO = "Layer6/TabDPT"
TABDPT_HF_REVISION = "4462ffbd1d8dea25d4862d30beed4b70cd596ae5"
TABDPT_WEIGHT_FILENAME = "tabdpt1_2.safetensors"
TABDPT_WEIGHT_SHA256 = "06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"


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


class TabularFeatureEncoder:
    """Train-fitted mixed-table encoder with explicit missing/unknown categorical codes."""

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.numeric_columns: set[str] = set()
        self.category_maps: dict[str, dict[str, int]] = {}
        self.is_fitted = False

    def fit(self, frame: pd.DataFrame) -> "TabularFeatureEncoder":
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


class TabDPTRegressionPipeline:
    def __init__(
        self,
        model_weight_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        use_flash: bool = True,
        compile_model: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model_weight_path = model_weight_path
        self.cache_dir = cache_dir
        self.device = device
        self.use_flash = use_flash
        self.compile_model = compile_model
        self.verbose = verbose
        self.feature_encoder = TabularFeatureEncoder()
        self.target_column: str | None = None
        self.estimator: Any | None = None

    def fit(self, frame: pd.DataFrame, target_column: str = "target", drop_columns: list[str] | None = None):
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate column names are not supported")
        if target_column not in frame.columns:
            raise ValueError(f"Target column {target_column!r} not found")
        drop_columns = [c for c in (drop_columns or []) if c != target_column]
        raw_target = frame[target_column]
        target = pd.to_numeric(raw_target, errors="coerce")
        invalid = raw_target.notna() & target.isna()
        if invalid.any():
            examples = raw_target[invalid].astype(str).head(5).tolist()
            raise ValueError(f"Regression target contains non-numeric values: {examples}")
        if target.isna().any() or not np.isfinite(target.to_numpy(dtype=np.float64)).all():
            raise ValueError("Regression target must be finite and non-missing")
        if target.nunique() < 2:
            raise ValueError("Regression target must not be constant")
        features = frame.drop(columns=[target_column, *drop_columns], errors="ignore")
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

    def _require_fitted(self):
        if self.estimator is None or self.target_column is None:
            raise RuntimeError("Pipeline is not fitted")

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = self.feature_encoder.feature_columns
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"Feature schema mismatch; missing={missing}")
        return frame.loc[:, required]

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
        features = frame.drop(columns=[self.target_column])
        pred = self.predict(features, **kwargs)
        mse = mean_squared_error(y_true, pred)
        return {
            "mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, pred)),
        }
