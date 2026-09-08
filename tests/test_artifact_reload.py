import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tabdpt_regressor_pipeline.pipeline import (
    TABDPT_HF_REPO,
    TABDPT_HF_REVISION,
    TABDPT_UPSTREAM_CODE_COMMIT,
    TABDPT_WEIGHT_FILENAME,
    TABDPT_WEIGHT_SHA256,
    TabDPTRegressionPipeline,
    TabularFeatureEncoder,
)


class FakeEstimator:
    def __init__(self, *args, **kwargs):
        device_arg = kwargs.get("device")
        # Simulate upstream TabDPT auto-selecting "cuda" if available when device is None
        self.device = "cuda" if device_arg is None else device_arg
        self.X = None
        self.y = None
        self.n_instances = None
        self.n_features = None
        self.max_features = 128
        self.feature_reduction = "pca"
        self.V = None
        self.imputer = SimpleNamespace(statistics_=None)
        self.scaler = SimpleNamespace(mean_=None, scale_=None)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_instances, self.n_features = X.shape
        self.imputer.statistics_ = np.nanmean(X, axis=0)
        self.scaler.mean_ = np.mean(X, axis=0)
        self.scaler.scale_ = np.std(X, axis=0) + 1e-6
        if self.n_features > self.max_features:
            rng_mat = np.random.randn(self.n_features, self.max_features).astype(np.float32)
            q, _ = np.linalg.qr(rng_mat)
            try:
                import torch

                if self.device == "cuda" and not torch.cuda.is_available():
                    self.V = SimpleNamespace(
                        data=q,
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    train_x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
                    _, _, self.V = torch.pca_lowrank(train_x, q=min(train_x.shape[0], self.max_features))
            except Exception:
                self.V = SimpleNamespace(
                    data=q,
                    device=self.device,
                    dtype="float32",
                )
        else:
            self.V = None
        return self

    def _check_device_alignment(self):
        if self.V is not None:
            v_device = getattr(self.V, "device", None)
            if v_device is not None:
                v_dev_str = str(v_device)
                est_dev_str = str(self.device)
                if ("cuda" in est_dev_str and "cpu" in v_dev_str) or ("cpu" in est_dev_str and "cuda" in v_dev_str):
                    raise RuntimeError(
                        f"Expected all tensors to be on the same device, but found at least two devices, {est_dev_str} and {v_dev_str}!"
                    )

    def _get_v_matrix(self):
        self._check_device_alignment()
        if self.V is None:
            return None
        if hasattr(self.V, "detach"):
            return self.V.detach().cpu().numpy()
        if hasattr(self.V, "data"):
            return np.asarray(self.V.data)
        return np.asarray(self.V)

    def predict(self, X, **kwargs):
        v_mat = self._get_v_matrix()
        if v_mat is not None:
            proj = X @ v_mat
            return proj.sum(axis=1).astype(np.float64)
        return np.ones(len(X), dtype=np.float64) * 42.0


@pytest.fixture
def mock_tabdpt(monkeypatch):
    import tabdpt_regressor_pipeline.pipeline as pipe_mod

    fake_module = SimpleNamespace(TabDPTRegressor=FakeEstimator)
    monkeypatch.setitem(__import__("sys").modules, "tabdpt", fake_module)
    monkeypatch.setattr(pipe_mod, "resolve_tabdpt_weights", lambda *a, **k: Path("fake.safetensors"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_artifact_reload_parity_preserves_numeric_looking_categories(tmp_path, mock_tabdpt):
    # 1. Prepare training table with numeric-looking string categories
    train = pd.DataFrame({
        "code": pd.Series(["01", "02", "03", "01"], dtype="object"),
        "val": [10.5, 20.5, 30.5, 40.5],
        "target": [100.0, 200.0, 300.0, 400.0],
    })

    pipe = TabDPTRegressionPipeline(compile_model=False, use_flash=False)
    pipe.fit(train, target_column="target")
    assert pipe.feature_encoder.numeric_columns == {"val"}
    assert pipe.feature_encoder.category_maps["code"] == {"01": 0, "02": 1, "03": 2}

    # 2. Export genuine tabdpt-dimer-context-v3 serving artifact bundle
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    context_path = artifact_dir / "training_context.parquet"
    train.to_parquet(context_path, index=False)

    preprocessing_state = pipe.export_preprocessing_state()
    manifest = {
        "format": "tabdpt-dimer-context-v3",
        "taskType": "tabular_regression",
        "targetColumn": "target",
        "dropColumns": list(preprocessing_state["dropColumns"]),
        "preprocessing": preprocessing_state,
        "baseModel": {
            "repo": TABDPT_HF_REPO,
            "revision": TABDPT_HF_REVISION,
            "filename": TABDPT_WEIGHT_FILENAME,
            "sha256": TABDPT_WEIGHT_SHA256,
            "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
        },
        "trainingContext": {
            "path": context_path.name,
            "sha256": _sha256(context_path),
        },
    }
    manifest_path = artifact_dir / "artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3. Prove that load_artifact restores exact fitted preprocessing without refitting
    restored_pipe = TabDPTRegressionPipeline.load_artifact(
        manifest_path,
        compile_model=False,
        use_flash=False,
    )
    assert restored_pipe.target_column == "target"
    assert restored_pipe.feature_encoder.numeric_columns == {"val"}
    assert restored_pipe.feature_encoder.category_maps["code"] == {"01": 0, "02": 1, "03": 2}

    # 4. Predict on new unlabelled test rows with string categories
    test_query = pd.DataFrame({
        "code": ["01", "02"],
        "val": [15.0, 25.0],
    })
    preds = restored_pipe.predict(test_query)
    assert len(preds) == 2
    assert np.allclose(preds.to_numpy(), 42.0)


def test_artifact_reload_backwards_compatible_with_v2_csv(tmp_path, mock_tabdpt):
    # 1. Prepare training table with numeric-looking string categories
    train = pd.DataFrame({
        "code": pd.Series(["01", "02", "03", "01"], dtype="object"),
        "val": [10.5, 20.5, 30.5, 40.5],
        "target": [100.0, 200.0, 300.0, 400.0],
    })

    pipe = TabDPTRegressionPipeline(compile_model=False, use_flash=False)
    pipe.fit(train, target_column="target")

    # 2. Export legacy tabdpt-dimer-context-v2 serving artifact bundle with training_context.csv
    artifact_dir = tmp_path / "artifacts_v2"
    artifact_dir.mkdir(parents=True)
    context_path = artifact_dir / "training_context.csv"
    train.to_csv(context_path, index=False)

    preprocessing_state = pipe.export_preprocessing_state()
    manifest = {
        "format": "tabdpt-dimer-context-v2",
        "taskType": "tabular_regression",
        "targetColumn": "target",
        "dropColumns": list(preprocessing_state["dropColumns"]),
        "preprocessing": preprocessing_state,
        "baseModel": {
            "repo": TABDPT_HF_REPO,
            "revision": TABDPT_HF_REVISION,
            "filename": TABDPT_WEIGHT_FILENAME,
            "sha256": TABDPT_WEIGHT_SHA256,
            "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
        },
        "trainingContext": {
            "path": context_path.name,
            "sha256": _sha256(context_path),
        },
    }
    manifest_path = artifact_dir / "artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3. Prove that naive CSV re-fitting destroys categorical semantics
    naive_csv = pd.read_csv(context_path)
    assert pd.api.types.is_numeric_dtype(naive_csv["code"]), "read_csv misinterprets string codes as integer"
    naive_enc = TabularFeatureEncoder().fit(naive_csv.drop(columns=["target"]))
    assert "code" in naive_enc.numeric_columns, "Naive refit misidentifies code as numeric"
    assert "code" not in naive_enc.category_maps, "Naive refit destroyed category maps"

    # 4. Prove that load_artifact restores exact fitted preprocessing from legacy v2 CSV
    restored_pipe = TabDPTRegressionPipeline.load_artifact(
        manifest_path,
        compile_model=False,
        use_flash=False,
    )
    assert restored_pipe.target_column == "target"
    assert restored_pipe.feature_encoder.numeric_columns == {"val"}
    assert restored_pipe.feature_encoder.category_maps["code"] == {"01": 0, "02": 1, "03": 2}

    # 5. Predict on new unlabelled test rows with string categories
    test_query = pd.DataFrame({
        "code": ["01", "02"],
        "val": [15.0, 25.0],
    })
    preds = restored_pipe.predict(test_query)
    assert len(preds) == 2
    assert np.allclose(preds.to_numpy(), 42.0)


def test_artifact_reload_rejects_context_digest_mismatch(tmp_path, mock_tabdpt):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    context_path = artifact_dir / "training_context.parquet"
    train = pd.DataFrame({"code": ["01", "02"], "val": [1.0, 2.0], "target": [10.0, 20.0]})
    train.to_parquet(context_path, index=False)

    manifest = {
        "format": "tabdpt-dimer-context-v3",
        "taskType": "tabular_regression",
        "preprocessing": {
            "schemaVersion": 1,
            "targetColumn": "target",
            "dropColumns": [],
            "encoder": {
                "schemaVersion": 1,
                "featureColumns": ["code", "val"],
                "numericColumns": ["val"],
                "categoryMaps": {"code": {"01": 0, "02": 1}},
            },
        },
        "trainingContext": {
            "path": "training_context.parquet",
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        },
    }
    manifest_path = artifact_dir / "artifact.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        TabDPTRegressionPipeline.load_artifact(manifest_path)


def test_artifact_reload_rejects_invalid_schemas_and_formats(tmp_path, mock_tabdpt):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    manifest_path = artifact_dir / "artifact.json"

    # Missing preprocessing
    manifest_path.write_text(json.dumps({"format": "tabdpt-dimer-context-v3", "taskType": "tabular_regression"}))
    with pytest.raises(ValueError, match="missing 'preprocessing' state"):
        TabDPTRegressionPipeline.load_artifact(manifest_path)

    # Wrong format
    manifest_path.write_text(json.dumps({"format": "wrong-format", "taskType": "tabular_regression"}))
    with pytest.raises(ValueError, match="Unsupported artifact format"):
        TabDPTRegressionPipeline.load_artifact(manifest_path)

    # Wrong taskType
    manifest_path.write_text(json.dumps({"format": "tabdpt-dimer-context-v3", "taskType": "tabular_classification"}))
    with pytest.raises(ValueError, match="Artifact taskType mismatch"):
        TabDPTRegressionPipeline.load_artifact(manifest_path)


def test_artifact_reload_parity_wide_dataset_exceeding_native_width(tmp_path, mock_tabdpt):
    # 1. Prepare wide training table with >128 features (135 features)
    n_samples = 30
    n_features = 135
    rng = np.random.default_rng(1234)
    data = {f"f_{i}": rng.normal(size=n_samples) for i in range(n_features)}
    data["target"] = rng.normal(loc=100.0, scale=10.0, size=n_samples)
    train = pd.DataFrame(data)

    # 2. Fit pre-export pipeline with deterministic seed=42
    pre_pipe = TabDPTRegressionPipeline(seed=42, compile_model=False, use_flash=False)
    pre_pipe.fit(train, target_column="target")
    assert pre_pipe.estimator.n_features == 135
    assert pre_pipe.estimator.V is not None

    pre_v = (
        pre_pipe.estimator.V.detach().cpu().numpy()
        if hasattr(pre_pipe.estimator.V, "detach")
        else (
            pre_pipe.estimator.V.data
            if hasattr(pre_pipe.estimator.V, "data")
            else np.asarray(pre_pipe.estimator.V)
        )
    )

    # Unlabelled test query
    test_query = train.drop(columns=["target"]).iloc[:5].copy()
    pre_preds = pre_pipe.predict(test_query)

    # 3. Export genuine serving artifact bundle
    artifact_dir = tmp_path / "artifacts_wide"
    artifact_dir.mkdir(parents=True)
    context_path = artifact_dir / "training_context.parquet"
    train.to_parquet(context_path, index=False)

    preprocessing_state = pre_pipe.export_preprocessing_state()
    assert "upstream" in preprocessing_state
    assert preprocessing_state["upstream"]["pca_basis"] is not None

    manifest = {
        "format": "tabdpt-dimer-context-v3",
        "taskType": "tabular_regression",
        "targetColumn": "target",
        "dropColumns": list(preprocessing_state["dropColumns"]),
        "preprocessing": preprocessing_state,
        "baseModel": {
            "repo": TABDPT_HF_REPO,
            "revision": TABDPT_HF_REVISION,
            "filename": TABDPT_WEIGHT_FILENAME,
            "sha256": TABDPT_WEIGHT_SHA256,
            "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
        },
        "trainingContext": {
            "path": context_path.name,
            "sha256": _sha256(context_path),
        },
    }
    manifest_path = artifact_dir / "artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4. Perturb global RNG state to simulate a fresh process in a different RNG state
    import random

    random.seed(999999)
    np.random.seed(999999)
    try:
        import torch

        torch.manual_seed(999999)
    except ImportError:
        pass

    # 5. Reload artifact in fresh pipeline
    post_pipe = TabDPTRegressionPipeline.load_artifact(
        manifest_path,
        compile_model=False,
        use_flash=False,
    )

    assert post_pipe.estimator.V is not None
    post_v = (
        post_pipe.estimator.V.detach().cpu().numpy()
        if hasattr(post_pipe.estimator.V, "detach")
        else (
            post_pipe.estimator.V.data
            if hasattr(post_pipe.estimator.V, "data")
            else np.asarray(post_pipe.estimator.V)
        )
    )

    # Verify exact PCA basis parity between pre-export and post-reload
    np.testing.assert_allclose(pre_v, post_v, rtol=1e-5, atol=1e-6)

    # Verify exact continuous predictions parity
    post_preds = post_pipe.predict(test_query)
    np.testing.assert_allclose(pre_preds.to_numpy(), post_preds.to_numpy(), rtol=1e-5, atol=1e-6)


def test_artifact_reload_preserves_estimator_device_for_pca_basis(tmp_path, mock_tabdpt):
    # 1. Prepare wide training table with >128 features (135 features)
    n_samples = 20
    n_features = 135
    rng = np.random.default_rng(42)
    data = {f"f_{i}": rng.normal(size=n_samples) for i in range(n_features)}
    data["target"] = rng.normal(loc=50.0, scale=5.0, size=n_samples)
    train = pd.DataFrame(data)

    # 2. Instantiate pipeline with device=None (default DIMER runtime behavior)
    pipe = TabDPTRegressionPipeline(device=None, compile_model=False, use_flash=False)
    pipe.fit(train, target_column="target")
    assert pipe.estimator.device == "cuda"
    assert pipe.estimator.V is not None
    assert str(pipe.estimator.V.device) == "cuda"

    # 3. Export genuine serving artifact bundle
    artifact_dir = tmp_path / "artifacts_device"
    artifact_dir.mkdir(parents=True)
    context_path = artifact_dir / "training_context.parquet"
    train.to_parquet(context_path, index=False)

    preprocessing_state = pipe.export_preprocessing_state()
    assert preprocessing_state["upstream"]["pca_basis"] is not None

    manifest = {
        "format": "tabdpt-dimer-context-v3",
        "taskType": "tabular_regression",
        "targetColumn": "target",
        "dropColumns": list(preprocessing_state["dropColumns"]),
        "preprocessing": preprocessing_state,
        "baseModel": {
            "repo": TABDPT_HF_REPO,
            "revision": TABDPT_HF_REVISION,
            "filename": TABDPT_WEIGHT_FILENAME,
            "sha256": TABDPT_WEIGHT_SHA256,
            "upstreamCodeCommit": TABDPT_UPSTREAM_CODE_COMMIT,
        },
        "trainingContext": {
            "path": context_path.name,
            "sha256": _sha256(context_path),
        },
    }
    manifest_path = artifact_dir / "artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4. Reload artifact with device=None (default DIMER serving behavior)
    restored = TabDPTRegressionPipeline.load_artifact(
        manifest_path,
        device=None,
        compile_model=False,
        use_flash=False,
    )
    assert restored.device is None
    assert restored.estimator.device == "cuda"
    assert restored.estimator.V is not None
    assert str(restored.estimator.V.device) == "cuda"

    # 5. Verify continuous inference executes cleanly without CUDA vs CPU device mismatch
    test_query = train.drop(columns=["target"]).iloc[:2]
    preds = restored.predict(test_query)
    assert len(preds) == 2

    # 6. Verify that if V is forced to CPU while estimator is CUDA, predict raises RuntimeError
    restored.estimator.V.device = "cpu"
    with pytest.raises(RuntimeError, match="found at least two devices"):
        restored.predict(test_query)

