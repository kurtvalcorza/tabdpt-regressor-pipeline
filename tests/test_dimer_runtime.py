import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tabdpt_regressor_pipeline.dimer_runtime import (
    DimerRuntimeConfig,
    SUPPORTED_HYPERPARAMETER_KEYS,
    SUPPORTED_PREPROCESSING_KEYS,
    TRANSPORT_HYPERPARAMETER_KEYS,
    load_dimer_tables,
    prepare_dimer_frames,
)


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_manifest_controls_match_runtime_contract():
    manifest = json.loads(Path("dimer-pipeline.json").read_text())
    assert set(manifest["datasetPreprocessing"]) == set(SUPPORTED_PREPROCESSING_KEYS)
    assert set(manifest["modelFinetuning"]) == set(SUPPORTED_HYPERPARAMETER_KEYS)
    assert "modelInference" not in manifest
    assert TRANSPORT_HYPERPARAMETER_KEYS == {"model_id"}
    assert "model_id" not in manifest["modelFinetuning"]


def test_runtime_accepts_transport_model_id_but_rejects_unknown_keys():
    config = DimerRuntimeConfig.from_payloads({}, {"model_id": "tabdpt-v1.2"})
    assert config == DimerRuntimeConfig()
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        DimerRuntimeConfig.from_payloads({}, {"bogus": 1})


def test_runtime_rejects_gradient_fine_tuning():
    with pytest.raises(ValueError, match="fine_tune must be false"):
        DimerRuntimeConfig.from_payloads({}, {"fine_tune": True})


def test_random_holdout_happens_before_support_cap():
    frame = pd.DataFrame({
        "x": range(500),
        "target": [float(i) for i in range(500)],
    })
    config = DimerRuntimeConfig(max_train_rows=200, validation_split=0.2, seed=7)
    train, val = prepare_dimer_frames(frame, None, config)
    assert len(train) == 200
    assert len(val) == 100
    assert set(train.index) == set(range(200))


def test_zip_rejects_archive_over_compressed_size_limit(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"train.csv": "x,target\n1,0\n2,1\n"})
    monkeypatch.setenv("DIMER_MAX_ARCHIVE_BYTES", "1")
    with pytest.raises(ValueError, match="DIMER_MAX_ARCHIVE_BYTES"):
        load_dimer_tables(dataset)


def test_zip_rejects_member_over_uncompressed_size_limit(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"train.csv": "x,target\n" + "1,0\n" * 100})
    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "32")
    with pytest.raises(ValueError, match="DIMER_MAX_MEMBER_BYTES"):
        load_dimer_tables(dataset)


def test_zip_rejects_aggregate_uncompressed_size_limit(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(
        archive,
        {
            "train.csv": "x,target\n1,0\n2,1\n",
            "notes.txt": "N" * 64,
        },
    )
    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "1024")
    monkeypatch.setenv("DIMER_MAX_UNCOMPRESSED_BYTES", "64")
    with pytest.raises(ValueError, match="DIMER_MAX_UNCOMPRESSED_BYTES"):
        load_dimer_tables(dataset)


def test_zip_rejects_suspicious_compression_ratio(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"train.csv": "x,target\n" + ("A" * 1024 + ",0\n") * 64})
    monkeypatch.setenv("DIMER_MAX_COMPRESSION_RATIO", "2")
    with pytest.raises(ValueError, match="DIMER_MAX_COMPRESSION_RATIO"):
        load_dimer_tables(dataset)


def test_zip_rejects_too_many_files(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(
        archive,
        {
            "train.csv": "x,target\n1,0\n2,1\n",
            "notes-1.txt": "a",
            "notes-2.txt": "b",
        },
    )
    monkeypatch.setenv("DIMER_MAX_DATASET_FILES", "2")
    with pytest.raises(ValueError, match="DIMER_MAX_DATASET_FILES"):
        load_dimer_tables(dataset)


def test_zip_rejects_path_traversal_member(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    archive = dataset / "dataset.zip"
    _write_zip(archive, {"../train.csv": "x,target\n1,0\n2,1\n"})
    with pytest.raises(ValueError, match="Unsafe dataset archive path"):
        load_dimer_tables(dataset)
