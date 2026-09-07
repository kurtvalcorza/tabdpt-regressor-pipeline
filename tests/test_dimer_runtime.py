import json
import os
import zipfile
from pathlib import Path

import numpy as np
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


def test_runtime_accepts_transport_model_id_but_rejects_unknown_and_classifier_only_keys():
    config = DimerRuntimeConfig.from_payloads({}, {"model_id": "tabdpt-v1.2"})
    assert config == DimerRuntimeConfig()
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        DimerRuntimeConfig.from_payloads({}, {"bogus": 1})
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        DimerRuntimeConfig.from_payloads({}, {"temperature": 1.0})
    with pytest.raises(ValueError, match="Unsupported preprocessing"):
        DimerRuntimeConfig.from_payloads({"bogus": 1}, {})


def test_runtime_rejects_gradient_fine_tuning():
    with pytest.raises(ValueError, match="fine_tune must be false"):
        DimerRuntimeConfig.from_payloads({}, {"fine_tune": True})


def test_random_holdout_happens_before_support_cap_without_leakage():
    frame = pd.DataFrame({
        "id": range(1000),
        "target": [float(i) for i in range(1000)],
    })
    config = DimerRuntimeConfig(max_train_rows=300, validation_split=0.2, seed=7)
    train, val = prepare_dimer_frames(frame, None, config)
    assert len(train) == 300
    assert len(val) == 200
    assert set(train["id"]).isdisjoint(set(val["id"]))


def test_regression_target_validation_paths():
    config = DimerRuntimeConfig(max_train_rows=200)
    valid_val = pd.DataFrame({"x": [1, 2], "target": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must not be constant"):
        prepare_dimer_frames(pd.DataFrame({"x": [1, 2], "target": [1.0, 1.0]}), valid_val, config)
    with pytest.raises(ValueError, match="finite and numeric"):
        prepare_dimer_frames(pd.DataFrame({"x": [1, 2], "target": [1.0, "bad"]}), valid_val, config)
    with pytest.raises(ValueError, match="finite and numeric"):
        prepare_dimer_frames(pd.DataFrame({"x": [1, 2], "target": [1.0, np.inf]}), valid_val, config)
    with pytest.raises(ValueError, match="val.csv target must be finite and numeric"):
        prepare_dimer_frames(
            pd.DataFrame({"x": [1, 2], "target": [1.0, 2.0]}),
            pd.DataFrame({"x": [3, 4], "target": [3.0, "bad"]}),
            config,
        )


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


def test_zip_rejects_absolute_and_backslash_traversal_paths(tmp_path):
    for member in ["/train.csv", r"..\train.csv", r"C:\train.csv"]:
        dataset = tmp_path / member.replace("/", "_").replace("\\", "_").replace(":", "_")
        dataset.mkdir()
        _write_zip(dataset / "dataset.zip", {member: "x,target\n1,0\n2,1\n"})
        with pytest.raises(ValueError, match="Unsafe dataset archive path"):
            load_dimer_tables(dataset)


def test_zip_rejects_duplicate_normalized_paths(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_zip(
        dataset / "dataset.zip",
        {
            "a/train.csv": "x,target\n1,0\n2,1\n",
            r"a\train.csv": "x,target\n3,0\n4,1\n",
        },
    )
    with pytest.raises(ValueError, match="Duplicate normalized archive path"):
        load_dimer_tables(dataset)


def test_dataset_rejects_multiple_zips_zip_csv_mix_and_malformed_zip(tmp_path):
    multiple = tmp_path / "multiple"
    multiple.mkdir()
    _write_zip(multiple / "a.zip", {"train.csv": "x,target\n1,0\n2,1\n"})
    _write_zip(multiple / "b.zip", {"train.csv": "x,target\n1,0\n2,1\n"})
    with pytest.raises(ValueError, match="exactly one ZIP"):
        load_dimer_tables(multiple)

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_zip(mixed / "dataset.zip", {"train.csv": "x,target\n1,0\n2,1\n"})
    (mixed / "train.csv").write_text("x,target\n1,0\n2,1\n")
    with pytest.raises(ValueError, match="either CSV files or exactly one ZIP"):
        load_dimer_tables(mixed)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "dataset.zip").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Invalid ZIP archive"):
        load_dimer_tables(malformed)


def test_direct_dataset_limits_are_enforced(tmp_path, monkeypatch):
    member = tmp_path / "member"
    member.mkdir()
    (member / "train.csv").write_text("x,target\n" + "1,0\n" * 20)
    monkeypatch.setenv("DIMER_MAX_MEMBER_BYTES", "16")
    with pytest.raises(ValueError, match="DIMER_MAX_MEMBER_BYTES"):
        load_dimer_tables(member)

    monkeypatch.delenv("DIMER_MAX_MEMBER_BYTES")
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    (aggregate / "train.csv").write_text("x,target\n1,0\n2,1\n")
    (aggregate / "notes.txt").write_text("N" * 64)
    monkeypatch.setenv("DIMER_MAX_UNCOMPRESSED_BYTES", "64")
    with pytest.raises(ValueError, match="DIMER_MAX_UNCOMPRESSED_BYTES"):
        load_dimer_tables(aggregate)

    monkeypatch.delenv("DIMER_MAX_UNCOMPRESSED_BYTES")
    count = tmp_path / "count"
    count.mkdir()
    (count / "train.csv").write_text("x,target\n1,0\n2,1\n")
    (count / "a.txt").write_text("a")
    (count / "b.txt").write_text("b")
    monkeypatch.setenv("DIMER_MAX_DATASET_FILES", "2")
    with pytest.raises(ValueError, match="DIMER_MAX_DATASET_FILES"):
        load_dimer_tables(count)


def test_direct_dataset_rejects_symlink_escape(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    outside = tmp_path / "outside.csv"
    outside.write_text("x,target\n1,0\n2,1\n")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    link = dataset / "train.csv"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation not permitted")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        load_dimer_tables(dataset)
