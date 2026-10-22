"""The E2E notebook's closing external-artifact section (NOTEBOOK_SPEC 3.0 §19.1, XAS1–XAS7, SART1–SART11).

Offline and model-free (CI's light pins only): the pin, the digest refusal before extraction, the deterministic export
ZIP and the section's ZIP layout, the SART10 contract check, that the section's cells use only names the setup cells
define (XAS5), the `outputs/external_artifact/` path (XAS6), and the ST1 exact-URL rule.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabdpt_regressor_colab.ipynb"
PINNED_URL = "https://github.com/kurtvalcorza/tabdpt-regressor-pipeline/releases/download/sample-artifact-diabetes-v1/tabdpt_regressor_artifact.zip"
PINNED_SHA256 = "66f3d162096646300bfa560d92660314f0f62d4b2d3a5d43c3c7338f5a31d244"
SECTION_START = "## 10. Using an artifact made elsewhere"
SECTION_END = "## 11. Optional activity"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("xas_tutorial_stages", TOOLS / "tutorial_stages.py")
VALIDATOR = _load("xas_validate_release_assets", TOOLS / "validate_release_assets.py")


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cells() -> list[dict]:
    return json.loads(E2E.read_text(encoding="utf-8"))["cells"]


def _section_cells() -> list[dict]:
    cells = _cells()
    start = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_START))
    end = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_END))
    return cells[start:end]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    try:
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    except OSError:
        shutil.copytree(ROOT / "src", root / "src")
    return root


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


# ---------------------------------------------------------------- the pin (SART6, XAS2)


def test_section_pins_the_release_asset_by_url_and_digest() -> None:
    assert STAGES.SAMPLE_ARTIFACT_URL == PINNED_URL
    assert STAGES.SAMPLE_ARTIFACT_SHA256 == PINNED_SHA256
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["commit"].startswith("1e816d9")
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["release_tag"] == STAGES.SAMPLE_ARTIFACT_TAG == "sample-artifact-diabetes-v1"


def test_default_path_fetches_the_pin_and_never_this_runs_export(tmp_path, monkeypatch) -> None:
    """XAS2: the default source downloads the pinned URL at the pinned digest; the current run's export is never read."""
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    (outputs / "artifact").mkdir(parents=True)
    (outputs / STAGES.ARTIFACT_ZIP).write_bytes(b"this run's export: must never be opened here")
    seen = []

    def fake_fetch(cache, url, expected):
        seen.append((cache, url, expected))
        raise RuntimeError("stop after the fetch")

    monkeypatch.setattr(STAGES, "fetch_pinned_artifact", fake_fetch)
    run = STAGES.Run(root, outputs, tmp_path / "weights", {})
    with pytest.raises(RuntimeError, match="stop after the fetch"):
        _quiet(STAGES.stage_external_artifact, run)
    assert seen == [(root / "inputs" / "external_artifact.zip", PINNED_URL, PINNED_SHA256)]
    for fn in (STAGES.stage_external_artifact, STAGES.stage_external_reconstruct, STAGES.stage_external_predict, STAGES.external_serving, STAGES.external_sample_rows):
        body = inspect.getsource(fn)
        assert not re.search(r"ARTIFACT_ZIP", body), fn.__name__
        for main_path_name in ("artifact-reload", "export_reference", "train.parquet", "test.parquet", "data.json", "split.json", "condition.json", 'run.out / "artifact"'):
            assert main_path_name not in body, (fn.__name__, main_path_name)


# ---------------------------------------------------------------- integrity (XAS4, VER6)


def test_pinned_digest_mismatch_refuses_before_extraction(tmp_path) -> None:
    archive = _zip(tmp_path / "tampered.zip", {"artifact.json": b"{}", "training_context.parquet": b"x"})
    cache = tmp_path / "cache" / "external_artifact.zip"
    with pytest.raises(ValueError, match=r"Trusted digest mismatch .* the pinned SAMPLE_ARTIFACT_SHA256 is .*nothing was extracted"):
        STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), PINNED_SHA256)
    assert not cache.exists() and not cache.with_name(cache.name + ".part").exists()


def test_pinned_digest_match_is_kept_and_reused(tmp_path) -> None:
    archive = _zip(tmp_path / "standin.zip", {"artifact.json": b"{}"})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cache = tmp_path / "cache" / "external_artifact.zip"
    assert STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), digest) == cache
    archive.unlink()  # a second call reuses the verified cache without the network
    assert STAGES.fetch_pinned_artifact(cache, "file:///nonexistent.zip", digest) == cache


def test_own_artifact_with_a_wrong_digest_is_refused_before_extraction(tmp_path) -> None:
    root = _root(tmp_path)
    archive = _zip(tmp_path / "mine.zip", {"artifact.json": b"{}", "training_context.parquet": b"x"})
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "own", "zip_path": str(archive), "expected_zip_sha256": "0" * 64})
    with pytest.raises(ValueError, match="EXPECTED_ZIP_SHA256 is 0{64}.*nothing was extracted"):
        _quiet(STAGES.stage_external_artifact, run)
    assert not (root / "state" / "external_bundle").exists()
    assert not (tmp_path / "outputs" / STAGES.EXTERNAL_DIR).exists()
    run.options["expected_zip_sha256"] = "abc"
    with pytest.raises(ValueError, match="must be 64 hexadecimal characters"):
        _quiet(STAGES.stage_external_artifact, run)


def test_a_failed_check_removes_an_earlier_success_record(tmp_path, monkeypatch) -> None:
    """VER6: a refused artifact leaves no earlier `outputs/external_artifact/` behind to look current."""
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    (outputs / STAGES.EXTERNAL_DIR).mkdir(parents=True)
    (outputs / STAGES.EXTERNAL_DIR / f"{STAGES.STEM}_external_result.json").write_text("{}")
    monkeypatch.setattr(STAGES, "fetch_pinned_artifact", lambda *a: (_ for _ in ()).throw(ValueError("Trusted digest mismatch")))
    with pytest.raises(ValueError):
        _quiet(STAGES.stage_external_artifact, STAGES.Run(root, outputs, tmp_path / "weights", {}))
    assert not (outputs / STAGES.EXTERNAL_DIR).exists()


# ---------------------------------------------------------------- the export ZIP and the section's ZIP layout


def _artifact_dir(tmp_path: Path) -> Path:
    art = tmp_path / "artifact"
    art.mkdir()
    (art / "artifact.json").write_text('{"format": "x"}\n', encoding="utf-8")
    (art / "training_context.parquet").write_bytes(b"PAR1-stand-in")
    return art


def test_export_zip_is_deterministic_sorted_fixed_time_and_mode(tmp_path) -> None:
    art = _artifact_dir(tmp_path)
    first = STAGES.deterministic_zip(art, tmp_path / "a.zip")
    os.utime(art / "artifact.json", (1_000_000_000, 1_000_000_000))  # a different mtime must not change the ZIP
    second = STAGES.deterministic_zip(art, tmp_path / "b.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as zf:
        infos = zf.infolist()
    assert [i.filename for i in infos] == ["artifact.json", "training_context.parquet"]
    assert {i.date_time for i in infos} == {(1980, 1, 1, 0, 0, 0)}
    assert {i.external_attr >> 16 for i in infos} == {0o644}
    assert {i.compress_type for i in infos} == {zipfile.ZIP_DEFLATED}
    assert {i.create_system for i in infos} == {3}, "a fixed creator OS, so the digest does not depend on the host"
    assert not list(tmp_path.glob("*.tmp"))
    source = inspect.getsource(STAGES.stage_export)
    assert "make_archive" not in source and "deterministic_zip(art, run.out / ARTIFACT_ZIP)" in source


def test_section_extracts_exactly_the_layout_the_export_writes(tmp_path) -> None:
    """A ZIP written by the export stage passes the section's extraction unchanged (a re-pin needs no code change)."""
    archive = STAGES.deterministic_zip(_artifact_dir(tmp_path), tmp_path / STAGES.ARTIFACT_ZIP)
    out = STAGES.safe_extract_artifact(archive, tmp_path / "bundle")
    assert sorted(p.name for p in out.iterdir()) == ["artifact.json", "training_context.parquet"]
    assert (out / "training_context.parquet").read_bytes() == b"PAR1-stand-in"


@pytest.mark.parametrize(
    "members",
    [
        {"artifact.json": b"{}"},
        {"artifact.json": b"{}", "training_context.parquet": b"x", "extra.py": b"print(1)"},
        {"artifact/artifact.json": b"{}", "artifact/training_context.parquet": b"x"},
        {"../artifact.json": b"{}", "training_context.parquet": b"x"},
    ],
    ids=["missing-member", "extra-member", "nested-directory", "traversal"],
)
def test_section_refuses_any_other_zip_layout_before_writing(tmp_path, members) -> None:
    archive = _zip(tmp_path / "bad.zip", members)
    with pytest.raises(ValueError, match="nothing was extracted"):
        STAGES.safe_extract_artifact(archive, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_section_refuses_a_symlink_member(tmp_path) -> None:
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("artifact.json", b"{}")
        info = zipfile.ZipInfo("training_context.parquet")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        STAGES.safe_extract_artifact(archive, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


# ---------------------------------------------------------------- SART10: the default data contract


def test_artifact_from_another_dataset_breaks_the_data_contract() -> None:
    """A regression artifact has no label set: the contract is the ordered feature columns and the target column."""
    contract = STAGES.default_contract()
    assert contract == {"task": "regression", "featureColumns": ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"], "targetColumn": "target"}
    STAGES.check_data_contract({k: contract[k] for k in ("featureColumns", "targetColumn")}, contract)
    with pytest.raises(RuntimeError, match=r"default data contract \(SART10\)"):
        STAGES.check_data_contract({"featureColumns": ["MedInc", "HouseAge"], "targetColumn": "target"}, contract)
    with pytest.raises(RuntimeError, match=r"its featureColumns"):
        STAGES.check_data_contract({"featureColumns": list(reversed(contract["featureColumns"])), "targetColumn": "target"}, contract)
    with pytest.raises(RuntimeError, match=r"its targetColumn"):
        STAGES.check_data_contract({"featureColumns": contract["featureColumns"], "targetColumn": "progression"}, contract)


def test_sample_rows_are_the_default_holdout_rows_recomputed_without_main_path_files() -> None:
    contract = STAGES.default_contract()
    rows = STAGES.external_sample_rows(contract["featureColumns"])
    assert rows["row_id"].tolist() == [287, 211, 72, 321, 73, 418, 367, 354]
    assert list(rows.columns) == ["row_id", *contract["featureColumns"]] and "target" not in rows.columns


# ---------------------------------------------------------------- XAS5: setup cells and its own cells only


def _defined(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def test_section_cells_reference_only_setup_cell_names() -> None:
    setup = [c for c in _cells() if c["cell_type"] == "code" and c.get("metadata", {}).get("cellView") == "form"]
    assert len(setup) == 4, "the four Infrastructure cells (Sections 1-3)"
    setup_names = set().union(*(_defined(ast.parse(_src(c))) for c in setup))
    section_code = [_src(c) for c in _section_cells() if c["cell_type"] == "code"]
    assert len(section_code) == 3
    own = set().union(*(_defined(ast.parse(s)) for s in section_code))
    used = {n.id for s in section_code for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    foreign = used - own - setup_names - set(dir(builtins))
    assert not foreign, f"the section uses names defined outside the setup cells and itself: {sorted(foreign)}"
    assert {"run_stage", "ROOT", "Path"} <= used & setup_names


def test_section_states_its_dependency_and_its_difference_from_the_reload() -> None:
    md = "\n".join(_src(c) for c in _section_cells() if c["cell_type"] == "markdown")
    assert "**This section needs only the setup cells (Sections 1–3).**" in md
    assert "this run's export survives serialization; this section shows that an artifact made elsewhere can be validated and served here" in md
    assert "**Trust boundary.**" in md and "outputs/external_artifact/" in md
    assert "github.com/kurtvalcorza" not in "\n".join(_src(c) for c in _section_cells())


def test_section_defaults_run_the_pinned_artifact_and_sample_rows_without_upload(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "run_stage": lambda stage, **o: calls.append((stage, o))}
    for source in (_src(c) for c in _section_cells() if c["cell_type"] == "code"):
        exec(compile(source, "<section>", "exec"), ns)
    assert calls == [("external_artifact", {"source": "pinned"}), ("external_reconstruct", {}), ("external_predict", {"rows_path": ""})]


def test_section_runs_last_on_the_default_path_after_the_reload() -> None:
    code = [_src(c) for c in _cells() if c["cell_type"] == "code"]
    reload_at = next(i for i, s in enumerate(code) if "run_stage('reload')" in s)
    section_at = next(i for i, s in enumerate(code) if "run_stage('external_artifact'" in s)
    assert reload_at < section_at
    later = "\n".join(code[section_at + 3:])
    assert "run_stage(" not in later.replace("run_stage('activity'", ""), "only the optional activity may follow the section"
    meta = json.loads(E2E.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "3.0" and meta["notebook_kind"] == "TUTORIAL"
    assert not list((ROOT / "tutorials").glob("*_artifact_inference_colab.ipynb")), "KND11: no separate artifact-inference notebook"
    assert not (ROOT / "examples" / "sample-artifact").exists(), "SART9: the script-built sample bundle is gone"


def test_interpretation_precedes_the_section_which_closes_the_canonical_path() -> None:
    heads = [line for c in _cells() if c["cell_type"] == "markdown" for line in _src(c).split("\n") if line.startswith("## ")]
    at = {h.split(" ·")[0]: i for i, h in enumerate(heads)}
    assert at["## 9. Export the serving artifact and verify a fresh reload"] < at["## Interpretation and limits"] < at["## Conclusion"] < at[SECTION_START] < at["## 11. Optional activity: how many ensemble members does this table need?"] < at["## Troubleshooting"] < at["## References"]


# ---------------------------------------------------------------- XAS6: the outputs path


class _Serving:
    target_column = "target"

    def predict(self, X, **kw):
        return pd.Series([150.0] * len(X))


def test_predictions_and_provenance_go_to_outputs_external_artifact(tmp_path, monkeypatch) -> None:
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    run = STAGES.Run(root, outputs, tmp_path / "weights", {})
    P = STAGES.package(root)
    contract = STAGES.default_contract()
    art = {"source": "pinned", "zip": "external_artifact.zip", "zip_sha256": PINNED_SHA256, "url": PINNED_URL, "trusted_digest": "verified (pinned SAMPLE_ARTIFACT_SHA256)", **contract, "numericColumns": contract["featureColumns"], "categoricalColumns": []}
    monkeypatch.setattr(STAGES, "external_serving", lambda _run: (P, art, _Serving()))
    monkeypatch.setattr(STAGES, "runtime_identity", lambda: {"device": "stub"})
    _quiet(STAGES.stage_external_predict, run)
    assert STAGES.EXTERNAL_DIR == "external_artifact"
    written = sorted(p.relative_to(outputs).as_posix() for p in outputs.rglob("*") if p.is_file())
    assert written == ["external_artifact/tabdpt_regressor_external_predictions.csv", "external_artifact/tabdpt_regressor_external_result.json"]
    result = json.loads((outputs / "external_artifact" / "tabdpt_regressor_external_result.json").read_text())
    assert result["artifact"]["zip_sha256"] == PINNED_SHA256 and result["input"]["kind"] == "sample" and result["input"]["rows"] == 8
    assert [f["verdict"] for f in result["input_manifest"]["findings"] if f["input"] == "missing-column-probe"] == ["rejected"]
    predictions = pd.read_csv(outputs / "external_artifact" / "tabdpt_regressor_external_predictions.csv")
    assert list(predictions.columns) == ["row_id", "prediction"]
    assert predictions["row_id"].tolist() == [287, 211, 72, 321, 73, 418, 367, 354]


def test_non_finite_predictions_are_refused_before_writing(tmp_path, monkeypatch) -> None:
    root = _root(tmp_path)
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {})
    contract = STAGES.default_contract()
    art = {"source": "pinned", **contract, "numericColumns": contract["featureColumns"], "categoricalColumns": []}

    class _Bad(_Serving):
        def predict(self, X, **kw):
            return pd.Series([float("nan")] * len(X))

    monkeypatch.setattr(STAGES, "external_serving", lambda _run: (STAGES.package(root), art, _Bad()))
    with pytest.raises(RuntimeError, match="not one finite value per row"):
        _quiet(STAGES.stage_external_predict, run)
    assert not (tmp_path / "outputs" / STAGES.EXTERNAL_DIR).exists()


def test_own_artifact_needs_own_rows(tmp_path, monkeypatch) -> None:
    root = _root(tmp_path)
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {})
    contract = STAGES.default_contract()
    art = {"source": "own", **contract, "numericColumns": [], "categoricalColumns": []}
    monkeypatch.setattr(STAGES, "external_serving", lambda _run: (types.SimpleNamespace(), art, _Serving()))
    with pytest.raises(ValueError, match="set EXTERNAL_ROWS_PATH"):
        _quiet(STAGES.stage_external_predict, run)
    assert not (tmp_path / "outputs" / STAGES.EXTERNAL_DIR).exists()


# ---------------------------------------------------------------- ST1: only the exact pinned URL


_REPO = "https://github.com/kurtvalcorza/tabdpt-regressor-pipeline"


@pytest.mark.parametrize(
    "reference",
    [
        f"{_REPO}/releases/download/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/%2e%2e/%2E%2E/archive/refs/heads/main.zip",
        "https://GITHUB.COM/kurtvalcorza/tabdpt-regressor-pipeline/releases/download/%2E%2E/archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/tabdpt_regressor_artifact.zip?raw=1",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/tabdpt_regressor_artifact.zip#x",
        "https://github.com/kurtvalcorza/tabicl-classifier-pipeline/releases/download/sample-artifact-magic-v1/tabiclv2_classifier_artifact.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v2/tabdpt_regressor_artifact.zip",
        f"{_REPO}/releases/download/sample-artifact-diabetes-v1/tabdpt_regressor_artifact.zip.sig",
        f"{_REPO}.git",
        "https://GitHub.com/KurtValcorza/tabdpt-regressor-pipeline/archive/refs/heads/main.zip",
    ],
    ids=["traversal", "tag-then-traversal", "pct2e", "pct2E-uppercase-host", "query", "fragment", "other-repo-release", "other-tag", "suffix", "clone", "case"],
)
def test_st1_only_the_exact_pinned_release_asset_is_allowed(reference: str) -> None:
    source = f'SAMPLE_ARTIFACT_URL = "{reference}"\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == [reference]


def test_st1_exact_pinned_url_passes_and_matches_the_runner() -> None:
    assert VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL == PINNED_URL == STAGES.SAMPLE_ARTIFACT_URL
    assert STAGES.SAMPLE_ARTIFACT_TAG in PINNED_URL
    source = f'SAMPLE_ARTIFACT_URL = "{PINNED_URL}"  # literal\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    assert VALIDATOR.disallowed_own_references(source) == [PINNED_URL], "kernel cells allow no repository reference at all"


def test_st1_real_notebook_passes_the_validator() -> None:
    runner = VALIDATOR.carried_runner(E2E)
    assert VALIDATOR.disallowed_own_references(runner, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    VALIDATOR.validate_notebooks()
