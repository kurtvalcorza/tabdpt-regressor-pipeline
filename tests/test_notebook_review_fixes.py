"""Regression tests for the 2026-10-05 notebook review (TDR-M1..M2, TDR-m1..m2).

The companion's findings (TDRA-*) retired with the companion notebook (NOTEBOOK_SPEC 3.0, KND11); the trust checks
they covered now live in tests/test_external_artifact_section.py.

They need only CI's dependencies (no torch, no tabdpt, no checkpoint): they exec the notebooks' own kernel cells with
stand-ins for `run_stage` and `google.colab`, run the stage runners' model-free stages (`data`, `validate`, `report`,
in-process, and check the generated notebooks statically. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabdpt_regressor_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tdr_tutorial_stages", TOOLS / "tutorial_stages.py")


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_cells(path: Path) -> list[str]:
    return [_src(c) for c in _nb(path)["cells"] if c["cell_type"] == "code"]


def _cell_with(path: Path, needle: str) -> str:
    found = [s for s in _code_cells(path) if needle in s]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # `from google.colab import files` -> ImportError


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o)), "Refused": type("Refused", (Exception,), {})}
    return ns, calls


def _run(root: Path, outputs: Path, options: dict | None = None, module=STAGES):
    src = root / "src"
    if not src.exists():
        root.mkdir(parents=True, exist_ok=True)
        src.symlink_to(ROOT / "src", target_is_directory=True)
    return module.Run(root, outputs, root / "weights", options or {})


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _diabetes() -> pd.DataFrame:
    return STAGES.load_sample_frame()


# ---------------------------------------------------------------- TDR-M1 / TDRA-M2: isolated runtime


@pytest.mark.parametrize("path", [E2E], ids=["TDR-M1"])
def test_m1_no_kernel_install_and_no_restart_instruction(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text.replace("no runtime restart", "")
    own = [s for c, s in zip(_nb(path)["cells"], [_src(c) for c in _nb(path)["cells"]], strict=True) if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
    install = _cell_with(path, "# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def _exec_check_cell(ns: dict, path: Path, **overrides) -> None:
    src = _cell_with(path, "# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)  # CI disks are small; the rule is tested, not the size
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


def test_m1_section1_is_idempotent(tmp_path, monkeypatch) -> None:
    """TDR-M1 (fleet Minor): re-running the Section 1 cell keeps the run directory, so later cells are not stranded."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, E2E)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns, E2E)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, E2E, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_m1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch) -> None:
    """TDR-M1 (fleet Minor): the venv is keyed on the lock digest; a second exec of the install cell builds nothing."""
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, E2E)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with(E2E, "# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    uv = ns["ENV_ROOT"] / f"uv-{version}"
    uv.write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "tabdpt": "x", "numpy": "x", "pandas": "x", "sklearn": "x", "cuda": True})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    builds = [c for c in commands if c[1] in ("venv", "pip")]
    assert len(builds) == 2, builds  # one venv + one install, on the first exec only
    assert ns["VENV"].name == "venv-" + ns["LOCK_SHA256"][:16]


# ---------------------------------------------------------------- TDR-M2 / TDRA-M3: guided layer

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion")


@pytest.mark.parametrize("path", [E2E], ids=["TDR-M2"])
def test_m2_guided_layer_and_collapsed_infrastructure(path: Path) -> None:
    nb = _nb(path)
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for heading in GUIDED:
        assert heading in md, heading
    assert "{{" not in md and "{MODEL_ID}" not in md and "@P:" not in md
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)
    activity = _cell_with(path, "RUN_ACTIVITY = False  # @param")
    assert "ACTIVITY_N_ENSEMBLES = 8  # @param" in activity


def test_m2_activity_writes_only_to_its_own_directory() -> None:
    """TDR-M2: the activity runs from form fields, writes under outputs/activity/ and checks canonical digests."""
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert "ACTIVITY_DIR" in body and "canonical_digests(run)" in body and "before != after" in body


@pytest.mark.parametrize("runner", ["tutorial_stages.py"])
def test_no_quality_assert_in_stage_runners(runner: str) -> None:
    tree = ast.parse((TOOLS / runner).read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]


# ---------------------------------------------------------------- TDR-m1: BYOD


def _write_byod(tmp_path: Path, name: str, frame: pd.DataFrame) -> Path:
    path = tmp_path / name
    if name.endswith(".csv"):
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("blank", [None, "", "   ", "NA"])
def test_m1_missing_targets_are_refused_before_outputs(tmp_path, blank) -> None:
    """TDR-m1 (no silent drops): missing/blank targets stop in Section 4 with the count and file lines; nothing reaches outputs/."""
    frame = _diabetes().astype({"target": object})
    frame.loc[[3, 10, 50], "target"] = blank
    path = _write_byod(tmp_path, "blank.csv", frame)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "tabdpt_regressor_result.json").write_text("{}")  # a previous run's export
    run = _run(tmp_path / "run", out, {"use_byod": True, "byod_path": str(path), "target": "target"})
    with pytest.raises(ValueError, match=r"blank\.csv: the target column 'target' has 3 missing or blank value\(s\) out of 442 rows \(file lines 5, 12, 52\)"):
        _quiet(STAGES.stage_data, run)
    assert list(out.iterdir()) == []


def test_m1_non_numeric_and_thousands_separator_targets_are_refused(tmp_path) -> None:
    """TDR-m1: a non-numeric target is refused with the count and examples; a thousands separator gets a specific hint."""
    frame = _diabetes().astype({"target": object})
    frame.loc[0, "target"] = "1,234"
    frame.loc[1, "target"] = "high"
    path = _write_byod(tmp_path, "text.csv", frame)
    run = _run(tmp_path / "run", tmp_path / "outputs", {"use_byod": True, "byod_path": str(path), "target": "target"})
    with pytest.raises(ValueError, match=r"has 2 non-numeric value\(s\) out of 442 rows, e\.g\. \['1,234', 'high'\]\. Thousands separators"):
        _quiet(STAGES.stage_data, run)


def test_m1_target_column_named_differently(tmp_path) -> None:
    """TDR-m1: TARGET is a form field; a CSV without `target` stops naming it and listing the header."""
    frame = _diabetes().rename(columns={"target": "progression"})
    path = _write_byod(tmp_path, "prog.csv", frame)
    run = _run(tmp_path / "run", tmp_path / "outputs", {"use_byod": True, "byod_path": str(path), "target": "target"})
    with pytest.raises(ValueError, match=r"prog\.csv: the target column 'target' is not in the header \['age'.*'progression'\]\. Set TARGET"):
        _quiet(STAGES.stage_data, run)
    run.options["target"] = "progression"
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    assert json.loads((run.state / "split.json").read_text())["holdoutRows"] == 89


@pytest.mark.parametrize("rows", [4, 9])
def test_m1_small_table_stops_in_section5_naming_the_minimum(tmp_path, rows) -> None:
    """TDR-m1: a four-row (or nine-row) table stops at validation, naming the minimum, before any model runs."""
    path = _write_byod(tmp_path, "small.csv", _diabetes().head(rows))
    run = _run(tmp_path / "run", tmp_path / "outputs", {"use_byod": True, "byod_path": str(path)})
    _quiet(STAGES.stage_data, run)
    with pytest.raises(ValueError, match=f"small.csv: {rows} rows is below the minimum of 10"):
        _quiet(STAGES.stage_validate, run)
    assert not (run.state / "train.parquet").exists()


def test_m1_parquet_byod_is_accepted(tmp_path) -> None:
    """TDR-m1: Parquet BYOD is read."""
    path = _write_byod(tmp_path, "good.parquet", _diabetes())
    run = _run(tmp_path / "run", tmp_path / "outputs", {"use_byod": True, "byod_path": str(path)})
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["sample_kind"] == "BYOD"


def test_m1_byod_path_field_and_upload_fallback(tmp_path, monkeypatch) -> None:
    """TDR-m1: BYOD_PATH works without google.colab; an empty path outside Colab, or a cancelled upload, gets a clear message."""
    cell = _cell_with(E2E, "USE_BYOD = False  # @param")
    ns, calls = _kernel(tmp_path)
    _no_colab(monkeypatch)
    exec(compile(_set(_set(cell, "USE_BYOD", True), "BYOD_PATH", "/data/mine.csv"), "<s4>", "exec"), ns)
    assert calls == [("data", {"use_byod": True, "byod_path": "/data/mine.csv", "target": "target", "categorical_columns": []})]
    ns, calls = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty, and the upload dialog exists only in Google Colab"):
        exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
    monkeypatch.undo()
    with _colab([{}]) as files:
        ns, calls = _kernel(tmp_path)
        with pytest.raises(ns["Refused"], match=r"^\(no file\): the upload was cancelled or empty"):  # DAT20: a refusal, not a RuntimeError
            exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
        assert files.calls == 1 and calls == []
    with _colab([{"x.csv": b"a,target\n1,y\n"}]):
        ns, calls = _kernel(tmp_path)
        exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
        assert calls[0][1]["byod_path"].endswith("inputs/x.csv")
    ns, calls = _kernel(tmp_path)
    exec(compile(cell, "<s4>", "exec"), ns)  # default path: no colab import, no upload
    assert calls == [("data", {"use_byod": False, "byod_path": "", "target": "target", "categorical_columns": []})]


# ---------------------------------------------------------------- TDR-m2: classical reference


def test_m2_report_has_linear_reference_and_range(tmp_path) -> None:
    """TDR-m2: the evaluation report carries a linear-regression reference and its split-to-split R² range."""
    run = _run(tmp_path / "run", tmp_path / "outputs")
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    frame, train, test = (run.read_frame(n, "test") for n in ("frame.parquet", "train.parquet", "test.parquet"))
    reference = STAGES.classical_reference(train, test, "target", frame)
    assert (round(reference["metrics"]["mae"], 2), round(reference["metrics"]["rmse"], 2), round(reference["metrics"]["r2"], 3)) == (42.79, 53.85, 0.453)
    spread = reference["split_to_split_r2"]
    assert (spread["splits"], round(spread["min"], 3), round(spread["max"], 3)) == (20, 0.332, 0.585)
    P = STAGES.package(run.root)
    baseline = P.training_mean_baseline(train["target"], test["target"])
    run.write_state("condition.json", {"metrics": {"mae": 40.0, "rmse": 52.0, "r2": 0.48}, "baseline": baseline, "reference": reference, "capacity": {}})
    _quiet(STAGES.stage_report, run)
    report = json.loads((run.out / "tabdpt_regressor_evaluation_report.json").read_text())
    assert [b["id"] for b in report["baselines"]] == ["training_mean", "standardised_linear_regression"]
    assert "0.332–0.585 over 20 seeded splits" in report["interpretation"]


@pytest.mark.parametrize("template", ["notebook_template.py"])
def test_stage_processes_import_neither_ipython_nor_google_colab(template: str) -> None:
    """Stages run in the isolated environment, which has neither IPython nor google.colab: only kernel cells may use them
    (the upload dialogs). There is no worker and no google.colab stub that would need a ModuleSpec (ENV15). No carried
    Python file imports IPython or google at all, so a stage cannot fail on Colab for that reason
    (tabpfn-regressor-pipeline e2d5fab / tabicl-regressor-pipeline fe7596f pattern)."""
    build = _load(f"tdr_build_notebook_{template[:-3]}", TOOLS / "build_notebook.py")
    tpl = _load(f"tdr_{template[:-3]}", TOOLS / template).TEMPLATE
    carried = [ROOT / source for dest, source in build.carried_sources(ROOT, tpl).items() if dest.endswith(".py")]
    runner = ROOT / tpl["stage_runner"]
    assert runner in carried and any(p.name == "pipeline.py" for p in carried)
    for p in carried:
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                assert not any(n.split(".")[0] in ("IPython", "google") for n in names), (p.name, node.lineno)
    text = (ROOT / "tutorials" / tpl["notebook_name"]).read_text(encoding="utf-8")
    assert "sys.modules['google" not in text and 'sys.modules[\\"google' not in text and "_WORKER_SOURCE" not in text
    assert "IPython" not in text
    # google.colab appears only in the kernel's own upload branches, imported inside them (TDR-m1).
    for cell in _code_cells(ROOT / "tutorials" / tpl["notebook_name"]):
        if "google.colab" in cell:
            assert re.search(r"^\s+from google\.colab import files$", cell, re.M), cell[:200]
            assert not re.search(r"^(from|import) google", cell, re.M), cell[:200]
