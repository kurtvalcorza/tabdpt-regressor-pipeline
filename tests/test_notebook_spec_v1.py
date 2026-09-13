import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"


def _load(name: str):
    return json.loads((TUTORIALS / name).read_text(encoding="utf-8"))


def _source(nb: dict, *, outside_modules: bool = False) -> str:
    """Notebook source; outside_modules=True skips the carried package cells (tagged embedded_module)."""
    chunks = []
    for cell in nb["cells"]:
        if outside_modules and cell.get("metadata", {}).get("dimer", {}).get("embedded_module"):
            continue
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def _assert_standalone_metadata(nb: dict, profile: str) -> None:
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_profile"] == profile
    assert dimer["notebook_spec"] == "1.1"
    assert dimer["standalone"] is True
    assert dimer["generated_from"]["repository"] == "tabdpt-regressor-pipeline"


def test_e2e_notebook_declares_and_exercises_release_profile():
    nb = _load("tabdpt_regressor_colab.ipynb")
    source = _source(nb)

    _assert_standalone_metadata(nb, "E2E")
    assert "DIMER E2E tabular regression tutorial (standalone)" in source
    assert "USE_BYOD" in source
    assert "training-mean baseline" in source
    assert "pipe.evaluate(" in source
    assert "pipe.predict(" in source
    assert "export_artifact_bundle(" in source
    assert "load_verified_artifact(" in source
    assert "np.testing.assert_allclose(" in source
    assert "does **not** establish" in source
    assert "smoke tutorial" not in source.lower()


def test_artifact_inference_is_external_and_never_self_produces():
    nb = _load("tabdpt_regressor_artifact_inference_colab.ipynb")
    source = _source(nb)

    _assert_standalone_metadata(nb, "ARTIFACT-INFERENCE")
    outside = _source(nb, outside_modules=True)
    assert "DIMER artifact inference tutorial (standalone)" in source
    assert "produced **outside this execution**" in source
    assert "validate_artifact_bundle(" in source
    assert "load_verified_artifact(" in source
    assert "files.upload(" in source
    assert "serving.predict(" in source  # the object reconstructed from the external artifact
    assert "to_csv(" in source
    assert "provenance" in source
    # The carried package cells define these; the companion's own cells must never call them.
    assert "load_diabetes(" not in outside
    assert "export_artifact_bundle(" not in outside
    assert ".fit(" not in outside


def test_release_notebooks_have_cleared_execution_state():
    for name in (
        "tabdpt_regressor_colab.ipynb",
        "tabdpt_regressor_artifact_inference_colab.ipynb",
    ):
        nb = _load(name)
        for cell in nb["cells"]:
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
