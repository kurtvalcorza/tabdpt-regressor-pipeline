import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"


def _load(name: str):
    return json.loads((TUTORIALS / name).read_text(encoding="utf-8"))


def _source(nb: dict) -> str:
    chunks = []
    for cell in nb["cells"]:
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def test_e2e_notebook_declares_and_exercises_release_profile():
    nb = _load("tabdpt_regressor_colab.ipynb")
    source = _source(nb)

    assert nb["metadata"]["dimer"] == {"notebook_profile": "E2E", "notebook_spec": "1.0"}
    assert "DIMER E2E tutorial" in source
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

    assert nb["metadata"]["dimer"] == {
        "notebook_profile": "ARTIFACT-INFERENCE",
        "notebook_spec": "1.0",
    }
    assert "DIMER artifact inference tutorial" in source
    assert "produced **outside this execution**" in source
    assert "validate_artifact_bundle(" in source
    assert "load_verified_artifact(" in source
    assert "files.upload(" in source
    assert "pipe.predict(" in source
    assert "to_csv(" in source
    assert "provenance" in source
    assert "load_diabetes(" not in source
    assert "export_artifact_bundle(" not in source
    assert ".fit(" not in source


def test_release_notebooks_have_cleared_execution_state():
    for name in (
        "tabdpt_regressor_colab.ipynb",
        "tabdpt_regressor_artifact_inference_colab.ipynb",
    ):
        nb = _load(name)
        for cell in nb["cells"]:
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
