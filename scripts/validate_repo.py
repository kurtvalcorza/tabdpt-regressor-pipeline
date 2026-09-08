#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tabdpt_regressor_pipeline.dimer_runtime import (
    SUPPORTED_HYPERPARAMETER_KEYS,
    SUPPORTED_PREPROCESSING_KEYS,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


required = [
    "README.md", "MODEL_CARD.md", "DIMER_CONTRACT.md", "LICENSE",
    "TABULAR_REGRESSION_DATASET_SPEC.md", "dimer-pipeline.json",
    "dimer_entrypoint.py", "pyproject.toml", "tutorials/tabdpt_regressor_colab.ipynb",
    "tutorials/tabdpt_regressor_artifact_inference_colab.ipynb", "DEPLOYMENT.md",
    "weights/README.md", "tutorials/README.md", "examples/sample-data/DATASET_CARD.md",
    "examples/build_sample_datasets.py",
]
missing = [name for name in required if not (ROOT / name).exists()]
require(not missing, f"Missing required files: {missing}")
manifest = json.loads((ROOT / "dimer-pipeline.json").read_text())
require(manifest.get("version") == 1, "dimer-pipeline.json version must be 1")
require(manifest.get("taskType") == "tabular_regression", "Unexpected taskType")
require(
    set(manifest.get("datasetPreprocessing", {})) == set(SUPPORTED_PREPROCESSING_KEYS),
    "datasetPreprocessing keys do not match runtime",
)
require(
    set(manifest.get("modelFinetuning", {})) == set(SUPPORTED_HYPERPARAMETER_KEYS),
    "modelFinetuning keys do not match runtime",
)
require("modelInference" not in manifest, "modelInference is not a supported DIMER namespace")
notebook = json.loads((ROOT / "tutorials/tabdpt_regressor_colab.ipynb").read_text())
require(notebook.get("nbformat") == 4, "Tutorial notebook must use nbformat 4")
require(len(notebook.get("cells", [])) >= 5, "Tutorial notebook must contain at least five cells")
requirements = [
    line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
require(requirements == [".[model]"], "requirements.txt must delegate to pyproject.toml via .[model]")
print("TabDPT regressor repository contract: OK")
