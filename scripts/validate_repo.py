#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
required = [
    "README.md", "MODEL_CARD.md", "DIMER_CONTRACT.md",
    "TABULAR_REGRESSION_DATASET_SPEC.md", "dimer-pipeline.json",
    "pyproject.toml", "tutorials/tabdpt_regressor_colab.ipynb"
]
missing = [name for name in required if not (ROOT / name).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")
manifest = json.loads((ROOT / "dimer-pipeline.json").read_text())
assert manifest["version"] == 1
assert manifest["taskType"] == "tabular_regression"
notebook = json.loads((ROOT / "tutorials/tabdpt_regressor_colab.ipynb").read_text())
assert notebook["nbformat"] == 4
assert len(notebook["cells"]) >= 5
print("TabDPT regressor repository contract: OK")
