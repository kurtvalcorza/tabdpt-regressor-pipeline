import json
from pathlib import Path

from tabdpt_regressor_pipeline import TABDPT_HF_REVISION, TABDPT_WEIGHT_SHA256


def test_pinned_model_identity():
    assert TABDPT_HF_REVISION == "4462ffbd1d8dea25d4862d30beed4b70cd596ae5"
    assert TABDPT_WEIGHT_SHA256 == "06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"


def test_dimer_manifest_is_regression():
    manifest = json.loads(Path("dimer-pipeline.json").read_text())
    assert manifest["version"] == 1
    assert manifest["taskType"] == "tabular_regression"
    assert manifest["modelInference"]["context_size"]["default"] <= manifest["datasetPreprocessing"]["max_train_rows"]["default"]
