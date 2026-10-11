"""Stage runner for the standalone TabDPT regressor ARTIFACT-INFERENCE tutorial (NOTEBOOK_SPEC 2.2 §25.13 pattern).

The companion notebook carries this file verbatim as ``tutorial_stages.py`` (beside the carried package under
``src/`` and the pinned sample artifact under ``sample-artifact/``) and runs every stage with the interpreter of an
isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage artifact --options '{...}'

Nothing is installed into the notebook kernel and no artifact is created here. The default path uses the pinned
sample artifact (``examples/sample-artifact/`` in the repository: the serving artifact the E2E notebook exports on
its default path) and its eight unlabelled rows, so **Run all** needs no upload. A user artifact (``ARTIFACT_DIR`` or
the upload dialog) is checked against ``EXPECTED_ARTIFACT_SHA256`` — the digest the E2E notebook prints when it
exports — before anything else reads it. On failure a stage writes ``RUN_DIR/state/<stage>.error.json``, which the
notebook re-raises in the kernel.

Stages: weights → artifact → reconstruct → rows → predict, plus the optional ``activity``. The ``artifact`` and
``rows`` stages import no model library, so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "tabdpt_regressor_artifact_inference"
PACKAGE = "tabdpt_regressor_pipeline"
SEED = 42
BATCH_SIZE = 512
SAMPLE_DIR = "sample-artifact"
SAMPLE_RECORD = "sample-artifact/SAMPLE_ARTIFACT.json"
SAMPLE_ROWS = "sample-artifact/new_rows.csv"
SAMPLE_ID_COLUMNS = ["row_id"]
ARTIFACT_FILES = ("artifact.json", "training_context.parquet")
CONTROL_LIMITS = {"n_ensembles": (1, 16), "context_size": (16, 4096)}
ACTIVITY_DIR = "activity"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    P = package(run.root)
    snapshot = run.weights / P.MODEL_KEY
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified_files": len(verified["files"]), "revision": verified["revision"], "sha256": [f["sha256"] for f in verified["files"]]})
    run.write_state("weights.json", {"snapshot": str(snapshot), "weight": str(snapshot / P.TABDPT_WEIGHT_FILENAME)})


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def check_trusted_digest(manifest_path: Path, expected: str, *, field: str = "EXPECTED_ARTIFACT_SHA256") -> str:
    """Refuse an artifact whose ``artifact.json`` SHA-256 is not the trusted digest (TDCA-m2). ``artifact.json`` pins
    ``training_context.parquet`` by size and SHA-256, which validation then checks, so one digest fixes both files."""
    expected = expected.strip().lower()
    if not SHA256_HEX.match(expected):
        raise ValueError(f"{field} must be 64 hexadecimal characters (the digest the E2E notebook printed when it exported the artifact), got {expected!r}.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} is missing: the artifact directory must hold {list(ARTIFACT_FILES)}.")
    observed = sha256_file(manifest_path)
    if observed != expected:
        raise ValueError(
            f"Trusted digest mismatch for {manifest_path.name}: {field} is {expected}, the supplied file's SHA-256 is {observed}. "
            "This is not the artifact the digest was issued for; nothing was reconstructed. Obtain the artifact and its digest from the producer again."
        )
    return observed


def stage_artifact(run: Run) -> None:
    P = package(run.root)
    opts = run.options
    source = opts.get("source", "sample")
    expected = str(opts.get("expected_sha256") or "")
    removed = clear_outputs(run)
    for name in ("artifact.json", "rows.json", "rows.parquet", "reconstruct.json"):
        (run.state / name).unlink(missing_ok=True)
    if removed:
        print({"removed_previous_outputs": removed})
    if source == "sample":
        directory = run.root / SAMPLE_DIR
        pinned = json.loads((run.root / SAMPLE_RECORD).read_text(encoding="utf-8"))
        expected = expected or pinned["artifact_sha256"]
        provenance = {"producer": pinned["producer"], "data": pinned["data"], "e2e_cross_check": pinned["e2e_cross_check"]}
    elif source in ("directory", "upload"):
        directory = Path(opts.get("artifact_dir") or "")
        if not str(directory) or not directory.is_dir():
            raise FileNotFoundError(f"ARTIFACT_DIR {str(directory)!r} is not a directory in this runtime: it must hold {list(ARTIFACT_FILES)}.")
        present = sorted(p.name for p in directory.iterdir() if p.is_file())
        missing = [name for name in ARTIFACT_FILES if name not in present]
        if missing:
            raise ValueError(f"The artifact needs exactly {list(ARTIFACT_FILES)}; {directory} is missing {missing} (found {present}).")
        provenance = {"producer": "supplied by the user (" + source + ")"}
    else:
        raise ValueError(f"unknown artifact source {source!r}")
    manifest_path = directory / "artifact.json"
    if expected:
        digest = check_trusted_digest(manifest_path, expected)
        trust = {"trusted_digest": "verified", "artifact_sha256": digest}
    else:
        digest = sha256_file(manifest_path)
        trust = {"trusted_digest": "not supplied", "artifact_sha256": digest}
        print("No EXPECTED_ARTIFACT_SHA256 was supplied: the checks below establish internal consistency only, not that this is the artifact you were sent. Ask the producer for the digest the E2E notebook printed.")
    # Validate, then work only on a private copy, so the files reconstructed later are the files validated now.
    copy = run.state / "artifact"
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    for name in ARTIFACT_FILES:
        shutil.copyfile(directory / name, copy / name)
    manifest, context_path = P.validate_artifact_bundle(copy / "artifact.json")
    if sha256_file(copy / "artifact.json") != digest:
        raise RuntimeError("artifact.json changed while it was being copied; supply it again.")
    import pandas as pd

    context = pd.read_parquet(context_path, engine="pyarrow")
    encoder_state = manifest["preprocessing"]["encoder"]
    target = manifest["targetColumn"]
    record = {
        "source": source,
        "directory": str(directory),
        **trust,
        **provenance,
        "format": manifest["format"],
        "formatVersion": manifest.get("formatVersion", "legacy-v3-implicit"),
        "artifactSemantics": manifest.get("artifactSemantics"),
        "targetColumn": target,
        "featureColumns": list(encoder_state["featureColumns"]),
        "numericColumns": list(encoder_state["numericColumns"]),
        "categoricalColumns": list(encoder_state["categoryMaps"].keys()),
        "contextRows": len(context),
        "contextTargetSummary": {k: round(float(v), 3) for k, v in context[target].describe().items() if k in ("mean", "std", "min", "max")},
        "contextBytes": context_path.stat().st_size,
        "contextSha256": manifest["trainingContext"]["sha256"],
        "baseModel": manifest["baseModel"],
    }
    run.write_state("artifact.json", record)
    print({k: record[k] for k in ("source", "trusted_digest", "artifact_sha256", "format", "formatVersion", "artifactSemantics")})
    print(json.dumps(manifest["baseModel"], indent=2))
    print({"targetColumn": target, "featureCount": len(record["featureColumns"]), "categoricalColumns": record["categoricalColumns"], "contextRows": record["contextRows"], "contextTargetSummary": record["contextTargetSummary"], "contextBytes": record["contextBytes"], "contextSha256": record["contextSha256"]})


def serving_pipeline(run: Run):
    P = package(run.root)
    weights = run.read_state("weights.json", "reconstruct")
    serving = P.load_verified_artifact(run.state / "artifact" / "artifact.json", model_weight_path=weights["weight"], compile_model=False, use_flash=False)
    if getattr(serving, "preprocessing_restored_", False) is not True:
        raise RuntimeError("This artifact used the legacy compatibility/reconditioning path. Supply a release artifact with complete fitted preprocessing state.")
    return serving


def stage_reconstruct(run: Run) -> None:
    art = run.read_state("artifact.json", "reconstruct")
    serving = serving_pipeline(run)
    target_column = art["targetColumn"]
    if serving.target_column != target_column:
        raise RuntimeError(f"Reconstructed target column {serving.target_column!r} disagrees with the validated manifest {target_column!r}.")
    encoded_feature_count = len(serving.feature_encoder.feature_columns)
    model_feature_ceiling = int(serving.estimator.max_features)
    feature_reduction = str(serving.estimator.feature_reduction)
    capacity = {"target": serving.target_column, "encodedFeatureCount": encoded_feature_count, "modelFeatureCeiling": model_feature_ceiling, "featureReduction": feature_reduction, "featureReductionActive": encoded_feature_count > model_feature_ceiling, "preprocessingRestoredWithoutRefit": True, "baseWeight": str(serving.model_weight_path)}
    run.write_state("reconstruct.json", capacity)
    print(capacity)


def read_rows(path: Path, categorical_columns: list[str]):
    import pandas as pd

    name = path.name
    if not path.is_file():
        raise FileNotFoundError(f"NEW_DATA_PATH {str(path)!r} does not exist in this runtime: upload the file or correct the path.")
    raw = path.read_bytes()
    if name.lower().endswith(".csv"):
        try:
            header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), [])
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name}: the CSV must be UTF-8 text ({exc}).") from None
        duplicates = sorted({x for x in header if header.count(x) > 1})
        if duplicates:
            raise ValueError(f"{name}: duplicate CSV columns {duplicates}; every column name must be unique.")
        frame = pd.read_csv(io.BytesIO(raw), dtype={c: "string" for c in categorical_columns if c in header})
    elif name.lower().endswith((".parquet", ".pq")):
        frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
        for column in categorical_columns:
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
    else:
        raise ValueError(f"{name}: input must be one CSV (.csv) or Parquet (.parquet/.pq) file.")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name}: duplicate columns are not supported.")
    if frame.empty:
        raise ValueError(f"{name}: the table has no data rows.")
    return frame


def stage_rows(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    art = run.read_state("artifact.json", "rows")
    opts = run.options
    source = opts.get("source", "sample")
    if source == "sample":
        if art["source"] != "sample":
            raise ValueError("The pinned sample rows match only the pinned sample artifact. With your own artifact, set NEW_DATA_PATH to your rows (or, in Colab, tick UPLOAD_NEW_DATA).")
        path, id_columns = run.root / SAMPLE_ROWS, list(SAMPLE_ID_COLUMNS)
    else:
        path = Path(opts.get("path") or "")
        if not str(path):
            raise ValueError("No new rows were supplied: set NEW_DATA_PATH to a CSV or Parquet file in this runtime.")
        id_columns = opts.get("id_columns") or []
        if not isinstance(id_columns, list) or not all(isinstance(c, str) for c in id_columns):
            raise ValueError(f"ID_COLUMNS must be a list of column names, got {id_columns!r}.")
    expected = art["featureColumns"]
    numeric_columns = art["numericColumns"]
    new_data = read_rows(path, art["categoricalColumns"])
    name = path.name
    absent_ids = [c for c in id_columns if c not in new_data.columns]
    if absent_ids:
        raise ValueError(f"{name}: ID_COLUMNS {absent_ids} are not in the file's columns {list(new_data.columns)}.")
    feature_ids = [c for c in id_columns if c in expected]
    if feature_ids:
        raise ValueError(f"{name}: {feature_ids} are fitted feature columns of the artifact and cannot be ID_COLUMNS.")
    reserved = [art["targetColumn"], "prediction"]
    present_reserved = [c for c in reserved if c in new_data.columns]
    if present_reserved:
        raise ValueError(f"{name}: remove target/prediction columns before inference: {present_reserved}")
    ids = new_data.loc[:, id_columns].copy()
    features = new_data.drop(columns=id_columns)
    print({"ceilings": {"MIN_DISTINCT_TARGETS": P.MIN_DISTINCT_TARGETS, "featureColumns": len(expected)}})
    try:
        input_manifest = P.validate_inputs(features, None, feature_columns=expected, names=[name])
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}. The file must hold exactly the artifact's fitted feature columns; list identifier columns in ID_COLUMNS.") from None
    input_manifest["id_columns"] = id_columns
    # Demonstrate rejection on a probe that breaks the fitted schema; the finding is recorded, not swallowed.
    try:
        P.validate_inputs(features.drop(columns=[expected[0]]), None, feature_columns=expected)
    except ValueError as exc:
        input_manifest["findings"].append({"input": "missing-column-probe", "verdict": "rejected", "message": str(exc)})
    features = features.loc[:, expected].copy()
    import pandas as pd

    for column in numeric_columns:
        original = features[column]
        converted = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & converted.isna()
        if bad.any():
            examples = [str(v) for v in original[bad].head(3).tolist()]
            raise ValueError(f"{name}: numeric feature {column!r} has {int(bad.sum())} non-numeric value(s), e.g. {examples}.")
        finite = converted.dropna().to_numpy(dtype=float)
        if finite.size and not np.isfinite(finite).all():
            raise ValueError(f"{name}: numeric feature {column!r} contains infinite values.")
        features[column] = converted
    run.write_output(f"{STEM}_input_manifest.json", input_manifest)
    table = pd.concat([ids.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    table.to_parquet(run.state / "rows.parquet", index=False, engine="pyarrow")
    sample_kind = "sample" if source == "sample" and art["source"] == "sample" else "BYOD"
    run.write_state("rows.json", {"source": source, "name": name, "rows": len(features), "id_columns": id_columns, "sample_kind": sample_kind})
    print(json.dumps(input_manifest, indent=2))


def _controls(opts: dict[str, Any]) -> dict[str, int]:
    controls = {}
    for key, default in (("n_ensembles", 2), ("context_size", 512)):
        value = opts.get(key, default)
        low, high = CONTROL_LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key.upper()} must be an integer from {low} to {high}, got {value!r}.")
        controls[key] = value
    return {**controls, "batch_size": BATCH_SIZE, "seed": SEED}


def runtime_identity() -> dict[str, Any]:
    import importlib.metadata
    import platform

    import numpy
    import pandas
    import sklearn
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__, "tabdpt": importlib.metadata.version("tabdpt"), "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "use_flash": False}


def score(run: Run, kw: dict[str, int]):
    import pandas as pd

    art = run.read_state("artifact.json", "predict")
    rows = run.read_state("rows.json", "predict")
    table = pd.read_parquet(run.state / "rows.parquet", engine="pyarrow")
    features = table.drop(columns=rows["id_columns"])
    serving = serving_pipeline(run)
    if serving.target_column != art["targetColumn"]:
        raise RuntimeError("Reconstructed target column disagrees with the validated manifest.")
    pred = serving.predict(features, **kw)
    results = table.loc[:, rows["id_columns"]].copy() if rows["id_columns"] else pd.DataFrame({"row_id": range(len(features))})
    results["prediction"] = pred.to_numpy()
    return art, rows, serving, results


def stage_predict(run: Run) -> None:
    P = package(run.root)
    kw = _controls(run.options)
    art, rows, serving, results = score(run, kw)
    cap = run.read_state("reconstruct.json", "predict")
    print("Requested context_size:", kw["context_size"], "effective support rows <=", min(art["contextRows"], kw["context_size"]))
    results.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    report = P.evaluation_report(None, n_holdout=0, target_column=serving.target_column, sample_kind=rows["sample_kind"])
    run.write_output(f"{STEM}_evaluation_report.json", report)
    input_manifest = json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload = {
        "predictions": results.to_dict(orient="records"),
        "evaluation_report": report,
        "input_manifest": input_manifest,
        "artifact": {k: art[k] for k in ("source", "trusted_digest", "artifact_sha256", "format", "formatVersion", "baseModel", "targetColumn", "contextSha256", "contextRows")},
        "preprocessing": {"preprocessingRestoredWithoutRefit": True, "encodedFeatureCount": cap["encodedFeatureCount"], "modelFeatureCeiling": cap["modelFeatureCeiling"], "featureReduction": cap["featureReduction"], "featureReductionActive": cap["featureReductionActive"], "numericMissingPolicy": "restored training-fitted mean imputation", "categoricalMissingPolicy": "restored fitted missing code", "unknownCategoryPolicy": "restored fitted unknown code"},
        "inference": {**kw, "output": "continuous point predictions in target units", "uncertaintyInterval": None, "effectiveSupportRowsAtMost": min(art["contextRows"], kw["context_size"])},
        "input": {"filename": rows["name"], "source": rows["source"], "rows": rows["rows"], "id_columns": rows["id_columns"], "features": art["featureColumns"]},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "upstream_code_commit": P.TABDPT_UPSTREAM_CODE_COMMIT,
        "runtime": runtime_identity(),
    }
    run.write_output(f"{STEM}_result.json", payload)
    run.write_state("predict.json", {"controls": kw})
    print(results.round(4).to_string(index=False))
    print(json.dumps({k: report[k] for k in ("verdict", "sample_kind", "reason")}, indent=2))
    print(sorted(p.name for p in run.out.iterdir() if p.name.startswith(STEM)))


def canonical_digests(run: Run) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}


def stage_activity(run: Run) -> None:
    """Optional activity: score the same rows with a different number of ensemble members; write to outputs/activity/
    and prove the canonical outputs did not change."""
    import numpy as np

    canonical = run.read_state("predict.json", "activity")["controls"]
    kw = _controls({"n_ensembles": run.options.get("n_ensembles", 8), "context_size": canonical["context_size"]})
    before = canonical_digests(run)
    import pandas as pd

    base = pd.read_csv(run.out / f"{STEM}_predictions.csv")
    _, _, _, results = score(run, kw)
    diff = np.abs(base["prediction"].to_numpy(dtype=float) - results["prediction"].to_numpy(dtype=float))
    record = {"changed": {"n_ensembles": [canonical["n_ensembles"], kw["n_ensembles"]]}, "rows": len(results), "max_abs_prediction_change": float(diff.max()), "mean_abs_prediction_change": float(diff.mean())}
    out = run.out / ACTIVITY_DIR / f"{STEM}_activity_n_ensembles_{kw['n_ensembles']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    run.write_output(f"{ACTIVITY_DIR}/{STEM}_activity_n_ensembles_{kw['n_ensembles']}.json", record)
    if canonical_digests(run) != before:
        raise RuntimeError("The activity changed a canonical output; it must write only to outputs/activity/.")
    print(rounded(record))
    print({"canonical_outputs_unchanged": True})


STAGES = {
    "weights": stage_weights,
    "artifact": stage_artifact,
    "reconstruct": stage_reconstruct,
    "rows": stage_rows,
    "predict": stage_predict,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
