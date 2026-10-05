#!/usr/bin/env python3
"""Build (or ``--check``) the pinned sample artifact the ARTIFACT-INFERENCE notebook uses by default (TDRA-M1).

The sample is the serving artifact the E2E notebook exports for its default path, produced here by the E2E stage
runner's own code — ``stage_data`` and ``stage_validate`` (the public diabetes table, the seeded 80/20
split) and the package's own ``fit`` + ``export_artifact_bundle`` — with one documented substitution: the upstream
``tabdpt.TabDPTRegressor`` is replaced by ``FitTimePreprocessing``, which performs exactly the fit-time preprocessing
of ``tabdpt`` 1.2.0 (``TabDPTEstimator.fit``: a mean ``SimpleImputer`` then a ``StandardScaler`` fitted on the
support rows; no PCA basis because the 10 encoded features are within the checkpoint's feature ceiling) and loads no
model. The artifact holds support rows and fitted preprocessing state only — no model output — so the substitution
does not change its bytes; the E2E notebook's ``export`` stage prints whether its own export is byte-identical
(``matches_pinned_sample_artifact``), which a hosted run confirms.

Writes ``examples/sample-artifact/``: ``artifact.json``, ``training_context.parquet``, ``new_rows.csv`` (the first 8
holdout rows without their target, which the artifact never saw) and ``SAMPLE_ARTIFACT.json`` (digests and producer).
Needs only the CI dependencies (numpy, pandas, pyarrow, scikit-learn, huggingface-hub).

    python tools/build_sample_artifact.py           # write
    python tools/build_sample_artifact.py --check   # exit 1 unless the committed files are reproduced byte for byte
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "sample-artifact"
FILES = ("artifact.json", "training_context.parquet", "new_rows.csv")


class FitTimePreprocessing:
    """``tabdpt`` 1.2.0 ``TabDPTEstimator.fit`` preprocessing (mean imputer + standard scaler), without the model."""

    feature_reduction = "pca"

    def __init__(self, **_kwargs) -> None:
        self.V = None
        self.imputer = None
        self.scaler = None

    def fit(self, X, y):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        self.scaler.fit_transform(X)
        return self


def _load_runner():
    spec = importlib.util.spec_from_file_location("tutorial_stages", ROOT / "tools" / "tutorial_stages.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path) -> dict:
    sys.modules["tabdpt"] = types.SimpleNamespace(TabDPTRegressor=FitTimePreprocessing)  # loads no model
    stages = _load_runner()
    P = stages.package(ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        run = stages.Run(Path(tmp) / "run", Path(tmp) / "outputs", Path(tmp) / "weights", {})
        with contextlib.redirect_stdout(io.StringIO()):  # the stages' learner-facing prints are not needed here
            stages.stage_data(run)
            stages.stage_validate(run)
        train = run.read_frame("train.parquet", "sample-artifact")
        test = run.read_frame("test.parquet", "sample-artifact")
    weight = Path(tempfile.mkdtemp()) / P.TABDPT_WEIGHT_FILENAME
    weight.write_bytes(b"")
    original = P.pipeline.resolve_tabdpt_weights
    P.pipeline.resolve_tabdpt_weights = lambda *_a, **_k: weight  # no checkpoint is read: nothing is inferred
    try:
        pipe = P.TabDPTRegressionPipeline(model_weight_path=weight, compile_model=False, use_flash=False, seed=stages.SEED)
        pipe.fit(train, target_column=stages.TARGET_DEFAULT, seed=stages.SEED)
    finally:
        P.pipeline.resolve_tabdpt_weights = original
    out.mkdir(parents=True, exist_ok=True)
    P.export_artifact_bundle(pipe, train, out)
    stages._new_rows(test, stages.TARGET_DEFAULT).reset_index(names="row_id").to_csv(out / "new_rows.csv", index=False)
    record = {
        "description": "Pinned sample artifact for the ARTIFACT-INFERENCE notebook's default path: the serving artifact the E2E notebook exports on its default path (support rows + fitted preprocessing; no model output).",
        "producer": "tools/build_sample_artifact.py: the E2E stage runner's stage_data + stage_validate and the package's fit + export_artifact_bundle, with tabdpt.TabDPTRegressor replaced by the fit-time preprocessing of tabdpt 1.2.0 (mean SimpleImputer + StandardScaler; no PCA basis); no checkpoint is read",
        "e2e_cross_check": "the E2E notebook's export stage prints matches_pinned_sample_artifact for its default path",
        "data": "scikit-learn load_diabetes (Efron et al. 2004; 442 rows, 10 standardised features), seeded 80/20 split, seed 42",
        "support_rows": len(train),
        "new_rows": stages.NEW_ROWS,
        "artifact_sha256": sha256(out / "artifact.json"),
        "files": {name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)} for name in FILES},
    }
    (out / "SAMPLE_ARTIFACT.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.check:
        record = build(OUT)
        print(json.dumps({"artifact_sha256": record["artifact_sha256"], "files": record["files"]}, indent=2))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp))
        stale = [name for name in (*FILES, "SAMPLE_ARTIFACT.json") if (Path(tmp) / name).read_bytes() != (OUT / name).read_bytes()]
    if stale:
        print(f"STALE: {stale} differ from tools/build_sample_artifact.py output", file=sys.stderr)
        return 1
    print("OK: examples/sample-artifact/ is reproduced byte for byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
