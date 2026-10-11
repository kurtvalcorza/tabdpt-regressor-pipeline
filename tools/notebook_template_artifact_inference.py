"""Companion template for tools/build_notebook.py /3 — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 2.2 §19, §25.13).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``. The notebook
carries the same package and pinned checkpoint manifest as the E2E notebook, its own stage runner
(``tools/tutorial_stages_artifact_inference.py``) and the pinned sample artifact of ``examples/sample-artifact/``, so
the default path reconstructs a trusted, digest-pinned artifact produced outside this execution and scores its
unlabelled sample rows with no upload. A user artifact (``ARTIFACT_DIR`` or an upload) is checked against
``EXPECTED_ARTIFACT_SHA256``. The notebook never creates an artifact.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
import json
from pathlib import Path

# The E2E template next to this file is the source of the shared keys (loaded by path so that the
# generator, the validator and the tests resolve it from any working directory).
_spec = importlib.util.spec_from_file_location("_e2e_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_e2e_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e_module)
BADGES, REPO, ENVIRONMENT, RUNTIME_PREREQ = _e2e_module.BADGES, _e2e_module.REPO, _e2e_module.ENVIRONMENT, _e2e_module.RUNTIME_PREREQ
SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "examples" / "sample-artifact" / "SAMPLE_ARTIFACT.json").read_text(encoding="utf-8"))
SAMPLE_SHA = SAMPLE["artifact_sha256"]

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabdpt_regressor_artifact_inference",
    "notebook_name": "tabdpt_regressor_artifact_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages_artifact_inference.py",
    "carried_extra": {
        "sample-artifact/artifact.json": "examples/sample-artifact/artifact.json",
        "sample-artifact/new_rows.csv": "examples/sample-artifact/new_rows.csv",
        "sample-artifact/SAMPLE_ARTIFACT.json": "examples/sample-artifact/SAMPLE_ARTIFACT.json",
    },
    "carried_binary": {"sample-artifact/training_context.parquet": "examples/sample-artifact/training_context.parquet"},
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a T4 GPU is recommended) builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage in its own process: it stages and digest-verifies the pinned TabDPT checkpoint; checks the carried **sample artifact** — the serving artifact the E2E notebook exports on its default path, pinned in the repository at `examples/sample-artifact/` — against its trusted SHA-256 `" + SAMPLE_SHA + "` and validates it before any state is reconstructed; reconstructs the serving state with the fitted preprocessing restored (nothing is refit from inference data); validates the eight carried unlabelled sample rows, which the artifact never saw, into an input manifest; predicts, reports what cannot be measured, and exports outputs. No upload dialog, repository clone, DIMER worker or service, credential, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5, §19). No hosted run of this revision has been recorded yet."
    ),
    "byod": (
        "Your own artifact is the `ARTIFACT_DIR` / `UPLOAD_ARTIFACT` branch in Section 4: set `EXPECTED_ARTIFACT_SHA256` to the digest the E2E notebook printed when it exported the artifact, and the notebook refuses any other `artifact.json` before reading it. Your own unlabelled rows are the `NEW_DATA_PATH` / `UPLOAD_NEW_DATA` branch in Section 6 (CSV or Parquet with exactly the artifact's fitted feature columns; identifier columns listed in `ID_COLUMNS` are kept out of the features and copied to the predictions). Paths work in Colab, Kaggle and Jupyter; the upload dialogs exist only in Colab. Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
    ),
    "title": "TabDPT Regressor — DIMER artifact inference tutorial (standalone)",
    "badges": [
        badge
        if badge[0] != "Open In Colab"
        else (
            badge[0],
            badge[1],
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabdpt_regressor_artifact_inference_colab.ipynb",
        )
        for badge in BADGES
    ],
    "capability": "serving-state reconstruction from an externally produced DIMER artifact (`artifact.json` + `training_context.parquet`) and point-estimate regression inference on genuinely new rows with the pinned `Layer6/TabDPT` v1.2 checkpoint",
    "intro": (
        "This notebook consumes `artifact.json` + `training_context.parquet` produced **outside this execution** — by default "
        "the pinned sample artifact, which the E2E tutorial's default path exports; or your own, from a separate E2E session "
        "— validates the artifact and its exact pinned base-model provenance, restores the fitted preprocessing/support "
        "state, accepts genuinely new unlabelled rows, predicts **continuous point estimates** (no per-prediction "
        "uncertainty interval), and exports results. No gradient fine-tuning occurs and **no artifact is created here**. The release-grade path "
        "restores the repository encoder plus the upstream fitted imputer/scaler/PCA state directly; it does not fit "
        "preprocessing on the artifact or on the inference rows. The carried package supplies `validate_artifact_bundle`, "
        "`load_verified_artifact`, `validate_inputs` and `evaluation_report`.\n\n"
        "**Trust boundary.** Digest and manifest checks establish internal consistency, not sender authenticity. A trusted "
        "digest closes most of that gap: when `EXPECTED_ARTIFACT_SHA256` is set (the default path sets it to the pinned "
        "sample's digest), an `artifact.json` with any other SHA-256 is refused before it is read, and because `artifact.json` "
        "pins the Parquet support table by size and SHA-256, one digest fixes both files. The digest is only as trustworthy as "
        "the channel it arrived through. The artifact format accepts no ZIP, pickle, or arbitrary Python-object payload: the "
        "support context is Parquet, the manifest is JSON, and the base checkpoint is acquired separately (Section 3) and "
        "digest-verified before use. Use only artifacts from a trusted producer."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** what a serving artifact for an in-context model contains, and why its support rows are part of the model (Sections 4, 5).\n"
        "2. **Verify** an artifact against a trusted digest and **distinguish** what that proves from what internal consistency checks prove (Section 4).\n"
        "3. **Diagnose** a refused artifact or input table from its message (Sections 4, 6).\n"
        "4. **Apply** the reconstructed pipeline to new unlabelled rows and read point predictions in target units, with no uncertainty interval (Section 7).\n"
        "5. **Explain** why the evaluation report says `not-measurable` here, and what data would change that (Section 7).\n"
        "6. **Predict**, run and **explain** the effect of the number of ensemble members on the same rows in an optional activity (Section 8)."
    ),
    "exclusions": (
        "artifact creation, in-notebook support fitting, classification, gradient fine-tuning, per-prediction "
        "uncertainty intervals, or any quality claim: without labelled rows nothing is measured, and the exported "
        "predictions are point estimates with no interval."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic pandas and how to read a printed Python dictionary. The E2E notebook explains in-context conditioning and the metrics; this notebook's glossary repeats the terms it uses.",
        "- **Artifact:** by default, the carried sample artifact (`examples/sample-artifact/` in the repository; produced by `tools/build_sample_artifact.py` with the E2E notebook's own data, split and export code; its SHA-256 and producer are recorded in `SAMPLE_ARTIFACT.json` and printed in Section 4). Optionally your own pair `artifact.json` + `training_context.parquet` from the E2E notebook (`outputs/artifact/`), by `ARTIFACT_DIR` or upload, with the digest that notebook printed. Nothing in this notebook manufactures an artifact.",
        "- **Data:** by default, the eight carried unlabelled sample rows (`new_rows.csv`: the first eight holdout rows of the E2E split, which the artifact's support rows do not contain), with their original `row_id`. Optionally one unlabelled CSV or Parquet file with exactly the artifact's fitted feature columns, by `NEW_DATA_PATH` or upload. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who have run, or read, the E2E TabDPT tutorial and want to see how a "
                "packaged in-context model is reused safely by someone else: checking what was received, rebuilding it without "
                "refitting, and scoring new rows. You need to be able to run notebook cells and read short Python; the glossary "
                "below explains every term.\n\n"
                "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default "
                "path needs no edit, no upload, no account, no token and no runtime restart: it uses the pinned sample artifact "
                "and sample rows carried in Section 2. Section 2 builds an isolated environment, which takes the longest; the "
                "stages after it take seconds each on a T4.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python and stops the notebook with the stage's own error message if it fails. Stages hand results "
                "to each other through files: the validated private copy of the artifact, the validated rows and JSON records.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–8) are the workflow. *Infrastructure cells* (Sections 1–3) are "
                "collapsed and titled **Infrastructure**; you may run them without studying their implementation — they exist for "
                "reproducibility and provenance.\n\n"
                "**Form controls.** `ARTIFACT_DIR`, `UPLOAD_ARTIFACT` and `EXPECTED_ARTIFACT_SHA256` (Section 4); `NEW_DATA_PATH`, "
                "`UPLOAD_NEW_DATA` and `ID_COLUMNS` (Section 6); `N_ENSEMBLES` and `CONTEXT_SIZE` (Section 7); `RUN_ACTIVITY` and "
                "`ACTIVITY_N_ENSEMBLES` (Section 8). Leave them at their defaults for the first run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 4 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Verify** | `artifact.json` + `training_context.parquet`, and a trusted digest | SHA-256 of `artifact.json`; `validate_artifact_bundle` | accepted or refused, with provenance printed |\n"
                "| **Reconstruct** | the validated artifact and the verified checkpoint | `load_verified_artifact` (no preprocessing refit) | a conditioned pipeline identical in state to the exporter's |\n"
                "| **Validate rows** | unlabelled rows with the fitted feature columns | `validate_inputs(target_column=None)` | an input manifest; refusals name the column and rule |\n"
                "| **Predict** | the validated rows | TabDPT reading the artifact's support rows | one point estimate per row in target units, a `not-measurable` report |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64, GPU and disk; a run directory | the machine |\n"
                "| 2. Carry the code, build the environment | [Engineering] | carried package, runner and sample artifact verified; isolated environment | versions |\n"
                "| 3. Pin, stage and verify the model | [Engineering] | the checkpoint at a fixed revision, digest-checked | identity and digest |\n"
                "| 4. Verify the artifact | [Engineering] | trusted digest, then `validate_artifact_bundle` | provenance, target, support rows |\n"
                "| 5. Reconstruct the serving state | [Concept] | no-refit reconstruction; capacity | restored state |\n"
                "| 6. Validate the new rows | [Evaluation practice] | input manifest and a refusal probe | the manifest |\n"
                "| 7. Predict and export | [Concept] | point predictions, `not-measurable` report, provenance | the outputs |\n"
                "| 8. Optional activity | [Concept] | change `n_ensembles` on the same rows (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | every refusal and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | what was and was not shown | your conclusion |"
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Serving artifact** | `artifact.json` (identity, fitted preprocessing, target column, digests) + `training_context.parquet` (the labelled support rows). |\n"
                "| **Support rows / context** | The labelled rows TabDPT reads at prediction time; for an in-context model they are part of the model, so the artifact carries them. |\n"
                "| **Trusted digest** | A SHA-256 you obtained from the producer through a channel you trust; here, of `artifact.json`. |\n"
                "| **Internal consistency** | The artifact's own recorded sizes and digests match its files and the pinned model; a forger who rewrites both still passes. |\n"
                "| **No-refit reconstruction** | Rebuilding the pipeline from the saved preprocessing statistics instead of fitting them again. |\n"
                "| **Legacy compatibility path** | Older artifacts without complete fitted state are reconditioned from their support rows; this notebook refuses them. |\n"
                "| **Feature ceiling (`max_features`)** | The most features the checkpoint accepts; wider tables are reduced (PCA) first. |\n"
                "| **`context_size` / `n_ensembles`** | How many support rows are read at once, and how many passes with different random views of the context are averaged. |\n"
                "| **Input manifest** | A JSON record of the validated rows: schema, row count, missing values, findings. |\n"
                "| **Point estimate** | One number per row in target units, with no interval around it. |\n"
                "| **`not-measurable`** | The evaluation verdict when no targets exist: nothing about error can be said. |\n"
                "| **ID columns** | Columns such as a record key that are carried to the outputs but never used as features. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Verify the artifact before any model state is reconstructed · [Engineering]\n\n"
                "The `artifact` stage takes the artifact from one of three places: the carried **pinned sample** (default), "
                "`ARTIFACT_DIR` (a directory already in the runtime holding `artifact.json` and `training_context.parquet`), or, in "
                "Colab, the upload dialog (`UPLOAD_ARTIFACT`). First it checks the **trusted digest**: the SHA-256 of `artifact.json` "
                "must equal `EXPECTED_ARTIFACT_SHA256`, which defaults to the pinned sample's recorded digest; for your own artifact "
                "paste the digest the E2E notebook printed in its Section 9. A mismatch stops here, naming the expected and observed "
                "digests, before anything reads the file. Without a digest, your own artifact still runs, with a printed warning "
                "that only internal consistency is checked.\n\n"
                "Then it copies the two files to a private directory and runs `validate_artifact_bundle` on the copy **before** "
                "reconstruction: format/task, the target column, the support table's path/size/SHA-256, fitted-preprocessing "
                "consistency, and the manifest's base-model repository/revision/filename/digest/upstream commit against the carried "
                "package contract. Release-grade artifacts carry explicit format metadata and complete fitted upstream "
                "preprocessing state; older v3 artifacts stay loadable through a legacy compatibility path, and Section 5 fails "
                "closed when the loader reports that path was used. The stage prints the provenance a consumer needs: source, "
                "digest status, format, version, base model, target column, feature count, the support rows' count and target "
                "summary, and the context digest.\n\n"
                "**Predict before running:** the sample artifact's support rows are the E2E notebook's training split. How many rows "
                "do you expect it to report, and roughly what target mean?"
            ),
            "code": (
                "ARTIFACT_DIR = ''  # @param {{type:\"string\"}}\n"
                "UPLOAD_ARTIFACT = False  # @param {{type:\"boolean\"}}\n"
                "EXPECTED_ARTIFACT_SHA256 = ''  # @param {{type:\"string\"}}\n\n"
                "artifact_source, artifact_dir = 'sample', ''\n"
                "if ARTIFACT_DIR:\n"
                "    artifact_source, artifact_dir = 'directory', ARTIFACT_DIR\n"
                "elif UPLOAD_ARTIFACT:\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError('UPLOAD_ARTIFACT needs the Google Colab upload dialog: elsewhere set ARTIFACT_DIR to a directory holding artifact.json and training_context.parquet.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError('The upload was cancelled or empty: no file was received. Run this cell again and choose artifact.json and training_context.parquet, or set ARTIFACT_DIR.')\n"
                "    required = {{'artifact.json', 'training_context.parquet'}}\n"
                "    if set(uploaded) != required:\n"
                "        raise ValueError(f'Upload exactly {{sorted(required)}}; got {{sorted(uploaded)}}.')\n"
                "    upload_dir = ROOT / 'inputs' / 'artifact'\n"
                "    shutil.rmtree(upload_dir, ignore_errors=True)\n"
                "    upload_dir.mkdir(parents=True)\n"
                "    for upload_name, payload in uploaded.items():\n"
                "        (upload_dir / upload_name).write_bytes(payload)\n"
                "    artifact_source, artifact_dir = 'upload', str(upload_dir)\n"
                "run_stage('artifact', source=artifact_source, artifact_dir=artifact_dir, expected_sha256=EXPECTED_ARTIFACT_SHA256)"
            ),
        },
        {
            "md": (
                "**What to notice:** `source: 'sample'`, `trusted_digest: 'verified'` and `artifact_sha256: " + SAMPLE_SHA + "`; "
                "format `tabdpt-dimer-context-v3`, `formatVersion: 3`; the base model `Layer6/TabDPT` at its pinned revision and "
                "checkpoint digest; target column `target`; 10 features; 353 support rows with their target summary; and the "
                "context digest.\n\n"
                "**Checkpoint:** someone shuffles the support targets in `training_context.parquet` and rewrites the size and SHA-256 "
                "inside `artifact.json` to match. Which check catches it — and which would not?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "`validate_artifact_bundle` would **not**: it compares the Parquet file with the size and digest recorded in "
                "`artifact.json`, and the forger rewrote those too, so the artifact is internally consistent and every prediction "
                "would change silently. The trusted digest **does**: rewriting `artifact.json` changes its SHA-256, so it no longer "
                "equals `EXPECTED_ARTIFACT_SHA256` and the stage refuses it before reading it. That only helps if the digest itself "
                "came to you by a channel the forger could not change — which is why the default path takes it from the carried, "
                "hash-verified `SAMPLE_ARTIFACT.json`, and why a user artifact without a digest gets a warning.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 5. Reconstruct the serving state with fitted preprocessing restored · [Concept]\n\n"
                "The `reconstruct` stage calls `load_verified_artifact` on the validated copy: it re-validates the artifact, "
                "restores the serialised feature encoder, the fitted upstream mean-imputation and "
                "standardisation state and any saved PCA basis, and re-registers the support table as TabDPT context. The base "
                "checkpoint is the digest-verified file from Section 3, so **no network fallback** can substitute another model. "
                "This is serving-state reconstruction for in-context inference, not gradient training, and the preprocessing "
                "statistics are not fitted again — the stage fails closed if the loader reports the legacy reconditioning path or "
                "a target-column disagreement. It also surfaces the model feature ceiling and whether feature reduction is active; "
                "`context_size` limits the support rows used at prediction time."
            ),
            "code": "run_stage('reconstruct')",
        },
        {
            "md": (
                "**What to notice:** `target: 'target'`, `encodedFeatureCount: 10`, the checkpoint's feature "
                "ceiling, `featureReductionActive: False` and `preprocessingRestoredWithoutRefit: True`."
            ),
        },
        {
            "md": (
                "## 6. Supply new unlabelled rows → validate → input manifest · [Evaluation practice]\n\n"
                "By default the `rows` stage uses the eight carried sample rows, which only fit the pinned sample artifact. For "
                "your own rows set `NEW_DATA_PATH` to a CSV or Parquet file already in the runtime, or, in Colab, tick "
                "`UPLOAD_NEW_DATA`. With your own artifact and neither set, the cell opens the upload dialog in Colab and, "
                "elsewhere, stops with a message naming `NEW_DATA_PATH`. The file must contain exactly the fitted feature columns "
                "printed above, plus any identifier columns you list in `ID_COLUMNS` (kept out of the features and copied to the "
                "predictions; the sample rows declare `row_id`). It must not contain the target or pre-existing "
                "`prediction` columns. For CSV, categorical columns are read explicitly as strings so values such as `01` "
                "are not silently converted to numbers; Parquet categorical columns are likewise cast. Non-numeric or infinite "
                "values in numeric columns fail clearly, with the count and examples, before any imputation.\n\n"
                "`validate_inputs(..., target_column=None, feature_columns=...)` is the pipeline's public validation stage for "
                "inference tables: it applies exactly the schema check `predict` applies and returns an **input manifest** "
                "naming the schema, the row count and the missing-value columns; it is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the stage also validates a probe with one "
                "fitted column removed and records the pipeline's own error message as a finding. Missing categorical values use "
                "the fitted missing code, unseen categories the fitted unknown code, and numeric NaN is transformed by the "
                "**restored** training-fitted mean imputer; nothing is fitted on inference data."
            ),
            "code": (
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "UPLOAD_NEW_DATA = False  # @param {{type:\"boolean\"}}\n"
                "ID_COLUMNS = []  # @param {{type:\"raw\"}}\n\n"
                "rows_source, rows_path = 'sample', ''\n"
                "if NEW_DATA_PATH:\n"
                "    rows_source, rows_path = 'path', NEW_DATA_PATH\n"
                "elif UPLOAD_NEW_DATA or artifact_source != 'sample':\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError('No new rows were supplied for your artifact: set NEW_DATA_PATH to a CSV or Parquet file in this runtime (the upload dialog exists only in Google Colab).') from None\n"
                "    new_upload = files.upload()\n"
                "    if not new_upload:\n"
                "        raise RuntimeError('The upload was cancelled or empty: no file was received. Run this cell again and choose one CSV or Parquet file, or set NEW_DATA_PATH.')\n"
                "    if len(new_upload) != 1:\n"
                "        raise ValueError(f'Upload exactly one CSV or Parquet input; got {{sorted(new_upload)}}.')\n"
                "    upload_name, payload = next(iter(new_upload.items()))\n"
                "    rows_file = ROOT / 'inputs' / Path(upload_name).name\n"
                "    rows_file.parent.mkdir(parents=True, exist_ok=True)\n"
                "    rows_file.write_bytes(payload)\n"
                "    rows_source, rows_path = 'upload', str(rows_file)\n"
                "run_stage('rows', source=rows_source, path=rows_path, id_columns=ID_COLUMNS)"
            ),
        },
        {
            "md": (
                "**What to notice:** the ceilings, then the input manifest: `mode: inference`, 8 rows, the 10 feature columns, "
                "`id_columns: ['row_id']`, no missing values, and one finding — the missing-column probe rejected with *Feature "
                "schema mismatch*, naming the removed column."
            ),
        },
        {
            "md": (
                "## 7. Predict, report what cannot be measured, and export · [Concept]\n\n"
                "The `predict` stage reconstructs the pipeline again from the validated copy and scores the rows. `prediction` is a "
                "**continuous point estimate** in target units; the pipeline produces no uncertainty interval, and any tolerance is "
                "the caller's to set on labelled data. The identifier columns (`row_id` for the sample rows; your `ID_COLUMNS` "
                "otherwise; the row position when there are none) map each prediction to its input row. `evaluation_report` is the "
                "pipeline's public evaluation stage and is produced even here: with no labelled rows its verdict is "
                "`not-measurable` and it states what labelled data would make the task measurable; its `sample_kind` is `sample` "
                "for the pinned sample artifact and rows and `BYOD` otherwise. It is written to "
                "`outputs/{stem}_evaluation_report.json`. The result JSON records the artifact identity and its trusted-digest "
                "status, the immutable model contract, the target column, the restored preprocessing/capacity state, the inference "
                "configuration, the input, the notebook's source and the runtime identity as a provenance record; it contains no "
                "credentials.\n\n"
                "**Predict before running:** these eight rows are the first eight holdout rows of the E2E notebook. If you ran it, "
                "will the predictions here equal the ones in its `outputs/tabdpt_regressor_predictions.csv`?"
            ),
            "code": (
                "N_ENSEMBLES = 2  # @param {{type:\"integer\"}}\n"
                "CONTEXT_SIZE = 512  # @param {{type:\"integer\"}}\n"
                "run_stage('predict', n_ensembles=N_ENSEMBLES, context_size=CONTEXT_SIZE)"
            ),
        },
        {
            "md": (
                "**What to notice:** `effective support rows <= 353`; eight rows with `row_id` 287, 211, 72, 321, 73, 418, 367 and "
                "354 and a `prediction` each; `verdict: not-measurable` and `sample_kind: sample`; and the four output files.\n\n"
                "**Checkpoint:** why is the verdict `not-measurable` even though the sample rows came from a labelled table — and "
                "what would you need to supply to get a measured result?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "This notebook receives the rows **without** their targets, as a deployed model would; with no ground truth in this "
                "execution there is nothing to compare predictions with, so any error figure would be invented. To measure, you need "
                "a labelled holdout with a finite, non-constant target, scored with `evaluate` against `training_mean_baseline` — "
                "what the E2E notebook does. On the prediction question: yes, within the reload tolerance — the sample artifact is "
                "byte-identical to the E2E default export, and the same rows, controls and seed give predictions equal within "
                "`rtol=1e-5`, `atol=1e-6` (GPU kernels can differ in the last digits).\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Optional activity: how much do more ensemble members move the predictions? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_N_ENSEMBLES = 8` instead of the "
                "canonical 2, how far will the eight predictions move, in target units, compared with the target's spread (about "
                "77)? **Change:** tick `RUN_ACTIVITY` and set `ACTIVITY_N_ENSEMBLES` (1–16). **Run** this cell. **Observe** the "
                "largest and mean absolute prediction change. **Explain** what you see. The activity scores the same rows, writes "
                "only to `outputs/activity/`, and stops if any canonical output changed."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_N_ENSEMBLES = 8  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', n_ensembles=ACTIVITY_N_ENSEMBLES)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the largest and mean absolute prediction change, and "
                "`canonical_outputs_unchanged: True`. No hosted run of the activity has been recorded, so compare with your own "
                "prediction.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "More members average out the randomness of any single pass, so the predictions move a little toward a steadier "
                "value. A change that is small next to the target's spread, and next to the model's typical error measured in the "
                "E2E notebook, does not matter for any decision; a large change on one row marks a prediction that is unstable and "
                "deserves the least trust.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel. |\n"
        "| `Not enough free disk`, `uv … mismatch`, `CalledProcessError` from `uv` | disk or network | See the E2E notebook's Troubleshooting; re-run the Section 2 install cell. Never remove a pin or a hash. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 was run with `NEW_RUN_DIRECTORY` ticked | Run Sections 1, 2 and 3 again in order, or choose *Run all*. |\n"
        "| `Trusted digest mismatch for artifact.json` | the artifact is not the one the digest was issued for (altered, re-exported, or the wrong digest pasted) | Do not proceed. Obtain the artifact and its digest from the producer again. |\n"
        "| `EXPECTED_ARTIFACT_SHA256 must be 64 hexadecimal characters` | a truncated or mistyped digest | Paste the full digest the E2E notebook printed. |\n"
        "| `No EXPECTED_ARTIFACT_SHA256 was supplied` (a warning) | your own artifact without a digest | It runs, but only internal consistency is checked; ask the producer for the digest. |\n"
        "| `ARTIFACT_DIR … is not a directory` or `The artifact needs exactly [...]` | a wrong path or a missing file | Point `ARTIFACT_DIR` at the directory holding both files. |\n"
        "| `Upload exactly ['artifact.json', 'training_context.parquet']` | one file, or a renamed file, was uploaded | Upload both files together, with their original names. |\n"
        "| `Training context size mismatch` / `SHA-256 mismatch` | the Parquet file does not match `artifact.json` | The pair is inconsistent; re-export it from the E2E notebook. |\n"
        "| `Artifact baseModel.… mismatch` | the artifact was made with another checkpoint or revision | Use an artifact from this repository's E2E notebook. |\n"
        "| `legacy compatibility/reconditioning path` | an older artifact without complete fitted state | Re-export it with the current E2E notebook. |\n"
        "| `The pinned sample rows match only the pinned sample artifact` | your own artifact with the default rows | Set `NEW_DATA_PATH` (or tick `UPLOAD_NEW_DATA` in Colab). |\n"
        "| `No new rows were supplied for your artifact` outside Colab | no rows for your artifact | Set `NEW_DATA_PATH`. |\n"
        "| `Feature schema mismatch; missing=[...], extra=[...]` | the columns differ from the artifact's | Match the fitted columns exactly; list identifier columns in `ID_COLUMNS`. |\n"
        "| `remove target/prediction columns before inference` | a labelled or already-scored file | Remove those columns; this notebook does not evaluate. |\n"
        "| `numeric feature … has N non-numeric value(s)` | text such as `n.a.` or `1,234` in a numeric column | Clean the column (leave missing values empty). |\n"
        "| `ID_COLUMNS [...] are not in the file's columns` | a typo in `ID_COLUMNS` | Use the exact column names. |\n\n"
        "## Interpretation and limits\n\n"
        "A successful run proves that a release artifact produced outside this execution is byte-identical to the one the "
        "trusted digest names (or, without a digest, internally consistent with the carried package's pinned model contract), "
        "that the fitted preprocessing was restored without refitting, that the support context was reconstructed, and that "
        "schema-compatible new records were scored with explicit capacity/context behaviour as continuous point estimates — "
        "without the repository being reachable. It does **not** authenticate the producer beyond the channel the digest came "
        "through, or establish predictive quality, robustness, calibration, fairness, or production fitness; the evaluation "
        "report says `not-measurable` because no targets exist here, and the exported point estimates carry no uncertainty "
        "interval. Never bypass a failed digest, manifest, target-column, preprocessing-state or schema check; obtain a correct "
        "trusted artifact. If base-model acquisition fails, the only acceptable checkpoint is the exact "
        "`tabdpt1_2.safetensors` named by the carried manifest, never a substitute. The pinned sample artifact was produced by "
        "`tools/build_sample_artifact.py` with the E2E notebook's own data, split and export code (it contains support rows "
        "and fitted preprocessing, no model output); a hosted E2E run confirms it byte for byte "
        "(`matches_pinned_sample_artifact`).\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, verify and reconstruct an external artifact, validate the "
        "supplied inference table, execute the public prediction path and emit the shown machine-readable outputs in the "
        "tested runtime. It does **not** establish benchmark superiority, safety for high-consequence decisions, or "
        "production fitness on an unseen domain.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three sentences, in your own words: what the trusted digest and the validation checks established about the "
        "artifact you used; what the reconstruction and the eight predictions show; and what you would need before "
        "trusting these predictions for a decision.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "The pinned sample artifact matched its trusted SHA-256 and passed every validation check, so the support rows, "
        "fitted preprocessing and model identity I used are exactly the ones the E2E default path exports, though the digest "
        "says nothing about who produced it beyond the repository it came from. The pipeline was rebuilt without refitting "
        "any preprocessing and scored eight unseen rows with point estimates; with no targets here the report is correctly "
        "`not-measurable`. Before using such predictions for a decision I would need a labelled, domain-representative test "
        "set to measure the error, and an uncertainty estimate for individual predictions.\n\n"
        "</details>\n\n"
        "**Next experiments:** run the E2E notebook, then point `ARTIFACT_DIR` at its `outputs/artifact/`, paste its printed "
        "digest into `EXPECTED_ARTIFACT_SHA256`, and score its `outputs/tabdpt_regressor_new_rows.csv` with `ID_COLUMNS = "
        "['row_id']`; paste a digest with one character changed and read the refusal; score rows with a deliberately unseen "
        "categorical value (on a BYOD artifact with a categorical column) and inspect the fitted unknown-code policy; compare "
        "predictions at `n_ensembles` 1 versus 16 with the Section 8 activity.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/README.md\n"
        f"- Pinned sample artifact: https://github.com/kurtvalcorza/{REPO}/tree/main/examples/sample-artifact\n"
        f"- E2E companion (produces artifacts): https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/tabdpt_regressor_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream inference code: https://github.com/layer6ai-labs/TabDPT-inference\n"
        "- TabDPT paper: https://arxiv.org/abs/2608.01400"
    ),
}
