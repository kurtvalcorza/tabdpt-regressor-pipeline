"""Companion template for tools/build_notebook.py — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 1.1 §3.6, §18).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``
(the generator resolves ``--out`` to ``tutorials/<notebook_name>``). The notebook carries the same three
package modules and the same pinned snapshot as the E2E notebook; it consumes an artifact produced by a
*separate* execution (upload, or an explicit directory for non-interactive executors) and never creates one.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
from pathlib import Path

# The E2E template next to this file is the source of the shared keys (loaded by path so that the
# generator, the validator and the tests resolve it from any working directory).
_spec = importlib.util.spec_from_file_location("_e2e_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_e2e_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e_module)
BADGES, REPO, _E2E = _e2e_module.BADGES, _e2e_module.REPO, _e2e_module.TEMPLATE

TEMPLATE = {
    **{k: _E2E[k] for k in ("package", "repo_name", "pipeline_class", "weights_key", "modules", "entry_module", "model_load", "runtime_imports")},
    "stem": "tabdpt_regressor_artifact_inference",
    "notebook_name": "tabdpt_regressor_artifact_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
    "mode": "GUIDED",
    "run_all": (
        "**Known NOTEBOOK_SPEC 2.0 gap (§19, SART1/RUN5/RUN2):** the default path does not yet obtain a trusted sample artifact or sample rows automatically — with `ARTIFACT_DIR` and `NEW_DATA_PATH` empty, Sections 4 and 6 open upload dialogs for a serving artifact produced by the E2E tutorial and for unlabelled rows; an executor sets both to paths already in the runtime to skip the dialogs. Until a published sample artifact and sample rows are wired in, this notebook is a `Candidate`, not release-grade. Once they are present, **Run all** installs the pinned dependencies, validates the artifact bundle before any state is reconstructed, reconstructs the serving state with the fitted preprocessing restored (nothing is refit from inference data), validates the new rows into an input manifest, emits point predictions (no per-prediction uncertainty), reports what cannot be measured, and exports outputs — all inside this kernel, with no DIMER worker or service and no credential."
    ),
    "byod": (
        "New-input BYOD is the `NEW_DATA_PATH`/upload branch in Section 6: your own unlabelled CSV or Parquet with exactly the artifact's fitted feature columns passes through the same validation, prediction and export cells. A user-supplied artifact is the separate `ARTIFACT_DIR`/upload branch in Section 4, validated before any state is reconstructed. Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
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
    "capability": "serving-state reconstruction from an externally produced DIMER artifact (`artifact.json` + `training_context.parquet`) and point-prediction inference on genuinely new rows with the pinned `Layer6/TabDPT` v1.2 checkpoint",
    "intro": (
        "This notebook consumes `artifact.json` + `training_context.parquet` produced **outside this execution** (for "
        "example by the E2E tutorial in a separate session), validates the artifact and its exact pinned base-model "
        "provenance, restores the fitted preprocessing/support state, accepts genuinely new unlabelled data, predicts "
        "continuous point estimates, and exports results. No gradient fine-tuning "
        "occurs and **no artifact is created here**. The release-grade path restores the repository encoder plus the "
        "upstream fitted imputer/scaler/PCA state directly; it does not fit preprocessing on the uploaded artifact or on "
        "the inference rows. The carried package supplies `validate_artifact_bundle`, `load_verified_artifact`, "
        "`validate_inputs` and `evaluation_report`.\n\n"
        "**Trust boundary.** Digest and manifest checks establish internal consistency, not sender authenticity. The "
        "artifact format accepts no ZIP, pickle, or arbitrary Python-object payload: the support context is Parquet, the "
        "manifest is JSON, and the base checkpoint is acquired separately (Section 3) and digest-verified before use. "
        "Use only artifacts from a trusted producer."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable "
        "upstream checkpoint, supply an externally produced artifact and validate it before any model state is "
        "reconstructed, inspect its provenance, reconstruct the serving state through the no-refit path, validate new "
        "unlabelled rows into an input manifest, predict continuous point estimates, "
        "produce an evaluation report that is `not-measurable` because no labels exist, and export machine-readable "
        "predictions plus provenance."
    ),
    "exclusions": (
        "artifact creation, in-notebook support fitting, classification, gradient fine-tuning, or any uncertainty "
        "interval or quality claim: without labelled rows nothing is measured, and the exported predictions are "
        "point estimates with no prediction interval."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.11+; the pins were executed locally on Python 3.12). GPU recommended, CPU supported but slower; FlashAttention is disabled (`use_flash=False`) for Tesla T4 portability.",
        "- **Artifact:** an externally produced pair `artifact.json` + `training_context.parquet` (the E2E tutorial writes one to `outputs/artifact/`). Supply it through the upload dialog, or set `ARTIFACT_DIR` to a directory already present in the runtime for non-interactive execution. Nothing in this notebook manufactures it.",
        "- **Data:** one separate, unlabelled CSV or Parquet file with exactly the artifact's fitted feature columns. It is supplied by upload or by `NEW_DATA_PATH`; no sample is bundled, because scoring self-generated rows would not be external-artifact evidence. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Supply the external artifact and validate it before any model state is reconstructed\n\n"
                "Leave `ARTIFACT_DIR` empty to upload exactly `artifact.json` and `training_context.parquet`; set it to a "
                "directory already in the runtime to skip the dialog (an executor places the files there). "
                "`validate_artifact_bundle` runs **before** reconstruction and checks format/task, "
                "the support table's path/size/SHA-256, fitted-preprocessing consistency, and the manifest's "
                "base-model repository/revision/filename/digest/upstream commit against the carried package contract "
                "— a mismatch stops the notebook. Release-grade artifacts carry explicit format metadata and complete "
                "fitted upstream preprocessing state; older v3 artifacts stay loadable through a legacy compatibility "
                "path, and Section 5 fails closed when the loader reports that path was used. The cell prints the "
                "provenance a consumer needs: format, version, base model, target column, feature schema and context digest."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "ARTIFACT_DIR = ''  # @param {{type:\"string\"}}\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if ARTIFACT_DIR:\n"
                "    ART = Path(ARTIFACT_DIR)\n"
                "    artifact_source = f'directory: {{ART}}'\n"
                "else:\n"
                "    from google.colab import files\n"
                "    ART = Path('external-artifact')\n"
                "    ART.mkdir(parents=True, exist_ok=True)\n"
                "    uploaded = files.upload()\n"
                "    required = {{'artifact.json', 'training_context.parquet'}}\n"
                "    if set(uploaded) != required:\n"
                "        raise ValueError(f'Upload exactly {{sorted(required)}}; got {{sorted(uploaded)}}')\n"
                "    for name, payload in uploaded.items():\n"
                "        (ART / name).write_bytes(payload)\n"
                "    artifact_source = 'upload dialog'\n"
                "manifest_path = ART / 'artifact.json'\n"
                "manifest, context_path = validate_artifact_bundle(manifest_path)\n"
                "encoder_state = manifest['preprocessing']['encoder']\n"
                "expected = list(encoder_state['featureColumns'])\n"
                "numeric_columns = list(encoder_state['numericColumns'])\n"
                "categorical_columns = list(encoder_state['categoryMaps'].keys())\n"
                "target_column = manifest['preprocessing']['targetColumn']\n"
                "print({{'artifact_source': artifact_source, 'format': manifest['format'], 'formatVersion': manifest.get('formatVersion', 'legacy-v3-implicit'), 'artifactSemantics': manifest.get('artifactSemantics')}})\n"
                "print(json.dumps(manifest['baseModel'], indent=2))\n"
                "print({{'targetColumn': target_column, 'featureCount': len(expected), 'numericColumns': numeric_columns, 'categoricalColumns': categorical_columns, 'contextBytes': context_path.stat().st_size, 'contextSha256': manifest['trainingContext']['sha256']}})"
            ),
        },
        {
            "md": (
                "## 5. Reconstruct the serving state with fitted preprocessing restored\n\n"
                "`load_verified_artifact` re-validates the artifact, restores the serialised feature encoder, "
                "the fitted upstream mean-imputation and standardisation state and any saved PCA "
                "basis, and re-registers the support table as TabDPT context. The base checkpoint is the digest-verified "
                "file from Section 3 (`pipe.model_weight_path`), so **no network fallback** can substitute another "
                "model. This is serving-state reconstruction for in-context inference, not gradient training, and the "
                "preprocessing statistics are not fitted again — the cell fails closed if the loader reports the legacy "
                "reconditioning path or a target-column disagreement. It also surfaces the model feature ceiling and "
                "whether feature reduction is active; `context_size` limits the support rows used at prediction time."
            ),
            "code": (
                "serving = load_verified_artifact(manifest_path, model_weight_path=pipe.model_weight_path, compile_model=False, use_flash=False)\n"
                "if getattr(serving, 'preprocessing_restored_', False) is not True:\n"
                "    raise RuntimeError('This artifact used the legacy compatibility/reconditioning path. Supply a release artifact with complete fitted preprocessing state.')\n"
                "if serving.target_column != target_column:\n"
                "    raise RuntimeError('Reconstructed target column disagrees with the validated manifest.')\n"
                "encoded_feature_count = len(serving.feature_encoder.feature_columns)\n"
                "model_feature_ceiling = int(serving.estimator.max_features)\n"
                "feature_reduction = str(serving.estimator.feature_reduction)\n"
                "print({{'target': serving.target_column, 'encodedFeatureCount': encoded_feature_count, 'modelFeatureCeiling': model_feature_ceiling, 'featureReduction': feature_reduction, 'featureReductionActive': encoded_feature_count > model_feature_ceiling, 'preprocessingRestoredWithoutRefit': True, 'baseWeight': str(pipe.model_weight_path)}})"
            ),
        },
        {
            "md": (
                "## 6. Supply new unlabelled rows → validate → input manifest\n\n"
                "Leave `NEW_DATA_PATH` empty to upload one CSV or Parquet file, or set it to a file already in the "
                "runtime. The file must contain exactly the fitted feature columns printed above, with no target or "
                "pre-existing `prediction` column. For CSV, categorical columns are read explicitly as strings "
                "so values such as `01` are not silently converted to numbers; Parquet categorical columns are likewise "
                "cast. Non-numeric or infinite values in numeric columns fail clearly, before any imputation.\n\n"
                "`validate_inputs(..., target_column=None, feature_columns=...)` is the pipeline's public validation "
                "stage for inference tables: it applies exactly the schema check `predict` applies and returns an "
                "**input manifest** naming the schema, the row count and the missing-value columns; it is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a probe "
                "with one fitted column removed and records the pipeline's own error message as a finding. Missing "
                "categorical values use the fitted missing code, unseen categories the fitted unknown code, and numeric "
                "NaN is transformed by the **restored** training-fitted mean imputer; nothing is fitted on inference data."
            ),
            "code": (
                "import csv\n"
                "import io\n\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "if NEW_DATA_PATH:\n"
                "    input_name, raw = os.path.basename(NEW_DATA_PATH), Path(NEW_DATA_PATH).read_bytes()\n"
                "else:\n"
                "    new_upload = files.upload()\n"
                "    if len(new_upload) != 1:\n"
                "        raise ValueError('Upload exactly one CSV or Parquet input.')\n"
                "    input_name, raw = next(iter(new_upload.items()))\n"
                "if input_name.lower().endswith('.csv'):\n"
                "    header = next(csv.reader(io.StringIO(raw.decode('utf-8-sig'))), [])\n"
                "    duplicates = sorted({{x for x in header if header.count(x) > 1}})\n"
                "    if duplicates:\n"
                "        raise ValueError(f'Duplicate CSV columns: {{duplicates}}')\n"
                "    new_data = pd.read_csv(io.BytesIO(raw), dtype={{c: 'string' for c in categorical_columns if c in header}})\n"
                "elif input_name.lower().endswith(('.parquet', '.pq')):\n"
                "    new_data = pd.read_parquet(io.BytesIO(raw), engine='pyarrow')\n"
                "    for column in categorical_columns:\n"
                "        if column in new_data.columns:\n"
                "            new_data[column] = new_data[column].astype('string')\n"
                "else:\n"
                "    raise ValueError('Input must be CSV or Parquet.')\n"
                "if new_data.columns.duplicated().any():\n"
                "    raise ValueError('Duplicate columns are not supported.')\n"
                "reserved = [serving.target_column, 'prediction']\n"
                "present_reserved = [c for c in reserved if c in new_data]\n"
                "if present_reserved:\n"
                "    raise ValueError(f'Remove target/prediction columns before inference: {{present_reserved}}')\n"
                "print({{'ceilings': {{'MIN_DISTINCT_TARGETS': MIN_DISTINCT_TARGETS, 'modelFeatureCeiling': model_feature_ceiling}}}})\n"
                "input_manifest = validate_inputs(new_data, None, feature_columns=expected, names=[input_name])\n"
                "# Demonstrate rejection on a probe that breaks the fitted schema; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(new_data.drop(columns=[expected[0]]), None, feature_columns=expected)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'missing-column-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "new_data = new_data.loc[:, expected].copy()\n"
                "for column in numeric_columns:\n"
                "    original = new_data[column]\n"
                "    converted = pd.to_numeric(original, errors='coerce')\n"
                "    if (original.notna() & converted.isna()).any():\n"
                "        raise ValueError(f'Numeric feature {{column!r}} contains non-numeric values.')\n"
                "    finite = converted.dropna().to_numpy(dtype=float)\n"
                "    if finite.size and not np.isfinite(finite).all():\n"
                "        raise ValueError(f'Numeric feature {{column!r}} contains infinite values.')\n"
                "    new_data[column] = converted\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 7. Predict, report what cannot be measured, and export\n\n"
                "`predict()` returns **continuous point estimates only** in the target's units — no prediction interval "
                "is produced, so any tolerance band is the caller's to set on labelled data. `row_id` maps "
                "each prediction to its input row. `evaluation_report` is the pipeline's public evaluation stage and is "
                "produced even here: with no labelled rows its verdict is `not-measurable` and it states what labelled "
                "data would make the task measurable; it is written to `outputs/{stem}_evaluation_report.json`. The "
                "prediction CSV carries `row_id` and `prediction`, and the result JSON records the "
                "externally supplied artifact identity, the immutable model contract, the target column, the restored "
                "preprocessing/capacity state, the inference configuration, the input shape, the notebook's source and the "
                "runtime identity; it contains no credentials."
            ),
            "code": (
                "kw = {{'n_ensembles': 2, 'context_size': 512, 'batch_size': 512, 'seed': 42}}\n"
                "context_rows = len(pd.read_parquet(context_path, engine='pyarrow'))\n"
                "print('Requested context_size:', kw['context_size'], 'effective support rows <=', min(context_rows, kw['context_size']))\n"
                "pred = serving.predict(new_data, **kw)\n"
                "results = pd.DataFrame({{'row_id': new_data.index.to_numpy(), 'prediction': pred.to_numpy()}})\n"
                "results.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "report = evaluation_report(None, n_holdout=0, target_column=serving.target_column, sample_kind='BYOD')\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "payload = {{\n"
                "    'predictions': results.to_dict(orient='records'),\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'artifact': {{'source': artifact_source, 'format': manifest['format'], 'formatVersion': manifest.get('formatVersion'), 'baseModel': manifest['baseModel'], 'targetColumn': serving.target_column, 'trainingContextSha256': manifest['trainingContext']['sha256'], 'contextRows': context_rows}},\n"
                "    'preprocessing': {{'preprocessingRestoredWithoutRefit': True, 'encodedFeatureCount': encoded_feature_count, 'modelFeatureCeiling': model_feature_ceiling, 'featureReduction': feature_reduction, 'featureReductionActive': encoded_feature_count > model_feature_ceiling, 'numericMissingPolicy': 'restored training-fitted mean imputation', 'categoricalMissingPolicy': 'restored fitted missing code', 'unknownCategoryPolicy': 'restored fitted unknown code'}},\n"
                "    'inference': {{**kw, 'output': 'continuous point predictions in target units', 'uncertaintyInterval': None, 'effectiveSupportRowsAtMost': min(context_rows, kw['context_size'])}},\n"
                "    'input': {{'filename': input_name, 'rows': len(new_data), 'features': list(new_data.columns)}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'upstream_code_commit': TABDPT_UPSTREAM_CODE_COMMIT,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'tabdpt': importlib.metadata.version('tabdpt'), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'use_flash': False}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(results.head())\n"
                "print(json.dumps(report, indent=2))\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A successful run proves that an independently supplied release artifact is internally consistent with the "
        "carried package's pinned model contract, that the fitted preprocessing was restored "
        "without refitting, that the support context was reconstructed, and that schema-compatible new records were "
        "scored with explicit capacity/context behaviour as continuous point estimates — without the "
        "repository being reachable. It does **not** authenticate the producer or establish predictive quality, "
        "robustness, calibration, fairness, or production fitness; the evaluation report says `not-measurable` because "
        "no labels exist here, and the exported point estimates carry no uncertainty interval. Never bypass a failed "
        "manifest, digest, target-column, preprocessing-state or schema check; "
        "obtain a correct trusted artifact. If base-model acquisition fails, the only acceptable checkpoint is the exact "
        "`tabdpt1_2.safetensors` named by the inline manifest, never a substitute.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate and reconstruct an external artifact, validate the "
        "supplied inference table, execute the public prediction path and emit the shown machine-readable outputs in the "
        "tested runtime. It does **not** establish benchmark superiority, deployment calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** score rows with a deliberately unseen categorical value and inspect the fitted unknown-code "
        "policy in the manifest; compare predictions at `n_ensembles` 1 versus 4; hand a labelled copy of the same rows "
        "to `evaluate` in the E2E tutorial to obtain a `sample-sanity` report with MAE / RMSE / R².\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/README.md\n"
        f"- E2E companion (produces the artifact): https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/tabdpt_regressor_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream inference code: https://github.com/layer6ai-labs/TabDPT-inference\n"
        "- TabDPT paper: https://arxiv.org/abs/2608.01400"
    ),
}
