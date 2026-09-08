# DIMER integration contract — TabDPT regressor

## Task identity

`dimer-pipeline.json` declares `taskType: tabular_regression`.

## Base model

The runtime is locked to TabDPT v1.2 / TabDPT-Turbo:

- `Layer6/TabDPT`
- revision `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- `tabdpt1_2.safetensors`
- SHA-256 `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`

`DIMER_BASE_MODEL_PATH` may provide an operator-mounted copy, but the wrapper still verifies the pinned digest. Missing files and digest mismatches fail closed and are covered by offline tests.

## DIMER field-to-runtime mapping

The existing platform transport is reused exactly:

| Manifest section | Runtime channel | Consumer |
|---|---|---|
| `datasetPreprocessing` | `DIMER_PREPROCESSING_ARGS_JSON` | `tabdpt_regressor_pipeline.dimer_runtime` |
| `modelFinetuning` | `DIMER_HYPERPARAMETERS_JSON` | `tabdpt_regressor_pipeline.dimer_runtime` |

Every declared key is consumed. CI compares the manifest key sets to `SUPPORTED_PREPROCESSING_KEYS` and `SUPPORTED_HYPERPARAMETER_KEYS`.

DIMER may additionally supply `model_id` inside `DIMER_HYPERPARAMETERS_JSON` as a **transport-only** Base Model selector. It is intentionally absent from `dimer-pipeline.json`, is accepted without changing TabDPT runtime controls, and all other undeclared preprocessing/hyperparameter keys are rejected. Classifier-only `temperature` is explicitly rejected by the regression runtime.

`modelFinetuning` is a platform compatibility namespace, not a claim that TabDPT weights are gradient-tuned. `fine_tune` must be `false`. The remaining values control support/context selection and validation/inference.

## Dataset execution

`dimer_entrypoint.py` consumes `DIMER_DATASET_DIR`. The mounted directory must provide `train.csv` and may provide `val.csv`, either directly or inside exactly one ZIP archive.

Before any CSV is loaded into pandas, the adapter enforces the standard DIMER dataset limits:

- `DIMER_MAX_ARCHIVE_BYTES` for a top-level ZIP;
- `DIMER_MAX_UNCOMPRESSED_BYTES` across dataset files/archive members;
- `DIMER_MAX_MEMBER_BYTES` per file/member;
- `DIMER_MAX_COMPRESSION_RATIO` per ZIP member;
- `DIMER_MAX_DATASET_FILES` across dataset files/archive members.

ZIP member paths are normalized and absolute/traversal/drive-qualified paths plus duplicate normalized paths are rejected. Direct-directory inputs enforce file-count, per-file, and total-byte limits before parsing. Symlinked dataset roots, files, and ZIPs are rejected so direct inputs cannot escape `DIMER_DATASET_DIR`.

When `val.csv` is absent, the adapter creates a deterministic random holdout using `validation_split` and `seed`. The holdout is created before `max_train_rows` caps the fitted support table, preventing validation rows from entering the model context. Categorical mappings are learned only from the resulting training support.

Training targets must be finite, numeric, non-missing, and non-constant. Supplied validation targets must be finite and numeric. `pipeline.evaluate()` additionally rejects a constant evaluation target because R² is undefined for that case; the run fails with a clear validation error rather than emitting a misleading R² value.

## Feature schema and persisted preprocessing

Configured `drop_columns` are removed before the fitted schema is established and may be present in raw inference/evaluation tables. After those columns are removed, missing features and any other extra columns are rejected.

`artifacts/artifact.json` uses `format: tabdpt-dimer-context-v2` and persists a versioned `preprocessing` object containing:

- fitted feature order;
- numeric/categorical column assignments;
- categorical value maps;
- missing/unseen categorical-code semantics;
- target and configured drop columns.

The state can be reconstructed with `TabularFeatureEncoder.from_state()` without re-inferring pandas dtypes from `training_context.csv`. This prevents numeric-looking string categories from silently changing semantics during a fresh-process reload.

## Outputs

A successful run writes:

- `result.json` at `DIMER_RESULT_PATH` (or under `DIMER_OUTPUT_DIR` by default);
- `artifacts/training_context.csv` containing the exact capped support rows;
- `artifacts/artifact.json` containing task/model identity, runtime controls, versioned fitted preprocessing state, and context digest.

The base checkpoint remains externally mounted/cached and is referenced by immutable identity rather than copied into each run output.

## Fine-tuning boundary

This runtime performs in-context fitting only. Gradient fine-tuning remains out of scope and `fine_tune=true` fails closed.

## Production acceptance boundary

Repository CI proves manifest/runtime mapping, transport-only `model_id` compatibility, bounded/symlink-safe dataset loading, deterministic split-before-cap behavior, target validation, preprocessing-state serialization, checkpoint-integrity guards, constant-target evaluation handling, license/provenance presence, and static tutorial validity across currently released Python 3.10–3.14. Final acceptance still requires a real pinned checkpoint on GPU and an on-platform execution/deployment test.
