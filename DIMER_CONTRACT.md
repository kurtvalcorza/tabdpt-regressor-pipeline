# DIMER integration contract — TabDPT regressor

## Task identity

`dimer-pipeline.json` declares `taskType: tabular_regression`.

## Base model

The runtime is locked to TabDPT v1.2 / TabDPT-Turbo:

- `Layer6/TabDPT`
- revision `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- `tabdpt1_2.safetensors`
- SHA-256 `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`

`DIMER_BASE_MODEL_PATH` may provide an operator-mounted copy, but the wrapper still verifies the pinned digest.

## DIMER field-to-runtime mapping

The existing platform transport is reused exactly:

| Manifest section | Runtime channel | Consumer |
|---|---|---|
| `datasetPreprocessing` | `DIMER_PREPROCESSING_ARGS_JSON` | `tabdpt_regressor_pipeline.dimer_runtime` |
| `modelFinetuning` | `DIMER_HYPERPARAMETERS_JSON` | `tabdpt_regressor_pipeline.dimer_runtime` |

Every declared key is consumed. CI compares the manifest key sets to `SUPPORTED_PREPROCESSING_KEYS` and `SUPPORTED_HYPERPARAMETER_KEYS`; undeclared/unknown runtime keys are rejected.

`modelFinetuning` is a platform compatibility namespace, not a claim that TabDPT weights are gradient-tuned. `fine_tune` must be `false`. The remaining values control support/context selection and validation/inference.

## Dataset execution

`dimer_entrypoint.py` consumes `DIMER_DATASET_DIR`. The mounted directory must provide `train.csv` and may provide `val.csv`, either directly or inside exactly one ZIP archive.

When `val.csv` is absent, the adapter creates a deterministic random holdout using `validation_split` and `seed`. The holdout is created before `max_train_rows` caps the fitted support table, preventing validation rows from entering the model context. Categorical mappings are learned only from the resulting training support.

## Feature schema

Configured `drop_columns` are removed before the fitted schema is established and may be present in raw inference/evaluation tables. After those columns are removed, missing features and any other extra columns are rejected.

## Outputs

A successful run writes:

- `result.json` at `DIMER_RESULT_PATH` (or under `DIMER_OUTPUT_DIR` by default);
- `artifacts/training_context.csv` containing the exact capped support rows;
- `artifacts/artifact.json` containing task, model identity, runtime controls, and context digest.

The base checkpoint remains externally mounted/cached and is referenced by immutable identity rather than copied into each run output.

## Fine-tuning boundary

This v1 runtime performs in-context fitting only. Gradient fine-tuning remains out of scope and `fine_tune=true` fails closed.

## Production acceptance boundary

Repository CI proves manifest/runtime mapping, deterministic split/cap behavior, preprocessing/schema semantics, license/provenance presence, and static tutorial validity. Final acceptance still requires a real pinned checkpoint on GPU and an on-platform execution/deployment test.
