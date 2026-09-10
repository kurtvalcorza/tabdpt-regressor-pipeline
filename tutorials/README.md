# TabDPT Regressor Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabdpt-regressor-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat)](https://huggingface.co/Layer6/TabDPT)
[![Upstream](https://img.shields.io/badge/Upstream-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white)](https://github.com/layer6ai-labs/TabDPT-inference)
[![arXiv](https://img.shields.io/badge/arXiv-2608.01400-b31b1b.svg)](https://arxiv.org/abs/2608.01400)

**DIMER Notebook Specification:** 1.0 (2026-09-10)

These notebooks exercise the repository's supported TabDPT regression API and DIMER serving-artifact contract. Static validation is enforced in CI. A notebook is promoted from release candidate to release-grade only after clean-runtime execution evidence is recorded for the exact release revision.

## Notebook registry

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| [`tabdpt_regressor_colab.ipynb`](tabdpt_regressor_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-regressor-pipeline/blob/main/tutorials/tabdpt_regressor_colab.ipynb) | `E2E` | Support-context conditioning, leakage-aware regression evaluation/baseline, preprocessing/capacity reporting, new-data inference, DIMER artifact export, no-refit fresh reload/equivalence | Python 3.11+; GPU recommended; CPU supported; `use_flash=False` | **release candidate** — clean-runtime execution evidence required |
| [`tabdpt_regressor_artifact_inference_colab.ipynb`](tabdpt_regressor_artifact_inference_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-regressor-pipeline/blob/main/tutorials/tabdpt_regressor_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | Validate externally supplied artifact/model provenance, restore fitted preprocessing/support state without refit, validate/score new data, report capacity/context, export predictions/provenance | Python 3.11+; GPU recommended; CPU supported; `use_flash=False` | **release candidate** — external-artifact clean-runtime evidence required |

## Runtime and model contract

- The tutorial lock set requires **Python 3.11+**. This matches the upstream v1.2.0 reproduction environment's `numpy==2.3.0` pin for Python >=3.11.
- Install the exact tutorial dependency set from [`requirements-colab.txt`](requirements-colab.txt) before importing model libraries.
- TabDPT package: `tabdpt==1.2.0`.
- Upstream source commit: `9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246`.
- Hugging Face model: `Layer6/TabDPT` at immutable revision `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`.
- Weight file: `tabdpt1_2.safetensors`, SHA-256 `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`.
- Both notebooks print effective Python/framework/model runtime identity. FlashAttention is explicitly disabled for portable Tesla T4 execution; CPU remains a valid but slower path.

## Preprocessing and capacity semantics

For TabDPT, `fit()` performs preprocessing fitting plus **in-context support conditioning**, not gradient fine-tuning. The repository feature encoder preserves numeric columns and deterministically maps categorical values, using separate fitted codes for missing and unseen categorical values. Upstream TabDPT uses a training/support-fitted mean imputer for numeric missing values and standard scaling. If encoded feature width exceeds the loaded model's `max_features`, the upstream configured feature-reduction path is surfaced by the tutorials; `context_size` separately limits support rows used during prediction.

The E2E notebook reports input shape/missingness, model feature ceiling, whether feature reduction activates, requested context size, and the effective upper bound on support rows. The artifact-inference notebook reports the same serving-relevant capacity information for an externally supplied artifact and validates new numeric data so non-numeric or infinite values do not silently become imputed values.

## Artifact contract and trust boundary

The reusable DIMER serving state includes the labelled support table, fitted repository encoder state, fitted upstream imputer/scaler state, any fitted PCA basis needed for wide tables, and the exact pinned base-model contract.

`export_artifact_bundle()` emits an explicit v3 release artifact. `validate_artifact_bundle()` checks artifact format/task, fitted preprocessing consistency, support-table path/size/SHA-256, and the declared base-model repository/revision/filename/digest/upstream commit before model loading. `load_verified_artifact()` reconstructs new release artifacts from saved fitted preprocessing state **without fitting those preprocessing statistics again**.

Established older `tabdpt-dimer-context-v3` manifests that predate the additive explicit version/semantics/size fields remain compatible. If such an artifact lacks the complete fitted upstream preprocessing state, `load_verified_artifact()` may use the legacy reconditioning compatibility path and sets `preprocessing_restored_ = False`. That fallback is retained for compatibility but **must not be used as release-grade `ARTIFACT-INFERENCE` evidence**; the companion notebook fails closed when it encounters that path.

The E2E notebook exports `artifact.json` and `training_context.parquet`, copies them across a fresh filesystem boundary, reconstructs the serving pipeline through the verified no-refit path, and checks prediction equivalence. The artifact-inference notebook does **not** manufacture its own artifact: the user must supply those files from a separate producing workflow, then provide separate unlabelled CSV/Parquet input for scoring.

Digest and manifest checks establish internal consistency with this repository's contract; they do not authenticate the sender. The release artifact format accepts Parquet context plus JSON metadata and does not accept ZIP, pickle, or arbitrary Python-object payloads. Because `training_context.parquet` contains labelled support data, treat an exported artifact with the same confidentiality, licensing, retention, and disclosure controls as its source dataset.

## Release verification

CI covers notebook JSON/source structure, Python-cell compilation, profile metadata, runtime-floor markers, preprocessing/capacity workflow assertions, artifact-contract/no-refit regression tests, and ordinary repository tests. These checks are not evidence that the current Colab model host/runtime/GPU path executes successfully.

Before marking either notebook `release-grade`, record a clean execution against the exact candidate commit. For the `E2E` notebook, verify the default sample path, pinned-weight digest, preprocessing/capacity report, metrics/baseline, machine-readable exports, artifact export, no-refit fresh reload, and prediction-equivalence assertion. For `ARTIFACT-INFERENCE`, use an artifact produced in a separate execution/session plus separately supplied new data, and verify pre-load validation, `preprocessing_restored_ = True`, reconstruction, capacity/context reporting, prediction, and exported results/provenance.

The durable execution-only gate is tracked in issue #15. Until that evidence exists, the correct status is **release candidate**, not release-grade.
