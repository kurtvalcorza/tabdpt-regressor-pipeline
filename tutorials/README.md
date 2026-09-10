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
| [`tabdpt_regressor_colab.ipynb`](tabdpt_regressor_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-regressor-pipeline/blob/main/tutorials/tabdpt_regressor_colab.ipynb) | `E2E` | Support-context conditioning, regression evaluation/baseline, new-data inference, DIMER artifact export, fresh reload/equivalence | GPU recommended; CPU supported; `use_flash=False` | **release candidate** — clean-runtime execution evidence required |
| [`tabdpt_regressor_artifact_inference_colab.ipynb`](tabdpt_regressor_artifact_inference_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-regressor-pipeline/blob/main/tutorials/tabdpt_regressor_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | Validate and reconstruct an externally supplied DIMER artifact; validate/score new data; export predictions/provenance | GPU recommended; CPU supported; `use_flash=False` | **release candidate** — external-artifact clean-runtime evidence required |

## Runtime and model contract

- Install the exact tutorial dependency set from [`requirements-colab.txt`](requirements-colab.txt) before importing model libraries.
- TabDPT package: `tabdpt==1.2.0`.
- Upstream source commit: `9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246`.
- Hugging Face model: `Layer6/TabDPT` at immutable revision `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`.
- Weight file: `tabdpt1_2.safetensors`, SHA-256 `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`.
- Both notebooks print the effective Python/framework/model runtime identity. FlashAttention is explicitly disabled for portable Tesla T4 execution; CPU remains a valid but slower path.

## Artifact contract and trust boundary

For TabDPT, `fit()` performs preprocessing fitting plus **in-context support conditioning**, not gradient fine-tuning. The reusable DIMER serving state therefore includes the labelled support table and fitted preprocessing state in addition to the exact pinned base-model contract.

The E2E notebook exports `artifact.json` and `training_context.parquet`, then reconstructs the serving pipeline from a fresh copied directory and checks prediction equivalence. The artifact-inference notebook does **not** manufacture its own artifact: the user must supply those files from a separate producing workflow, then provide separate unlabelled CSV/Parquet input for scoring.

`validate_artifact_bundle()` checks the artifact format/task, fitted preprocessing consistency, support-table path/size/SHA-256, and the declared base-model repository/revision/filename/digest/upstream commit before model loading. These checks establish internal consistency with this repository's contract; they do not authenticate the sender. The artifact format does not accept ZIP, pickle, or arbitrary Python-object payloads.

Because `training_context.parquet` contains labelled support data, treat an exported artifact with the same confidentiality, licensing, retention, and disclosure controls as its source dataset.

## Release verification

CI covers notebook JSON/source structure, Python-cell compilation, profile metadata, static workflow assertions, artifact-contract unit tests, and ordinary repository tests. These checks are not evidence that the current Colab model host/runtime/GPU path executes successfully.

Before marking either notebook `release-grade`, record a clean execution against the exact candidate commit. For the `E2E` notebook, verify the default sample path, pinned-weight digest, metrics/baseline, machine-readable exports, artifact export, fresh reload, and prediction-equivalence assertion. For `ARTIFACT-INFERENCE`, use an artifact produced in a separate execution/session plus separately supplied new data, and verify validation, reconstruction, prediction, and exported results/provenance.
