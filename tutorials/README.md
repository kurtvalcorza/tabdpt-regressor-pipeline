# TabDPT Regressor Colab Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabdpt-regressor-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat)](https://huggingface.co/Layer6/TabDPT)
[![Upstream](https://img.shields.io/badge/Upstream-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white)](https://github.com/layer6ai-labs/TabDPT-inference)
[![arXiv](https://img.shields.io/badge/arXiv-2608.01400-b31b1b.svg)](https://arxiv.org/abs/2608.01400)

These interactive tutorials demonstrate how to run TabDPT Regressor in standalone environments like Google Colab or Kaggle while respecting repository contracts, GPU hardware portability, and serving artifact integrity.

## Available Tutorials

| Notebook | Accelerator | Expected Runtime | Focus |
|---|---|---|---|
| [`tabdpt_regressor_colab.ipynb`](tabdpt_regressor_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-regressor-pipeline/blob/main/tutorials/tabdpt_regressor_colab.ipynb) | GPU (Tesla T4 or newer; CPU supported for smoke test) | ~1–2 minutes | End-to-end tutorial: install package, fit support table in-context, predict continuous values, evaluate test holdout (MAE, RMSE, R²). |
| [`tabdpt_regressor_artifact_inference_colab.ipynb`](tabdpt_regressor_artifact_inference_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabdpt-regressor-pipeline/blob/main/tutorials/tabdpt_regressor_artifact_inference_colab.ipynb) | GPU (Tesla T4 or newer; CPU supported) | ~1–2 minutes | Artifact inference tutorial: load exported DIMER bundle (`artifact.json` + `training_context.csv`), verify context digest, score unlabelled batch data without refitting. |

---

## Accelerator & Portability Notes

- **Tesla T4 Portability:** On common Google Colab Tesla T4 instances (compute capability 7.5), FlashAttention is unsupported. Both tutorial notebooks explicitly set `use_flash=False` in `TabDPTRegressionPipeline(compile_model=False, use_flash=False)` to guarantee error-free portable execution across both T4 and modern Ampere/Hopper (`sm_80+`) devices.
- **CPU Fallback:** Pretrained in-context inference executes on CPU if no GPU is visible, though latency is higher on larger tables.

---

## Artifact Trust and Data Governance

1. **Digest Verification:** Always verify `training_context.csv` SHA-256 against `artifact.json` before loading.
2. **Data Governance:** In-context foundation models preserve the training context table as part of the model state. Treat exported artifact bundles with the same confidentiality, retention, and access policies as the underlying training dataset.
