# TabDPT Regressor Pipeline

DIMER-ready inference pipeline for **TabDPT v1.2 (TabDPT-Turbo)**, an open-weight tabular foundation model from Layer 6 AI for in-context supervised learning.

## What this repository provides

- supervised tabular **regression** with `TabDPTRegressor`;
- train-fitted preprocessing for mixed numeric/categorical CSV tables;
- immutable upstream weight provenance and SHA-256 verification;
- a DIMER manifest plus executable local/on-prem runtime adapter;
- deterministic validation splitting and support-row capping from DIMER runtime controls;
- a lightweight Colab tutorial and CI tests that do not download the 254 MB model in routine CI.

## Pinned model identity

| Field | Value |
|---|---|
| Upstream package | `tabdpt==1.2.0` |
| Upstream inference source | `layer6ai-labs/TabDPT-inference` commit `9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246` (`v1.2.0`) |
| Hugging Face model | `Layer6/TabDPT` |
| Weight file | `tabdpt1_2.safetensors` |
| HF revision | `4462ffbd1d8dea25d4862d30beed4b70cd596ae5` |
| SHA-256 | `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd` |
| Weight size | 254,098,072 bytes |
| License | Apache-2.0 |

The weight bytes are not committed to this repository. Runtime acquisition is pinned to the immutable Hugging Face revision and verified before model construction.

## DIMER runtime contract

DIMER's current tabular transport is reused rather than inventing a new channel:

- `datasetPreprocessing` → `DIMER_PREPROCESSING_ARGS_JSON`
- `modelFinetuning` → `DIMER_HYPERPARAMETERS_JSON`

For TabDPT, `modelFinetuning` is a compatibility transport namespace only. `fine_tune` must remain `false`; the other fields control ICL context/inference (`n_ensembles`, `context_size`, `batch_size`, `seed`). Every manifest key is consumed by `tabdpt_regressor_pipeline.dimer_runtime`, and CI checks the manifest/runtime key sets for exact equality.

`dimer_entrypoint.py` reads `DIMER_DATASET_DIR`, accepts `train.csv` plus optional `val.csv` directly or inside one ZIP, applies the requested validation split/support cap, executes TabDPT, and writes `result.json`, `training_context.parquet`, and `artifact.json` under the configured output paths.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[model]'
```

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tabdpt_regressor_pipeline import TabDPTRegressionPipeline

frame = load_diabetes(as_frame=True).frame
train, test = train_test_split(frame, test_size=0.2, random_state=42)
pipe = TabDPTRegressionPipeline()
pipe.fit(train, target_column='target')
print(pipe.evaluate(test))
```

For DIMER, mount/cache the verified base weight through `DIMER_BASE_MODEL_PATH` so execution does not depend on live internet access.

## Evaluation caveat

TabDPT was pretrained on real-world tabular datasets. Public tutorial/benchmark datasets may overlap directly or indirectly with upstream pretraining. Tutorial metrics are plumbing/sanity checks, not clean evidence of out-of-distribution quality.

## Upstream references

- Hosseinzadeh et al., *TabDPT-Turbo: Efficient In-Context Learning for Tabular Prediction*, arXiv:2608.01400 (2026). This is the paper for the v1.2 / TabDPT-Turbo release packaged here.
- Ma et al., *TabDPT: Scaling Tabular Foundation Models*, arXiv:2410.18164 / NeurIPS 2025. This is the base TabDPT work.

This repository is an integration project and is not affiliated with or endorsed by Layer 6 AI or The Toronto-Dominion Bank.
