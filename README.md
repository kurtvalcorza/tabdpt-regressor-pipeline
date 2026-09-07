# TabDPT Regressor Pipeline

DIMER-ready inference pipeline for **TabDPT v1.2 (TabDPT-Turbo)**, an open-weight tabular foundation model from Layer 6 AI for in-context supervised learning.

## What this repository provides

- supervised tabular **regression** with `TabDPTRegressor`;
- train-fitted preprocessing for mixed numeric/categorical CSV tables;
- immutable upstream weight provenance and SHA-256 verification;
- a DIMER pipeline manifest and integration contract;
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

The weight bytes are **not committed to this repository**. Runtime acquisition is pinned to the immutable Hugging Face revision and verified before model construction.

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

For production or DIMER, pass a locally mounted verified weight file with `model_weight_path=` to avoid runtime network dependence.

## Operational defaults

TabDPT v1.2 can use full context when memory permits. DIMER exposes context and ensemble count because both directly affect memory and latency. The initial manifest uses a conservative 2,048-row context default and a 10,000-row support-table cap.

## Important evaluation caveat

TabDPT was pretrained on **real-world tabular datasets**. Public benchmark or tutorial datasets may overlap directly or indirectly with its pretraining corpus. Built-in datasets in the tutorial are therefore plumbing/sanity checks, **not clean evidence of out-of-distribution model quality**. Use application-specific held-out data for deployment decisions.

## Repository map

- `MODEL_CARD.md` — model provenance, intended use, limitations, and DIMER status.
- `DIMER_CONTRACT.md` — repository/platform integration boundary.
- `TABULAR_REGRESSION_DATASET_SPEC.md` — accepted dataset shape and split rules.
- `dimer-pipeline.json` — versioned DIMER controls.
- `src/tabdpt_regressor_pipeline/` — reusable inference wrapper.
- `tutorials/tabdpt_regressor_colab.ipynb` — end-to-end smoke tutorial.
- `scripts/validate_repo.py` — static contract validation.

## Upstream

TabDPT: *Scaling Tabular Foundation Models on Real Data*, NeurIPS 2025, arXiv:2410.18164.

This repository is an integration project and is not affiliated with or endorsed by Layer 6 AI or The Toronto-Dominion Bank.