# Model Card — TabDPT Regressor v1.2

## Model overview

TabDPT v1.2, released as **TabDPT-Turbo**, is an open-weight tabular foundation model for in-context supervised prediction. This repository packages the upstream regression estimator for reproducible DIMER-oriented inference.

- **Task:** supervised tabular regression
- **Model family:** tabular transformer / in-context learner
- **Upstream release:** TabDPT v1.2 / TabDPT-Turbo
- **Upstream package:** `tabdpt==1.2.0`
- **Weight artifact:** `tabdpt1_2.safetensors`
- **Artifact size:** 254,098,072 bytes
- **Weight license:** Apache-2.0
- **Code license:** Apache-2.0

## Immutable provenance

- Upstream inference source commit: `9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246` (`v1.2.0`)
- Hugging Face repository: `Layer6/TabDPT`
- Hugging Face revision: `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- Filename: `tabdpt1_2.safetensors`
- SHA-256: `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`

The wrapper verifies SHA-256 before constructing the upstream estimator.

## Primary v1.2 reference

**Hosseinzadeh, R., Labach, A., Xue, Z., Han, S., Thomas, V., & Caterini, A. L. (2026). _TabDPT-Turbo: Efficient In-Context Learning for Tabular Prediction_. arXiv:2608.01400.**

The Turbo paper explicitly describes the released model as TabDPT v1.2. The earlier base work remains relevant background:

**Ma, J., et al. (2025). _TabDPT: Scaling Tabular Foundation Models_. NeurIPS 2025; arXiv:2410.18164.**

## How inference works

`fit(X_train, y_train)` prepares the labelled table as in-context support and fits preprocessing statistics; it does not update the pretrained TabDPT weights. Prediction conditions the frozen model on this support table. Regression uses ordinary context subsampling when a requested context cap is smaller than the available support set.

## Input contract

- numeric and categorical/string columns are supported;
- learned categorical mappings are fitted only on training data;
- unseen and missing categorical values receive dedicated codes;
- `drop_columns` are excluded before the fitted schema is established and may be present in later raw tables;
- after configured drop columns are removed, inference must contain exactly the fitted feature names—missing or unconfigured extra columns are rejected;
- regression targets must be numeric, finite, non-missing, and non-constant during fit.

## Output contract

- `predict(...)`: one floating-point prediction per row;
- `evaluate(...)`: MAE, RMSE, and R².

## DIMER runtime status

The repository includes an executable local/on-prem DIMER adapter. `datasetPreprocessing` is consumed from `DIMER_PREPROCESSING_ARGS_JSON`; the platform's existing `modelFinetuning` transport is consumed from `DIMER_HYPERPARAMETERS_JSON`, but `fine_tune=true` is explicitly rejected because v1.2 uses ICL rather than gradient fine-tuning.

The adapter supports `train.csv` and optional `val.csv`, applies deterministic splitting/support capping, executes the model, and writes result/provenance/context artifacts. Routine CI tests this contract without downloading the model weight. A real checkpoint/GPU smoke test and on-platform deployment/reload test remain production-acceptance work.

## Training-data / benchmark caveat

TabDPT was pretrained on real-world tables. Common public benchmarks or tutorial datasets may overlap with upstream pretraining, so their metrics should be treated as smoke-test evidence rather than independent benchmark claims.

## Intended use and limitations

Suitable for conventional supervised tabular regression after task-specific validation. It is not a forecasting-specific or causal model and is not independently validated for high-stakes medical, financial, legal, policy, or safety-critical automation.

## License

This repository, the upstream TabDPT inference package, and the referenced public model weights are Apache-2.0 licensed. See `LICENSE` and preserve applicable upstream notices when redistributing upstream materials.
