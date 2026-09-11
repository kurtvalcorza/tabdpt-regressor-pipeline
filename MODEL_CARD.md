---
license: apache-2.0
model_card_spec: "1.0"
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
  - tabdpt
base_model: Layer6/TabDPT
---

# TabDPT Regressor v1.2

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat)](https://huggingface.co/Layer6/TabDPT)
[![GitHub](https://img.shields.io/badge/GitHub-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white)](https://github.com/layer6ai-labs/TabDPT-inference)
[![arXiv](https://img.shields.io/badge/arXiv-2608.01400-b31b1b.svg)](https://arxiv.org/abs/2608.01400)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

###### Description

TabDPT v1.2, released as **TabDPT-Turbo**, is an open-weight tabular foundation model designed for in-context supervised regression on structured datasets. Rather than iteratively training neural network weights or tree ensembles on each new dataset via gradient descent or heuristic splits, TabDPT processes a labelled support table (the in-context prompt) containing continuous targets alongside unlabelled test observations through a specialized tabular Transformer architecture. Task adaptation occurs entirely at inference time through in-context forward evaluation without gradient updates or per-dataset training loops. For single-context queries without ensembling, inference requires only a forward evaluation; when ensembling over multiple support subsets (`n_ensembles > 1`) or batching query chunks, predictions are aggregated across multiple forward passes. Pretrained on a diverse corpus of real-world tabular datasets and optimized with FlashAttention and key-value caching in v1.2 (Turbo), it delivers rapid, zero-shot tabular regression without per-dataset hyperparameter tuning. This repository packages the upstream regression estimator for reproducible, DIMER-ready deployment.

#### Intended Use and Limitations

###### Primary Intended Uses

Supervised tabular regression tasks, predicting continuous numerical values from structured tables where features consist of numerical and categorical columns. Suitable target applications include asset pricing and valuation, duration and latency estimation, physical property estimation in scientific and engineering research, resource and demand estimation with static feature tables, and general row-per-observation regression problems. It is designed to act as an out-of-the-box strong baseline and inference engine requiring zero hyperparameter optimization.

###### Primary Intended Users

Machine learning engineers, data scientists, quantitative researchers, and software engineers developing predictive regression pipelines for structured datasets in enterprise, scientific, or academic environments. The envisioned deployment setting is internal enterprise or research use through the DIMER platform, not a public-facing service. Users are expected to understand data validation, leakage prevention, distribution shift, and standard regression evaluation methodology, and to recognise that the point prediction is a mean-style estimate with no attached interval; a user who cannot tell a held-out evaluation from an in-context evaluation should not be setting operational cutoffs on this pipeline's output.

###### Out-of-scope use cases

- Categorical target classification (use TabDPT Classifier v1.2 instead).
- Unsupervised clustering, dimensionality reduction, density estimation, or tabular data synthesis.
- Raw unstructured modalities (unprocessed images, audio, video, free-form long text) without prior tabular feature extraction.
- Non-stationary time-series forecasting requiring autoregressive temporal sequence modeling without supervised lag feature engineering.
- Autonomous high-impact or safety-critical decisions without human oversight (e.g., automated medical drug dosing, high-frequency autonomous trading without risk collars).
- Datasets with non-numeric, infinite, missing, or constant targets during fit.
- Tables where linear feature compression is unsuitable: TabDPT-Turbo's native row encoder accommodates up to 128 features; for tables exceeding 128 features, the upstream estimator automatically applies PCA feature reduction down to its native 128-dimensional width. On very wide datasets where linear PCA discards critical non-linear signals, prior domain-specific feature selection is recommended. (Note that feature column width is distinct from transformer row-context capacity, which governs support row count `context_size`).

---

#### Factors

###### Groups

TabDPT v1.2 was pretrained on a broad corpus of public real-world tabular datasets and is not inherently tailored to, or debiased for, any specific demographic, phenotypic, or protected group (such as age, gender, race, ethnicity, or socioeconomic status). In human-centric applications, downstream users must rigorously audit group-level fairness and parity of residual errors across sensitive subpopulations.

###### Instrumentation

The model operates on normalized tabular data matrices (floating-point numbers and encoded categorical integers) rather than direct physical sensor streams. The "instruments" are upstream data collection systems, SQL databases, survey instruments, laboratory diagnostic assays, and ETL pipelines. Inaccuracies, sensor drifts, or calibration discrepancies in the underlying instruments directly propagate into model features.

###### Environment

TabDPT operates across generic computational environments (CPU, CUDA GPUs with `sm_80+` for FlashAttention, or `sm_75` like Tesla T4 with FlashAttention disabled). In terms of application environments, the model assumes that feature and target distributions between the in-context support set and the query test set are drawn from the same data-generating distribution; severe covariate shifts, target drifts, or institutional data discrepancies will degrade predictive fidelity.

---

#### Metrics

###### Performance Measures

Model evaluation in the pipeline and upstream benchmarks reports:
- **MAE (Mean Absolute Error)**: Average absolute magnitude of prediction errors in native target units.
- **RMSE (Root Mean Squared Error)**: Square root of mean squared residuals, heavily penalizing large estimation outliers.
- **R² (Coefficient of Determination)**: Proportion of target variance explained by the model relative to the baseline mean target predictor.

These measures assess both typical error magnitude, outlier sensitivity, and explanatory power across regression benchmarks.

###### Decision thresholds

The pipeline applies no decision threshold. `predict()` returns the raw continuous estimate as a `pd.Series` named `prediction` (`src/tabdpt_regressor_pipeline/pipeline.py`), and no acceptance threshold on MAE, RMSE, or R² was set during development because the pipeline is domain-agnostic and the tolerable error is a property of the deployment, not of the model. Any cutoff that turns a prediction into an action — a resource alert, a cost-overrun warning, a ±5 % tolerance band — is the deployment owner's to define and to calibrate on their own held-out data. Set it from the asymmetric cost of over- versus under-prediction in the target domain: where an under-estimate is the expensive error, place the cutoff below the point prediction by a margin derived from the held-out residual distribution, and revisit it whenever the input distribution shifts.

###### Approaches to uncertainty and variability

The shipped `TabDPTRegressionPipeline` wrapper outputs point predictions only (returning the ensemble mean across `n_ensembles` support subsets). It does **not** expose calibrated predictive variance, prediction intervals, or posterior quantiles. Users requiring uncertainty quantification must compute empirical residual distributions or conduct cross-validation across support partitions outside the core inference wrapper.

---

#### Ethical considerations and biases

###### Data

Pretrained by upstream authors on a broad collection of public tabular datasets sourced from open repositories. The pipeline distribution provides only model weights (`tabdpt1_2.safetensors`) and wrapper code, and does not distribute pretraining datasets. Downstream users should note that public pretraining tables may still reflect historical demographic skews, societal biases, or sensitive domain attributes present in their original sources. Operators deploying the model are responsible for auditing their own in-context support data for sensitive attributes, proprietary information, or PII before conditioning the model.

###### Human Life

The model is **not** certified, validated, or intended for autonomous decision-making in situations central to human life, health, or flourishing (such as clinical drug dosing, intensive care life support, structural failure risk automation, or autonomous vehicle control). Any application in sensitive domains requires human oversight and rigorous independent validation.

###### Mitigations

- Self-contained open weights (`tabdpt1_2.safetensors`) with cryptographic SHA-256 verification.
- Pinned upstream Hugging Face revision (`4462ffbd1d8dea25d4862d30beed4b70cd596ae5`).
- Context subsampling during support set construction to maintain stable computational bounds.
- Strict input schema validation preventing silent column misalignment.
- Deterministic random seed controls for reproducible sampling and ensembling.

###### Risks and harms

- **Overconfidence on out-of-distribution tables**: The model may produce substantial prediction errors when evaluating inputs outside the support distribution.
- **Amplification of dataset bias**: Conditioning on a biased support table will reproduce or amplify those biases in continuous predictions.
- **Automation bias**: Users uncritically accepting model predictions without domain review.
- **Leakage**: Accidental inclusion of target-correlated artifacts in the feature set leading to spurious high performance.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible output:
- Predictive algorithms designed for exploitative price discrimination or predatory lending.
- Automated resource-allocation systems that penalise protected or vulnerable groups, including any regression on a target that proxies a protected attribute.
- Generating deceptive analytical forecasts for fraudulent or manipulative purposes, or presenting the point prediction as a certified measurement.
- Any use that violates the Apache-2.0 terms of the upstream Layer6/TabDPT weights or the terms of the DIMER deployment.

---

## Model Details

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

## Primary references

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

## License

This repository, the upstream TabDPT inference package, and the referenced public model weights are Apache-2.0 licensed. See `LICENSE` and preserve applicable upstream notices when redistributing upstream materials.
