# DIMER integration contract — TabDPT regressor

## Repository-owned contract

### Task identity

`dimer-pipeline.json` declares `taskType: tabular_regression`.

### Base model

The default base model is TabDPT v1.2:

- `Layer6/TabDPT`
- immutable revision `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- `tabdpt1_2.safetensors`
- SHA-256 `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`

The runtime may supply `model_weight_path`; the wrapper rejects any file whose digest differs from the repository-pinned identity.

### Dataset and preprocessing

The wrapper consumes a labelled pandas table. Categorical encoders are fitted only on training data. Numeric missing-value imputation remains inside TabDPT and is fitted by the upstream estimator on the training context. Feature schema is locked after fit. The regression target must be numeric, finite, and non-constant.

For a single uploaded labelled CSV, DIMER may create a random validation split only when IID rows are a defensible assumption. Temporal/grouped/entity data must be split upstream and supplied with preserved partitions.

### Inference controls

The versioned manifest exposes `n_ensembles`, `context_size`, `batch_size`, and `seed`. These control throughput/memory and prediction ensembling; they do not train model weights.

### Fine-tuning

This repository intentionally has **no DIMER fine-tuning contract in v1**. Upstream's supported inference API is in-context. The separate TabDPT training repository is not silently treated as a production fine-tuner.

## DIMER-side requirements

1. Mount or cache the pinned model artifact so production inference does not depend on live internet access.
2. Preserve the base-model digest in model registry metadata.
3. Persist the fitted support/context table and preprocessing metadata whenever a fitted ICL predictor is promoted as a reusable deployment artifact.
4. Treat `context_size` and `n_ensembles` as resource controls and enforce deployment-specific ceilings.
5. Keep validation splitting semantics aligned with the dataset's temporal/group/entity structure.
6. Run a production acceptance test: upload/preserved split → fit context → predict → persist state → fresh reload → prediction parity.

## Acceptance boundary

Repository CI does not download the 254 MB checkpoint. CI proves deterministic preprocessing and manifest/provenance invariants with lightweight tests. Final model execution must be tested in a GPU-capable integration environment before DIMER marks the model production-ready.