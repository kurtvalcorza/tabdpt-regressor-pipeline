# Deployment and Operations — TabDPT Regressor

## DIMER components

- **Validator**: `tabdpt-regressor-dataset-validator`, CPU image; DIMER entrypoint `validate.py`.
- **Fine-tuner / Worker**: `tabdpt-regressor-finetuner`, CUDA image; DIMER entrypoint `train.py`.
- **Base checkpoint**: `tabdpt1_2.safetensors`.
- **Python package**: `tabdpt==1.2.0`, `tabdpt_regressor_pipeline`.

Both deployable container repositories build with the repository root as the Docker build context and expose their DIMER entrypoint at the repository root (`validate.py` for the validator, `train.py` for the fine-tuner). `model_id` is intentionally not a manifest parameter — the DIMER Base Model selection is authoritative (see [`DIMER_CONTRACT.md#dimer-field-to-runtime-mapping`](DIMER_CONTRACT.md#dimer-field-to-runtime-mapping)).

## Network

Distinguish build-time egress from runtime egress:

- **Build-time egress:** Building the worker image bakes the pinned TabDPT checkpoint by fetching `tabdpt1_2.safetensors` from Hugging Face (see *Base model handoff* below). Building the default image requires Hugging Face egress at build time. Clean or network-isolated DIMER build agents must allow build-time access, or the image build fails.
- **Runtime egress:** Once baked, the default `pinned-baked` path performs **no runtime download**. Runtime model-download egress is only needed if operating under the `pinned-download` fallback (no baked copy present). When an operator mounts a model directory via `DIMER_BASE_MODEL_PATH`, runtime execution is fully air-gapped and requires no external network calls.

## Hardware and GPU resources

In-context inference requires sufficient host memory to hold the tabular dataset and transformer activations. For production workloads:
- **CPU / Memory:** Minimum 4 CPU cores, 16 GB RAM recommended.
- **GPU / Accelerator:** Recommended 1 CUDA-capable GPU with at least 8 GB VRAM (e.g., NVIDIA T4, A10G, L4, or A100).
- **GPU Attention Compatibility:** For automatic device capability detection, FlashAttention support, and fallback behavior, see [`DIMER_CONTRACT.md#gpu-attention-compatibility`](DIMER_CONTRACT.md#gpu-attention-compatibility).

## Artifact serving contract

Because TabDPT is an in-context learner, prediction conditions on the saved support table. A completed run writes:

- `artifacts/training_context.csv`: the capped in-context support table;
- `artifacts/artifact.json`: task metadata, versioned fitted preprocessing state, and context digest;
- `result.json`: run execution status, validation metrics (MAE, RMSE, R²), and component hashes.

See [`DIMER_CONTRACT.md#outputs`](DIMER_CONTRACT.md#outputs) and [`DIMER_CONTRACT.md#feature-schema-and-persisted-preprocessing`](DIMER_CONTRACT.md#feature-schema-and-persisted-preprocessing) for schema definitions. The inference service reconstructs preprocessing state via `TabularFeatureEncoder.from_state()` and calls `predict`.

Because the artifact bundle embeds the training context rows, treat the exported artifact with the same data-governance controls as the source training dataset.

## Base model handoff

This pipeline is fixed to the pinned TabDPT v1.2 foundation model:
- Hugging Face repository: `Layer6/TabDPT`
- Hugging Face revision: `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- File: `tabdpt1_2.safetensors`
- SHA-256: `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`

See [`weights/README.md`](weights/README.md) for offline staging instructions.

Resolution precedence:
1. `dimer-provided`: `DIMER_BASE_MODEL_PATH` is set by the DIMER operator. The file must exist and match the pinned SHA-256 (or fail closed).
2. `pinned-baked`: Baked checkpoint inside the container image. SHA-256 verified at startup.
3. `pinned-download`: Automatic download fallback from Hugging Face when no local copy is found, hard-verified by SHA-256.

## Pre-enable checklist (release gate)

Production enablement requires completing this 9-step gate. Steps 1–6 are repository-verifiable via CI and local smoke tests; steps 7–9 are platform-owned and must be verified on the target DIMER deployment. Production enablement is strictly withheld until step 7 passes on-platform.

### Repository-verifiable steps
1. **Repository contracts:** CI passes across Python 3.10–3.14, verifying manifest key mapping, dataset limits, and non-constant continuous target handling.
2. **Adversarial dataset rejection:** Confirm validator rejects duplicate splits, nested/zip-bomb archives, path traversal, symlinked roots, and malformed configurations with explicit errors.
3. **Reproducible sample verification:** Confirm `examples/build_sample_datasets.py` deterministically regenerates `examples/sample-data/diabetes.zip` matching documented checksums.
4. **Tutorial AST & path validation:** Confirm `scripts/validate_colab_tutorial.py` passes on all notebooks under `tutorials/`, verifying AST syntax, path safety, and explicit `use_flash=False`.
5. **Fresh-process reload test:** Confirm `tutorials/tabdpt_regressor_artifact_inference_colab.ipynb` successfully loads `artifact.json` + `training_context.csv` and reproduces expected outputs without refitting.
6. **Provenance integrity:** Confirm `result.json` records validation metrics, base model revision, context SHA-256, and runtime configurations.

### Platform-owned steps
7. **On-platform serving acceptance (release blocker):** Deploy the image to DIMER Workbench. Verify the full pipeline: BYOD upload → validator → worker run → artifact export → fresh reload → DIMER deployment → API `predict` parity. Production enablement is withheld until this step passes.
8. **Manifest control verification:** Confirm each `dimer-pipeline.json` control modifies runtime behavior as expected (e.g. `n_ensembles`, `context_size`, `seed`).
9. **Resource profiling:** Measure peak memory and GPU usage under representative workload bounds to establish production instance sizing.
