# Release verification

`tutorials/tabdpt_regressor_colab.ipynb` (`E2E`) and `tutorials/tabdpt_regressor_artifact_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed
top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER
Notebook Specification 1.1. This file is the durable release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the two notebooks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the
  notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, module commit, the three module paths, the package
  SHA-256, generator); `dimer_runtime.py`'s top-level `__main__` guard is rewritten to `if False:` by the
  generator so the worker entrypoint cannot execute inside the kernel;
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; three cells tagged `embedded_module`, in dependency order, equal to
  `src/tabdpt_regressor_pipeline/{pipeline,artifact,dimer_runtime}.py` after the generator's documented
  rewrite (the `DEFAULT_WEIGHTS_DIR` line) and the removal of package-relative imports; the inline `MANIFEST`
  equal to the committed `weights/tabdpt-1.2/dimer-base-manifest.json` and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py` output for its
  template; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and
  the same identity string appears in `README.md`, `MODEL_CARD.md`, and `weights/README.md` with no stray revisions
  (the upstream `TabDPT-inference` code commit is the one permitted extra 40-hex string);
- the profile-specific public-API calls — E2E: `stage_missing_files`, `verify_snapshot`,
  `TabDPTRegressionPipeline.from_pretrained(weights_dir=..., compile_model=False, use_flash=False, seed=42)`,
  `validate_inputs`, `training_mean_baseline`, `fit`, `evaluate`, `evaluation_report`, `predict`,
  `export_artifact_bundle`, `load_verified_artifact` with the no-refit and equivalence assertions; companion:
  `validate_artifact_bundle` before reconstruction, `load_verified_artifact(..., model_weight_path=pipe.model_weight_path, ...)`,
  the fail-closed legacy-path and target-column checks, `validate_inputs(..., target_column=None, feature_columns=...)`,
  `predict`, a `not-measurable` `evaluation_report` — the ceiling prints (`MIN_DISTINCT_TARGETS`,
  the model feature ceiling), the four exports per notebook, the learner-facing regression statements
  (continuous point estimates only, no uncertainty interval, MAE/RMSE in target units, no gradient training,
  no-refit reload, trust boundary, no artifact created in the companion) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, `worker.run(` / `worker_cli(` / `subprocess.run([` outside the generator-owned
  install cell, direct `from tabdpt import` / `TabDPTRegressor(` / `resolve_tabdpt_weights(` / `hf_hub_download(` /
  `sklearn.metrics` use **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`,
  `extractall(`; in the companion also `load_diabetes(`, `export_artifact_bundle(`, `.fit(` outside the modules);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check` for both templates, `scripts/validate_repo.py`,
`scripts/validate_colab_tutorial.py` (the repository's earlier source checks, kept and updated to spec 1.1) and the
offline unit suite (`tests/`, including `test_snapshot_helpers.py`, `test_role_helpers.py` and
`test_notebook_parity.py`; injected downloader, no weights, no model). These are source/provenance and unit checks.
They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime, Python 3.11+ | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle kernel, Python 3.11+ image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebooks are standalone). For the companion the shim also places the E2E run's `outputs/artifact/` and a separately generated unlabelled CSV, and sets `ARTIFACT_DIR` / `NEW_DATA_PATH` to them |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim (this repository ships none; the classifier repository's `scripts/execute_notebook_release.py` is the closest model, minus its `/content` rewriting — the standalone pair writes under `outputs/`) | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle executor above)
   with **no repository checkout** and a clean model cache;
3. run it top-to-bottom without editing implementation cells (form parameters at their defaults for the
   sample path: `USE_BYOD = False`, `CATEGORICAL_COLUMNS = []`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`; note `huggingface-hub==0.36.2` differs from the 0.33.2 of the previous `requirements-colab.txt` lock set);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (define `TabDPTRegressionPipeline`, the artifact helpers and the
     DIMER runtime, whose `__main__` guard is disabled) with no import of the repository package;
   - pinned `Layer6/TabDPT` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/tabdpt-1.2/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports the one manifest entry (`tabdpt1_2.safetensors`,
     254,098,072 bytes) on a clean runtime, `verify_snapshot` returns the manifest dict, and `from_pretrained(...)`
     reports `source == 'local-snapshot'` without loading a model;
   - the diabetes sample loaded from the installed scikit-learn with its CSV SHA-256 printed, the ceiling
     (`MIN_DISTINCT_TARGETS` 2) and request parameters surfaced;
   - `validate_inputs` writes `outputs/tabdpt_regressor_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the missing-target probe); the random 80/20 split is reported;
   - `training_mean_baseline`, `fit` (in-context conditioning, `use_flash=False`), the capacity report
     (`modelFeatureCeiling`, `featureReductionActive`, imputer), and `evaluate` with MAE / RMSE / R²;
   - `evaluation_report` writes `outputs/tabdpt_regressor_evaluation_report.json` with verdict `sample-sanity`,
     the three metric ids and the training-mean baseline;
   - `predict` on 8 held-out rows; `outputs/tabdpt_regressor_predictions.csv` (`row_id`, `prediction`) and `outputs/tabdpt_regressor_result.json` written with `NOTEBOOK_SOURCE`, model
     revision, model licence, runtime versions and device;
   - `export_artifact_bundle` writes `outputs/artifact/{artifact.json,training_context.parquet}`; the copy in
     `outputs/artifact-reload/` reloads through `load_verified_artifact` with `preprocessing_restored_ is True`
     and predictions equal within `rtol=1e-5, atol=1e-6`;
6. in a **second** clean runtime, run the exact companion notebook revision with `ARTIFACT_DIR` pointing at a copy of
   the E2E run's `outputs/artifact/` and `NEW_DATA_PATH` at a separately generated unlabelled CSV (or supply both
   through the upload dialog); verify `validate_artifact_bundle` passes before reconstruction, the reconstructed
   `preprocessing_restored_ is True` and the target column matches, `validate_inputs(..., target_column=None, ...)` writes
   `outputs/tabdpt_regressor_artifact_inference_input_manifest.json` with one recorded rejection finding, predictions
   and the `not-measurable` `outputs/tabdpt_regressor_artifact_inference_evaluation_report.json`,
   `..._predictions.csv` and `..._result.json` are written;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabdpt, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with
`git rev-parse <commit>:tutorials/<notebook>`). Wall times, when recorded, are the sum of per-cell
times reported by the executor and include installs and the model download; they are measurements
for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `46c5e17` / `0611e07348b4` | Kaggle T4 (`kurtvalcorza/dimer-nb2-tabdpt-regressor` v1) | Default sample path | 231.7 s | **PASSED** — 11/11 ok code cells executed cleanly, 4 files, 254 MB staged |
| | | | Standalone ARTIFACT-INFERENCE with an external artifact | | pending — queued to the GPU lane |

## Current status

No clean-runtime execution of the standalone notebooks has been recorded yet; both runs are **pending**
and queued to the GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blobs under review and
an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real
`hf_hub_download` fetch of `tabdpt1_2.safetensors` into a fresh `weights/tabdpt-1.2/` has not been executed);
`from_pretrained` loads no model, so the first real load of the checkpoint through the snapshot path happens inside
`fit`; no execution of the previous notebook pair was ever recorded either (issue #15); and the standalone
carrier itself — executing the three carried module cells in a runtime that has no repository checkout — has been
validated statically only (parity PASS, carrier probe with the package import blocked), never run. The clean runs
will be the first execution of the standalone path, of the staging path, and of the helper stages
(`validate_inputs`, `training_mean_baseline`, `evaluation_report`) against the real weights.