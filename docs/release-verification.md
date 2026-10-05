# Release verification

`tutorials/tabdpt_regressor_colab.ipynb` (`E2E`) and `tutorials/tabdpt_regressor_artifact_inference_colab.ipynb`
(`ARTIFACT-INFERENCE`) are **release candidates** until the exact notebook revisions have executed
top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER
Notebook Specification 1.1. This file is the durable release-gate record for both notebooks.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for each of the two notebooks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating commit, package module paths and SHA-256, carried-file digests,
  generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and the isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose
  `CARRIED_FILES` / `CARRIED_BINARY` equal the repository's package (`src/tabdpt_regressor_pipeline/`), the notebook's
  stage runner (`tools/tutorial_stages.py` or `tools/tutorial_stages_artifact_inference.py`), the hash lock
  `tutorials/requirements-colab.lock.txt` (which must pin every `pyproject.toml` runtime pin, every entry hashed), the
  committed snapshot manifest, the licence and, for the companion, `examples/sample-artifact/`; `CARRIED_HASHES` match;
  the notebook is byte-identical to `tools/build_notebook.py` output for its template; no cell pip-installs into the
  notebook kernel and no text asks for a runtime restart; the install cell uses a pinned `uv` wheel, a managed CPython,
  `--require-hashes`, a lock-digest-keyed environment that is reused, `MPLBACKEND=Agg`, and drops
  `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`; the four Infrastructure cells are collapsed (`cellView: form`); every
  learner cell runs a stage;
- the profile-specific public-API calls in the carried stage runner — E2E: `from_pretrained(weights_dir=..., compile_model=False,
  use_flash=False, seed=SEED)`, `validate_inputs`, `training_mean_baseline`, the linear-regression reference with its
  split-to-split R² range, `fit`, `evaluate`, `evaluation_report`, `predict`,
  `export_artifact_bundle`, `load_verified_artifact` in a fresh process with the no-refit and equivalence checks, the
  BYOD refusals for an absent, blank or non-numeric target and for a table below 10 rows; companion: the trusted-digest check before `validate_artifact_bundle`,
  `load_verified_artifact(..., model_weight_path=<verified checkpoint>, compile_model=False, use_flash=False)`, the
  fail-closed legacy-path and target-column checks, `validate_inputs(..., target_column=None, feature_columns=...)`,
  `predict`, a `not-measurable` `evaluation_report` — the exports per notebook, the learner-facing
  regression statements and the guided layer (how-to-use, task contract, roadmap, glossary, predictions,
  checkpoints, troubleshooting, conclusion), and the optional-input gates at their non-interactive defaults; no
  `assert` in the stage runners (verdicts are reported, contract checks raise with a message); forbidden patterns
  (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` on the primary path, a mutable `revision='main'`,
  direct `tabdpt` / `huggingface_hub` / `sklearn` / `torch` use in the notebook's own cells, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`; in the companion also `load_diabetes(`, `export_artifact_bundle(`,
  `.fit(` in its stage runner);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `tools/build_notebook.py --check` for both templates, `scripts/validate_repo.py`,
`scripts/validate_colab_tutorial.py` and the offline unit suite (`tests/`, including `test_notebook_parity.py`,
`test_companion_parity.py` and `test_notebook_review_fixes.py`, which execs the notebooks' own kernel cells with
stand-ins and runs the model-free stages; injected downloader, no weights, no model). `tools/build_sample_artifact.py
--check` reproduces the pinned sample artifact byte for byte. These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 GPU runtime (CPU also works); the kernel's Python does not matter — the stages run in an isolated CPython 3.12.12 environment | The runtime the tutorials are written for; a clean one-pass top-to-bottom run here is promotion evidence |
| Kaggle kernel | Kaggle T4 kernel | Reproducible clean-room executor of the same class; the notebook is pushed verbatim (no repository checkout is needed — the notebooks are standalone, and the companion's default path needs no supplied files) |
| Local harness (pre-flight only) | Workstation | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence. `scripts/execute_notebook_release.py` was written for the previous repository-installing pair and does not apply to the /3 notebooks |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a **fresh** Colab T4 runtime (or the Kaggle executor) with no repository
   checkout and a clean model cache;
3. choose **Run all** once, with every form field at its default (`USE_BYOD = False`, `RUN_ACTIVITY = False`); no
   restart may be needed — record `restarted: false`;
4. verify that Section 2 reports the generating revision recorded in `metadata.dimer.generated_from`, and that the
   isolated environment reports Python 3.12.12, `torch` 2.7.1, `tabdpt` 1.2.0, `numpy` 2.3.0, `pandas` 2.3.2,
   `scikit-learn` 1.7.0 and `cuda: True` on a T4; then re-run the Section 2 install cell once and check
   `environment_reused: True`;
5. verify every default-path stage completes:
   - `weights`: `fetched` names `tabdpt1_2.safetensors` (254,098,072 bytes) on a clean runtime and `verify_snapshot`
     passes;
   - `data` / `validate`: the diabetes sample (442 rows, CSV SHA-256 printed), the input manifest with one recorded
     rejection finding, the 353 / 89 random split;
   - `condition`: capacity (`modelFeatureCeiling`, `featureReductionActive`, imputer) and TabDPT's MAE, RMSE and R²
     beside the training-mean baseline (MAE 64.01, RMSE 73.22, R² −0.012) and the linear-regression reference (MAE 42.79,
     RMSE 53.85, R² 0.453; R² 0.332–0.585 over 20 seeded splits) — record TabDPT's printed numbers;
   - `report`: `outputs/tabdpt_regressor_evaluation_report.json` with verdict `sample-sanity`, both baselines and the
     interpretation;
   - `predict`: `outputs/tabdpt_regressor_predictions.csv`, `outputs/tabdpt_regressor_new_rows.csv` and
     `outputs/tabdpt_regressor_result.json` with the runtime identity and device;
   - `export` / `reload`: `outputs/artifact/{artifact.json,training_context.parquet}`, the printed artifact digest and
     `matches_pinned_sample_artifact` (record it; `True` confirms the companion's pinned sample), then the fresh-process
     reload with `preprocessing_restored_ is True` and predictions within `rtol=1e-5, atol=1e-6`;
6. in a **second** fresh runtime, run the exact companion notebook revision with every field at its default (no upload):
   verify the trusted digest of the pinned sample is `verified`, `validate_artifact_bundle` passes before
   reconstruction, `preprocessing_restored_ is True` and the target column matches, the input manifest has one recorded
   rejection finding, and the predictions, the `not-measurable` evaluation report (`sample_kind: sample`) and the result
   JSON are written; then the REL12 journey: one compatible user input (the E2E run's `outputs/artifact/` with its
   printed digest in `EXPECTED_ARTIFACT_SHA256` and its `outputs/tabdpt_regressor_new_rows.csv` with
   `ID_COLUMNS = ['row_id']`) and one incompatible input (a digest with one changed character, or rows with a missing
   column), plus, in the E2E notebook, one compatible and one incompatible BYOD CSV (blank or non-numeric targets) through
   `BYOD_PATH` or the upload dialog;
7. verify the exports exist and the interpretation sections match the observed paths;
8. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabdpt, device), `restarted`,
   model identifier and immutable revision, whether the model cache was clean, outcome, the printed metrics, the
   artifact digest, and any warning or applicable `SHOULD` deviation in the table below;
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