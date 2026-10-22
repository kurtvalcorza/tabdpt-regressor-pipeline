# Release verification

`tutorials/tabdpt_regressor_colab.ipynb` (`E2E`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 3.0. This file is the durable release-gate record for the notebook. Under NOTEBOOK_SPEC 3.0 (KND11) the
former `ARTIFACT-INFERENCE` companion, `tutorials/tabdpt_regressor_artifact_inference_colab.ipynb`, was folded into the
E2E notebook's closing external-artifact section (Section 10, §19.1); its rows below remain as historical records.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks, for the notebook:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook (no separate artifact-inference notebook, KND11), named in `tutorials/README.md` with its
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `3.0`, kind `TUTORIAL`, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating commit, package module paths and SHA-256, carried-file digests,
  generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and the isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose
  `CARRIED_FILES` / `CARRIED_BINARY` equal the repository's package (`src/tabdpt_regressor_pipeline/`), the notebook's
  stage runner (`tools/tutorial_stages.py`), the hash lock
  `tutorials/requirements-colab.lock.txt` (which must pin every `pyproject.toml` runtime pin, every entry hashed), the
  committed snapshot manifest and the licence; `CARRIED_HASHES` match;
  the notebook is byte-identical to `tools/build_notebook.py` output for its template; no cell pip-installs into the
  notebook kernel and no text asks for a runtime restart; the install cell uses a pinned `uv` wheel, a managed CPython,
  `--require-hashes`, a lock-digest-keyed environment that is reused, `MPLBACKEND=Agg`, and drops
  `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`; the four Infrastructure cells are collapsed (`cellView: form`); every
  learner cell runs a stage;
- the profile-specific public-API calls in the carried stage runner — E2E: `from_pretrained(weights_dir=..., compile_model=False,
  use_flash=False, seed=SEED)`, `validate_inputs`, `training_mean_baseline`, the linear-regression reference with its
  split-to-split R² range, `fit`, `evaluate`, `evaluation_report`, `predict`,
  `export_artifact_bundle`, `load_verified_artifact` in a fresh process with the no-refit and equivalence checks, the
  BYOD refusals for an absent, blank or non-numeric target and for a table below 10 rows, the deterministic export ZIP
  (`deterministic_zip`: sorted members, fixed 1980-01-01 timestamp, mode 0644); the external-artifact section: the
  exact pinned `SAMPLE_ARTIFACT_URL` / `SAMPLE_ARTIFACT_SHA256`, the digest check before extraction, the member-by-member
  safe extraction, `validate_artifact_bundle`, the SART10 data-contract check (feature columns and target column),
  `load_verified_artifact(..., model_weight_path=<verified checkpoint>, compile_model=False, use_flash=False)` with the
  fail-closed legacy-path and target-column checks, and outputs under `outputs/external_artifact/` — the exports, the learner-facing
  regression statements and the guided layer (how-to-use, task contract, roadmap, glossary, predictions,
  checkpoints, troubleshooting, conclusion), and the optional-input gates at their non-interactive defaults; no
  `assert` in the stage runners (verdicts are reported, contract checks raise with a message); forbidden patterns
  (credential-in-URL, any `git clone`, any `github.com/kurtvalcorza` reference other than the exact pinned sample-artifact
  URL in the stage runner — compared as a whole token, refusing `..`, `%2e`, `?` and `#` (ST1) — a mutable `revision='main'`,
  direct `tabdpt` / `huggingface_hub` / `sklearn` / `torch` use in the notebook's own cells, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `tools/build_notebook.py --check`, `scripts/validate_repo.py`, `scripts/validate_colab_tutorial.py` and the
offline unit suite (`tests/`, including `test_notebook_parity.py`, `test_notebook_review_fixes.py`, which execs the
notebook's own kernel cells with stand-ins and runs the model-free stages, and `test_external_artifact_section.py`, which
checks the pin, the digest refusal before extraction, the deterministic ZIP, the SART10 contract, the setup-only
dependency of the section's cells and the ST1 exact-URL rule; injected downloader, no weights, no model). These are
source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab T4 GPU runtime (CPU also works); the kernel's Python does not matter — the stages run in an isolated CPython 3.12.12 environment | The runtime the tutorials are written for; a clean one-pass top-to-bottom run here is promotion evidence |
| Kaggle kernel | Kaggle T4 kernel | Reproducible clean-room executor of the same class; the notebook is pushed verbatim (no repository checkout is needed — the notebook is standalone, and its external-artifact section downloads the pinned asset itself) |
| Local harness (pre-flight only) | Workstation | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence. `scripts/execute_notebook_release.py` was written for the previous repository-installing pair and does not apply to the /3 notebooks |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a **fresh** Colab T4 runtime (or the Kaggle executor) with no repository
   checkout and a clean model cache;
3. choose **Run all** once, with every form field at its default (`USE_BYOD = False`, `ARTIFACT_SOURCE = 'Pinned sample artifact'`,
   `USE_OWN_ROWS = False`, `RUN_ACTIVITY = False`); no
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
   - `export` / `reload`: `outputs/artifact/{artifact.json,training_context.parquet}` and
     `outputs/tabdpt_regressor_artifact.zip`, the printed `zip_sha256` (record it; on a Linux runtime's default path at an
     unchanged package and lock it is expected to equal the pinned `66f3d162…d244`, and a difference is recorded, not an
     error — Section 10 never uses this run's export), then the fresh-process reload with
     `preprocessing_restored_ is True` and predictions within `rtol=1e-5, atol=1e-6`;
   - `external_artifact` / `external_reconstruct` / `external_predict` (Section 10): the provenance line printed before
     the download, `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256)`, `data_contract` matching the diabetes
     default path, `preprocessing_restored_: True`, eight predictions for `row_id` 287, 211, 72, 321, 73, 418, 367, 354,
     and `outputs/external_artifact/tabdpt_regressor_external_{predictions.csv,result.json}`;
6. the REL12 journey in the same notebook: one compatible and one incompatible BYOD CSV (blank or non-numeric targets)
   through `BYOD_PATH` or the upload dialog; in Section 10, one compatible own input (this run's
   `outputs/tabdpt_regressor_artifact.zip` with its printed digest in `EXPECTED_ZIP_SHA256` and
   `outputs/tabdpt_regressor_new_rows.csv` in `EXTERNAL_ROWS_PATH`) and one incompatible one (a digest with one changed
   character), then switch Section 10 back to its defaults;
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
| 2026-09-14 | `46c5e17` / `0611e07348b4` | Kaggle T4 (`kurtvalcorza/dimer-nb2-tabdpt-regressor` v1) | Default sample path | 231.7 s | **PASSED** — 11/11 ok code cells executed cleanly, 4 files, 254 MB staged (previous E2E blob, superseded; outcome only — no metrics, package versions or `restarted` flag were recorded) |
| 2026-10-10 | `e54340a` / `7c878eb19872` of `tabdpt_regressor_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabdpt_regressor_colab_e54340a_colab-cli-t4.ipynb`, sha256 `39f10ab9a8b9c0c126cda21724151afc0044c70434b0565c6cd6212f7ef2b13f` (`…_run_summary.json` `99bb8739f90c…`, `…_exec.log` `063393aab372…` (redacted 2026-10-22, now `3b149ef2633f…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (15,360 MiB, reported by cell 1), via the workspace `colab-cli-serial-test-suite`; isolated uv CPython 3.12.12 environment from the carried hash lock (55 packages; torch 2.7.1, tabdpt 1.2.0, numpy 2.3.0, pandas 2.3.2, scikit-learn 1.7.0). Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/11–11/11) | E2E default path: Diabetes sample, 353 support / 89 holdout rows, in-context conditioning only (no fine-tuning), device cuda; weight `tabdpt1_2.safetensors` fetched from the Hub at revision `4462ffbd1d8d` and digest-verified (`06680220…`) | 171.0 s session wall | **PASSED — one pass, no restart, 0 errors, 11/11 code cells.** Missing-target probe rejected; TabDPT holdout MAE 40.9804, RMSE 51.6813, R² 0.4959 — identical to the CPU reference to 4 decimals; training-mean baseline 64.0065 / 73.2225 / −0.012; standardised linear regression 42.7941 / 53.8534 / 0.4526; linear R² over 20 seeded splits 0.3322–0.5851 (median 0.4596). `matches_pinned_sample_artifact: True` — the exported artifact's SHA-256 `0f75e2582eb283e9…` equals the pinned sample's. Fresh-process reload PASSED: preprocessing restored without refit, max abs prediction difference 0.0. Optional activity not run. Carried provenance names `df095f8` (generation-time HEAD, not `e54340a`). Status stays **Candidate** |
| 2026-10-10 | `e54340a` / `ec42aa44be9f` of `tabdpt_regressor_artifact_inference_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabdpt_regressor_artifact_inference_colab_e54340a_colab-cli-t4.ipynb`, sha256 `8043429ae96f68ac65745f643098153a1dff0acd5d02fc6e3a4ca39c6891c2cf` (`…_run_summary.json` `7621aa13118b…`, `…_exec.log` `63cbdba3d473…` (redacted 2026-10-22, now `fb010fe3c94c…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`; same 55-package isolated uv environment. Not a browser Run all; order from the CLI log (1/9–9/9) | Companion default path: the carried sample artifact (no release, no upload), base weight fetched from the Hub and digest-verified | 107.7 s session wall | **PASSED — one pass, no restart, 0 errors, 9/9 code cells.** No upload in any output; trusted digest verified (`0f75e2582eb283e9…`, format `tabdpt-dimer-context-v3`); preprocessing restored without refit; missing-column probe rejected (`Feature schema mismatch; missing=['age']`); the 8 predictions agree with the E2E run's to 2 decimals, the E2E's printed precision (a finer comparison is not visible in the outputs). Optional activity not run. Carried provenance `df095f8`. Status stays **Candidate** |
| 2026-10-22 | `1e816d9` / `7c878eb19872` of `tabdpt_regressor_colab.ipynb` (main, before NOTEBOOK_SPEC 3.0); executed file `docs/execution-evidence/2026-10-22-1e816d9/tabdpt_regressor_colab_1e816d9_colab-cli-t4.ipynb`, sha256 `0a498ae0…37eb` | Colab CLI 0.7.4 `colab exec -f`, fresh T4 session | Default sample path | 111.8 s | **PASSED** — 11/11 code cells, no error, one pass. TabDPT holdout MAE 40.9804 / R² 0.4959; training mean MAE 64.0065 / R² −0.012; linear regression MAE 42.7941 / R² 0.4526; fresh reload PASS (no refit). Export `artifact.json` `0f75e258…c783`, `training_context.parquet` `186ce81d…4b28`. **This run produced the pinned sample artifact:** the VM script `…_pack.py` zipped its `outputs/artifact/` (sorted members, 1980-01-01, ZIP_DEFLATED, mode 0644, force_zip64) into `tabdpt_regressor_artifact.zip`, SHA-256 `66f3d162096646300bfa560d92660314f0f62d4b2d3a5d43c3c7338f5a31d244`, 14,960 bytes, published as the release asset `sample-artifact-diabetes-v1` (SART8). The exporter's `deterministic_zip` rebuilds that digest byte for byte from the asset's members. |

Captured → committed SHA-256 of the 2026-10-22 evidence files (local run directory redacted to `<local-run-dir>/`):

| File | Captured | Committed |
|---|---|---|
| `tabdpt_regressor_colab_1e816d9_colab-cli-t4.ipynb` | `0a498ae03766cafbb84c64520d312acef3d58ef6ff29f7f75ffa44fb3b3137eb` | unchanged |
| `tabdpt_regressor_colab_1e816d9_colab-cli-t4_exec.log` | `49cd9b91ebaa0bbc340b1b5148d7404804d7164d36271109b71201b098ede599` | `5e9b9c2e2f32b3052f83d22f2254dd5b2b30df9ac7dd3d62197837a4e205219a` |
| `tabdpt_regressor_colab_1e816d9_colab-cli-t4_run_summary.json` | `54a6bf35bebb6a66254881ef8f94484de97ff20d281eea07d2384b0ff70c9ba8` | unchanged |
| `tabdpt_regressor_colab_1e816d9_colab-cli-t4_pack.py` | `8522908d82414e770d13a7b0d7ea799f21f3ae0ab1d70b30d938e52f320dcc02` | unchanged |
| `tabdpt_regressor_colab_1e816d9_colab-cli-t4_suite.log` | `22a61d426ee3d613d6a8806560fbc11baf424c2d371259c9bc36412f8f3ec44c` | `832d2fda28c350c5d0d2ba9b95b5efdc6c8208dd002f61acfdc7bf835b164d43` |

### CI integration execution (not clean-runtime evidence)

The pull request's `Notebook integration execution` job (`.github/workflows/notebook-release.yml`) ran both notebooks top to
bottom with nbconvert on a GitHub-hosted `ubuntu-24.04` CPU runner, each stage in the environment the notebook itself built
from `tutorials/requirements-colab.lock.txt` (CPython 3.12.12, `torch` 2.7.1+cu126, `tabdpt` 1.2.0), with the real checkpoint
downloaded from the Hub at the pinned revision. The companion was parametrised to take the E2E run's artifact by the CI
upload shim with its trusted digest, so its default (pinned-sample) path was not exercised there. It is not a Colab run and
does not count toward promotion.

| Date (UTC) | Commit / notebook blob | Path exercised | Outcome |
|---|---|---|---|
| 2026-10-05 | `a67ffd2` / `7c878eb19872` (E2E) | Default sample path, CPU | 11/11 code cells, no error. TabDPT holdout MAE 40.98 / RMSE 51.68 / R² 0.496; standardised linear regression 42.79 / 53.85 / 0.453 (R² 0.332–0.585 over 20 seeded splits); training mean 64.01 / 73.22 / −0.012. Exported `artifact.json` SHA-256 `0f75e258…c783` and `training_context.parquet` `186ce81d…4b28`, byte-identical to `examples/sample-artifact/` (`matches_pinned_sample_artifact: True`) |
| 2026-10-05 | `a67ffd2` / `ec42aa44be9f` (companion) | The E2E run's artifact by upload with its trusted digest, eight independent rows by upload, CPU | 9/9 code cells, no error |

**First hosted run of the migrated notebook (2026-10-22, `c61b0f0` / blob `b00122574c48`, Notebook Spec 3.0; the second pass of
§32.2).** Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4, default path of `tutorials/tabdpt_regressor_colab.ipynb` on
PR #25, started 2026-10-22T04:01:11Z, wall 124.6 s, exit 0, input unchanged; executed cell sources equal the blob (44 cells);
verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-c61b0f0/`. The fresh reload restored the fitted preprocessing
without refit. **Section 10 ran end to end on the default path:** it downloaded the release asset of tag
`sample-artifact-diabetes-v1`, verified SHA-256 `66f3d162096646300bfa560d92660314f0f62d4b2d3a5d43c3c7338f5a31d244` before
extraction, rebuilt the regressor from the files alone and wrote `outputs/external_artifact/tabdpt_regressor_external_predictions.csv`
and `..._external_result.json` (printed in the executed notebook; the files stayed on the released VM). The run's own
deterministic export, `outputs/tabdpt_regressor_artifact.zip`, printed the same digest `66f3d162…` as the pinned asset: the
exporter reproduces the published artifact byte for byte on the hosted runtime. Local paths were replaced by `<local-run-dir>/`
before the first commit:

| File (under `execution-evidence/2026-10-22-c61b0f0/`) | Captured | Committed |
|---|---|---|
| `tabdpt_regressor_colab_c61b0f0_colab-cli-t4.ipynb` | `66a9e2e8aba18845e049bd611c28de05fd830a8c142d0ae2e11a54a5c5a8351d` | unchanged (no local path) |
| `tabdpt_regressor_colab_c61b0f0_colab-cli-t4_run_summary.json` | `d99abe4d78e2bb9ba88918200ed7a4f3221b3438c78a6a08114e5895c89defa2` | unchanged (no local path) |
| `tabdpt_regressor_colab_c61b0f0_colab-cli-t4_exec.log` | `4bd5fe898a0d201be1c5850115fade392b71cb5fe32b86032f052edbf94bbef2` | `e0025d564f9a82e3a5e7abde8b7009313a054c2d1b2417113d164c133925fb19` |
| `tabdpt_regressor_colab_c61b0f0_colab-cli-t4_suite.log` | `bcc39a5fb8cd98a88f166ffed4e7a086f583fb8cda1d5e0600ba72cc59b2bc84` | `9a77e009c4dda4f4649a16f3bd826c03eca3067b786fe29906b1ff8a719b969a` |

**In-run export identity run (2026-10-22, `b869ab5` / blob `23e0a7eaa0d1`, PR #27; SART9).** Colab CLI 0.7.5 `colab exec -f` on a fresh Colab Tesla T4, default path of `tutorials/tabdpt_regressor_colab.ipynb` at the head of `spec3/sart9-inrun-zip`, started 2026-10-22T12:53:36Z, wall 160.2 s, exit 0, input unchanged, 14/14 code cells, no error; verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-b869ab5/`. The export stage wrote `outputs/tabdpt_regressor_artifact.zip` in-run and recorded `artifact.zip_sha256` `66f3d162096646300bfa560d92660314f0f62d4b2d3a5d43c3c7338f5a31d244` in the run's result file. That digest equals the release asset of `sample-artifact-diabetes-v1` byte for byte, as the export of the `c61b0f0` run did. Section 10 downloaded the pinned asset and verified the same digest before extraction. Local paths were replaced by `<local-run-dir>/` before the first commit:

| File (under `execution-evidence/2026-10-22-b869ab5/`) | Captured | Committed |
|---|---|---|
| `tabdpt_regressor_colab_b869ab5_colab-cli-t4.ipynb` | `3133770ac6c6d77d04c141b9d27cc9a8c2fc1f5a37b8f862fe4f5a52e32dc297` | unchanged (no local path) |
| `tabdpt_regressor_colab_b869ab5_colab-cli-t4_exec.log` | `ee648c28ea9a4683243d4b61ff506fc7bfdfa8bfafc4a450049f1b07c043789c` | `9e08a6b5d5980e5d032c5b976055c79eabeeee7b583011bba0dd8c365de1c984` |
| `tabdpt_regressor_colab_b869ab5_colab-cli-t4_run_summary.json` | `66db956e190dbb8aaf9c5db8e90b090903668aa1943760548c782d63ee06933b` | unchanged (no local path) |
| `tabdpt_regressor_colab_b869ab5_colab-cli-t4_suite.log` | `34bba8ec33c9239a2f7fbde233387af60e97107c358dde0cd841f72e86e16e41` | `b55feb938ee0ff5d276eafe6b277f74c493146b91eaaff3ebf97ff88decffc07` |

**Optional-activity run (2026-10-22, `7d4b6f3` / committed blob `23e0a7ea`, `RUN_ACTIVITY = True`).** A run copy of committed blob `23e0a7ea` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `10fc750969dc`, SHA-256 `7a13ee476c65cd7e6f40caab9ae2e5ab617bcd642b7956c1b6670cd9bf5bc58a`; it differs from the committed blob only in that assignment literal and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser Run all) on a fresh Colab Tesla T4 at `main` `7d4b6f3`, started 2026-10-22T13:36:23Z, wall 214.6 s, exit 0, sequential, input (the run copy) unchanged, 14/14 code cells, no error; verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-7d4b6f3-activity/`. The Section 11 activity (`n_ensembles` 2 → 8) printed, on 89 holdout rows, MAE 40.9804 → 41.055, RMSE 51.6813 → 51.7115, R² 0.4959 → 0.4953, and `{'canonical_outputs_unchanged': True, 'files_checked': 8}`. Local paths were replaced by `<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-22-7d4b6f3-activity/`) | Captured | Committed |
|---|---|---|
| `tabdpt_regressor_colab_7d4b6f3_colab-cli-t4_activity.ipynb` | `503175bd4b87e56dc204bace0e85667edfce0f2aa3439e9dd0045582803f5be2` | unchanged (no local path) |
| `tabdpt_regressor_colab_7d4b6f3_colab-cli-t4_activity_exec.log` | `8086b8081fcc5aebb668f18403b4e4f5b860be44c88d4380a8625c98bd272a30` | `7dcb9578e24313d770d7befcd5c2ee74cd3239cbfa636da7c1c7493499e4c364` (2 replacements) |
| `tabdpt_regressor_colab_7d4b6f3_colab-cli-t4_activity_run_summary.json` | `96ece6dfd94abb6c6cec3375787404d727d83a8e5d5b255d48f9e881df7ffabe` | unchanged (no local path) |
| `tabdpt_regressor_colab_7d4b6f3_colab-cli-t4_activity_run_copy.ipynb` | `7a13ee476c65cd7e6f40caab9ae2e5ab617bcd642b7956c1b6670cd9bf5bc58a` | unchanged (no local path) |

### Local lock-only drives (not clean-runtime evidence)

These ran the notebooks' own code cells in order (nbclient, a separate ipykernel venv) in a Linux x86_64 WSL2 container,
CPU only (`CUDA_VISIBLE_DEVICES=-1`), with every stage in the environment the notebook itself built from
`tutorials/requirements-colab.lock.txt` (55 packages, `torch` 2.7.1+cu126, `tabdpt` 1.2.0). One local deviation: the pinned
checkpoint was pre-seeded into `weights/tabdpt-1.2/` after a SHA-256 check (this container's IPv6 route to the Hub stalls);
the stage then reported `fetched: []` and re-verified it. They are not hosted runs and do not count toward promotion.

| Date (UTC) | Commit / notebook blob | Path exercised | Wall | Outcome |
|---|---|---|---|---|
| 2026-10-11 | `a67ffd2` / `ec42aa44be9f` (companion) | Defaults: pinned sample artifact and rows, no upload | 453 s (incl. 391 s environment build) | 9/9 code cells, no error. `trusted_digest: verified` (`0f75e258…c783`), `sample_kind: sample`, report `not-measurable`; the eight predictions equal the CI E2E run's (max abs difference 4.6e-5; row 287: 152.0143) |
| 2026-10-11 | `a67ffd2` / `7c878eb19872` (E2E) | Default sample path | 36 s (environment reused) | 11/11 code cells, no error. TabDPT holdout MAE 40.9804 / RMSE 51.6813 / R² 0.4959 (equal to the CI run); `matches_pinned_sample_artifact: True` |

## Current status

**Current status (2026-10-22): Candidate.** Recorded: the hosted Colab T4 run of blob `23e0a7eaa0d1` at `b869ab5` (PR #27; Colab CLI, not a browser Run all) passed the default path end to end, including Section 10 with the v1 pin (see "In-run export identity run" above). The optional activity ran once on the same blob (see "Optional-activity run" above: a `RUN_ACTIVITY = True` run copy at `7d4b6f3`, Colab CLI 0.7.5, fresh T4, PASS), which satisfies the owner's 2026-10-22 ruling that it runs before promotion. The earlier run of blob `b00122574c48` at `c61b0f0` is history for the current blob (REL14). The notebook stays Candidate pending:

1. **REL12:** a BYOD run that accepts one representative input through the downstream stages and refuses one
   incompatible input;
2. **REL2:** a run in the hosted runtime's browser interface for the interactive journeys (BYOD upload, the optional
   activity toggle, the own-artifact ZIP).

**SART9 is met by identity (owner ruling, 2026-10-22).** The release asset of `sample-artifact-diabetes-v1` was packed on the VM after the `1e816d9` run from that run's exported `outputs/artifact/`. Since then the notebook's own export stage writes the deterministic ZIP in-run (sorted members, 1980-01-01 00:00:00, mode 0644, `create_system` 3, atomic write) and records its SHA-256 under `artifact.zip_sha256`. The in-run ZIP of the hosted runs at `c61b0f0` and `b869ab5` is byte-identical to the v1 asset (`66f3d162…`), so the pinned bytes are the export of the default path of a recorded run. The owner chose to keep the v1 pin rather than republish identical bytes under a new tag.

**SART7 deviation (2026-10-22).** `sample-artifact-diabetes-v1` is a mutable release: this repository does not enable immutable releases. The notebook pins the asset's SHA-256 (`SAMPLE_ARTIFACT_SHA256` = `66f3d162096646300bfa560d92660314f0f62d4b2d3a5d43c3c7338f5a31d244`), and Section 10 refuses the download before extraction unless the digest matches, so a replaced asset cannot be used silently. The owner accepted this deviation on 2026-10-22.

**History, 2026-10-10 — one-pass hosted runs of both blobs current at that time (the 2.2-era pair) were recorded** (the 2026-10-10 rows of the clean-runtime table): `tabdpt_regressor_colab.ipynb` (`7c878eb19872`) and `tabdpt_regressor_artifact_inference_colab.ipynb` (`ec42aa44be9f`) at `e54340a`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the T4 metrics are identical to the CPU reference to 4 decimals (MAE 40.9804 / RMSE 51.6813 / R² 0.4959), and the exported artifact's SHA-256 equals the pinned sample's (`0f75e258…`); (b) the companion's 8 predictions agree with the E2E run's to 2 decimals, the E2E's printed precision — a finer comparison is not visible in the outputs; (c) both carried revision labels read `df095f8` (generation-time HEAD, not `e54340a`) — follow-up; (d) the optional activity was not run; (e) the maintainer accepted the carried sample (no release; Apache-2.0, no weights; byte-identical to a real CI export at `a67ffd2`). Both notebooks stay **Candidate**; remaining gates: reviewer confirmation of these runs, the BYOD journeys and the optional activities.

Before 2026-10-10: no clean-runtime execution of the current standalone notebook blobs had been recorded (the 2026-09-14 Kaggle row is
the previous E2E blob). The CI integration execution above
is the first real execution of the standalone path, the checkpoint download and the helper stages, on CPU only. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blobs under review and
an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh: the real `hf_hub_download` fetch of `tabdpt1_2.safetensors`, the standalone carrier, the
helper stages (`validate_inputs`, `training_mean_baseline`, `evaluation_report`) and the first real load of the checkpoint
inside `fit` have run only in the CPU CI integration execution above, never on a GPU or in Colab; no execution of the previous
notebook pair was ever recorded (issue #15); and the companion's default pinned-sample path has run only in the local lock-only drive above, so the Colab T4 run
will be its first hosted execution.

## Local-path redaction of earlier evidence (2026-10-22)

The files below were committed with local absolute paths in them. Each local run-directory prefix was replaced with `<local-run-dir>/`, the convention the evidence committed from 2026-10-22 on already uses. No other byte changed and line endings are unchanged. Git history still holds the original files. "Before" is the SHA-256 of the file as committed on `main` at `467b733`.

| File | Replacements | SHA-256 before | SHA-256 after |
| --- | --- | --- | --- |
| `docs/execution-evidence/2026-10-10/tabdpt_regressor_artifact_inference_colab_e54340a_colab-cli-t4_exec.log` | 2 | `63cbdba3d47308638f90e4de411d0f41a4a03613ec4fce53cab28bacf645088f` | `fb010fe3c94cc60488f0ba1a8e42f07ae03a7ca80110e03a1fc1838e59e844df` |
| `docs/execution-evidence/2026-10-10/tabdpt_regressor_colab_e54340a_colab-cli-t4_exec.log` | 2 | `063393aab37257f75e1689c31694966a4cd9fd99a0386a3b263a553506d6966b` | `3b149ef2633f8c92c1f8aa5466a3154d4084531e4f6685febf80432ed2db6746` |

The Recorded executions table above quotes the pre-redaction hash of 2 of these logs as the hash of the committed file. Those quotes are kept as recorded, and the post-redaction hash is noted beside each one.
