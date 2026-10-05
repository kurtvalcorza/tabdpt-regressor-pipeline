"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment) — E2E.

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the guided layer and the learner
cells. Every learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process
in an isolated, hash-locked environment, so nothing is installed into the notebook kernel. The ARTIFACT-INFERENCE
companion has its own template, ``tools/notebook_template_artifact_inference.py``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "tabdpt-regressor-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabdpt_regressor_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Layer6%2FTabDPT-ffcc4d?style=flat",
        "https://huggingface.co/Layer6/TabDPT",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-layer6ai--labs%2FTabDPT--inference-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/layer6ai-labs/TabDPT-inference",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2608.01400-b31b1b.svg", "https://arxiv.org/abs/2608.01400"),
]

# Shared by both templates: the isolated-environment pins (the fleet's uv wheel, a managed CPython).
UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}
ENVIRONMENT = {
    "package": "tabdpt_regressor_pipeline",
    "repo_name": REPO,
    "weights_key": "tabdpt-1.2",
    "modules": ["__init__.py", "pipeline.py", "artifact.py", "dimer_runtime.py"],
    "entry_module": "pipeline.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.3, "environment": 7.0},
    "runtime_modules": ["torch", "tabdpt", "numpy", "pandas", "scikit-learn"],
    # `antlr4-python3-runtime` (required by omegaconf, required by tabdpt) is published only as a source archive; it
    # is still hash-pinned in the lock and is the one package built from source.
    "install_flags": ["--only-binary", ":all:", "--no-binary", "antlr4-python3-runtime"],
}

RUNTIME_PREREQ = (
    "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab with a **T4 GPU** is the documented runtime; Kaggle or a Linux Jupyter kernel also work, and a CPU-only runtime works more slowly. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`tabdpt` 1.2.0, `torch` 2.7.1 with its CUDA 12.6 libraries, `numpy` 2.3.0, `pandas` 2.3.2, `scikit-learn` 1.7.0). About 0.3 GB of disk is needed for the checkpoint and about 7 GB for the isolated environment. FlashAttention is disabled (`use_flash=False`) for Tesla T4 portability."
)

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabdpt_regressor",
    "notebook_name": "tabdpt_regressor_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "carried_extra": {"sample-artifact/SAMPLE_ARTIFACT.json": "examples/sample-artifact/SAMPLE_ARTIFACT.json"},
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a T4 GPU is recommended) builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage below in its own process: it stages and digest-verifies the pinned TabDPT checkpoint, loads scikit-learn's bundled diabetes table (no download), validates it into an input manifest (finite numeric target) and splits it, compares a training-mean baseline and a standardised linear-regression reference, **adapts TabDPT by in-context conditioning on the training split** (the adaptation stage: TabDPT registers the support rows; it has no gradient fine-tuning route), reports the capacity used, evaluates on the held-out split and writes the evaluation report, scores new rows and writes machine-readable outputs, exports the serving artifact and reloads it in a fresh process through `load_verified_artifact`. No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5). No hosted run of this revision has been recorded yet."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 and run Sections 4–9 again to use one labelled CSV or Parquet file of your own: set `BYOD_PATH` to a file already in the runtime (works in Colab, Kaggle and Jupyter), or leave it empty in Colab to choose the file in an upload dialog. Name your numeric target column in `TARGET` and any identifier-like categorical columns in `CATEGORICAL_COLUMNS`. The table needs at least 10 rows, so that the 20 % holdout has two rows with different targets. A missing, blank, non-numeric or infinite target value is refused in Section 4 with the count and examples, before anything is written to `outputs/`. It then enters the same validation, split, baselines, in-context conditioning, evaluation, new-data inference, export and fresh-reload stages as the sample. The file stays inside this runtime. BYOD is optional and never part of the default path."
    ),
    "title": "TabDPT Regressor — DIMER E2E tabular regression tutorial (standalone)",
    "badges": BADGES,
    "capability": "supervised tabular regression by in-context conditioning on labelled support rows with the pinned `Layer6/TabDPT` v1.2 checkpoint",
    "intro": (
        "TabDPT (Ma et al.) is a **tabular foundation model**: a transformer pretrained on many real tables to predict a "
        "row's target from a set of labelled example rows shown to it at prediction time, the **support rows** or "
        "**context**. Using it on a new table therefore needs no gradient training: `fit()` fits the repository's feature "
        "encoder and registers the labelled support rows as context — **in-context conditioning**. It does **not** "
        "gradient-train or fine-tune the pretrained weights. For regression TabDPT predicts a distribution over binned target "
        "values and reports its expectation, averaged over `n_ensembles` passes: one **point estimate** per row.\n\n"
        "The upstream project supplies TabDPT and its checkpoint; the carried package adds immutable provenance, snapshot "
        "verification, schema-safe preprocessing, deterministic controls, the DIMER serving-artifact contract, regression "
        "evaluation, and the `validate_inputs`, `training_mean_baseline` and `evaluation_report` helpers. The default sample "
        "is scikit-learn's public diabetes table (442 patients, 10 standardised measurements, disease progression one year "
        "later). It is small and noisy: a standardised linear regression explains about 45 % of the holdout variance on this "
        "split, and its R² moves from 0.33 to 0.59 across seeded splits, so the notebook reports that classical reference and "
        "its range beside TabDPT's metrics. The metrics are tutorial sanity evidence, not a benchmark or production claim, "
        "and upstream pretraining overlap with this public table cannot be ruled out."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** how an in-context tabular model predicts a number from labelled support rows, and why `fit` here is "
        "conditioning, not training (Sections 4, 6).\n"
        "2. **Diagnose** an invalid table from a validation refusal, including a missing or non-numeric target and a table "
        "too small to evaluate (Sections 4, 5).\n"
        "3. **Compare** TabDPT with a training-mean baseline and a classical linear-regression reference, using the "
        "reference's split-to-split R² range rather than one number (Sections 6, 7).\n"
        "4. **Interpret** MAE, RMSE and R², in target units and relative to the target's spread (Section 6).\n"
        "5. **Apply** the conditioned model to new rows and read point predictions beside the true targets (Section 8).\n"
        "6. **Verify** that an exported serving artifact reproduces the predictions after a fresh reload without refitting preprocessing (Section 9).\n"
        "7. **Predict**, run and **explain** the effect of one change — the number of ensemble members — in an optional activity (Section 10).\n"
        "8. **Write** an evidence-based conclusion that names the baselines, the split-to-split range and the limits of one small public table (Conclusion)."
    ),
    "exclusions": (
        "classification, gradient fine-tuning, or calibrated per-prediction uncertainty intervals. Predictions are "
        "**continuous point estimates only**; the pipeline produces no prediction interval, and any tolerance or "
        "decision band is the caller's to set on labelled data. Nor does it show benchmark superiority: on 89 holdout "
        "rows R² moves by more than 0.2 between seeded splits."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic pandas (a DataFrame, a column, `describe`), what a train/holdout split is, and how to read a printed Python dictionary. MAE, RMSE, R² and in-context conditioning are explained where they are first used, and the glossary collects them.",
        "- **Model file:** one checkpoint, `tabdpt1_2.safetensors` (safetensors: a tensor container read without unpickling, so no code is executed when it is loaded), released under Apache-2.0.",
        "- **Data:** the default sample is scikit-learn's bundled diabetes table (442 rows, 10 numeric features already centred and scaled by the dataset's authors, a continuous disease-progression target), loaded from the installed package, so nothing is downloaded and no private data is needed. Optional BYOD is switched off by default so the sample path runs top to bottom without interaction. Expected BYOD input: one CSV or Parquet file with unique column names, a target column named by `TARGET` (default `target`) holding finite numbers (no thousands separators), at least 10 rows and at least two distinct target values in the holdout. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook (Google Colab or Jupyter) and read "
                "short Python and pandas, and who want to see how a pretrained tabular foundation model is applied to a regression "
                "task, evaluated honestly against simple baselines, and packaged for reuse. No experience with transformers or "
                "in-context learning is assumed; each term is explained where it is first needed, and the glossary below collects "
                "them.\n\n"
                "**Running it.** In Colab choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default "
                "path needs no edit, no upload, no account, no token and no runtime restart. Section 2 builds an isolated "
                "environment, which takes the longest (PyTorch and its CUDA libraries are a few GB); the model stages that follow "
                "take seconds each on a T4. You can also run one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files: the verified checkpoint, the validated table and "
                "split, and JSON records. A stage that needs the conditioned model conditions it again from the saved split — "
                "conditioning is fast and deterministic for a fixed seed.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–10) are the machine-learning workflow; each runs one stage and "
                "prints compact dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the runtime check, the carried "
                "code, the isolated environment and the checkpoint staging) are collapsed and titled **Infrastructure**. You may run "
                "them without studying their implementation: they exist for reproducibility and provenance, not as prerequisite "
                "machine-learning knowledge.\n\n"
                "**Form controls.** Some learner cells start with fields that Colab renders as a form: `USE_BYOD`, `BYOD_PATH`, "
                "`TARGET` and `CATEGORICAL_COLUMNS` (Section 4), `N_ENSEMBLES` and `CONTEXT_SIZE` (Section 6), and `RUN_ACTIVITY` "
                "and `ACTIVITY_N_ENSEMBLES` (Section 10). Leave them at their defaults for the first run: the notes and sample "
                "answers describe the default path.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. "
                "**[Evaluation practice]** — how the evidence is produced and how to read it. **[Engineering]** — reproducibility, "
                "provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6 and 9 a **Predict before running** prompt asks you to commit to an "
                "expectation; after each stage, **What to notice** describes normal output; a collapsed **Check your reasoning** "
                "answer follows each checkpoint. Write your own answer first, then open it. GPU kernels are not bitwise "
                "deterministic, so TabDPT's last digits can vary between runs; the notes describe the shape of a normal result."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate and split** | a labelled table: feature columns + a numeric `target` column | `validate_inputs`, a seeded random 80/20 split | an input manifest; 353 support rows and 89 holdout rows |\n"
                "| **Condition** | the support rows | the repository's feature encoder; TabDPT's support-fitted mean imputer and standard scaler; the support rows registered as context | a conditioned pipeline (no weights change) |\n"
                "| **Predict** | query rows with the same feature columns | the pinned TabDPT checkpoint reading up to `context_size` support rows, `n_ensembles` passes | one point estimate per row, in target units |\n"
                "| **Evaluate** | the holdout rows and their targets | `evaluate`, `training_mean_baseline`, a linear-regression reference | MAE, RMSE, R²; an evaluation report |\n"
                "| **Package** | the conditioned pipeline and its support rows | `export_artifact_bundle`, `load_verified_artifact` | `artifact.json` + `training_context.parquet`, reloaded without refitting |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64, GPU and disk checked; a run directory | the machine and directories |\n"
                "| 2. Carry the code, build the environment | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify the model | [Engineering] | the 254 MB checkpoint downloaded at a fixed revision and digest-checked | identity and digest |\n"
                "| 4. Load the sample or your own table | [Concept] | the diabetes table (or your file) loaded and checked | rows, target summary, digest |\n"
                "| 5. Validate and split | [Evaluation practice] | input manifest; a refusal probe; a random split | the manifest and split |\n"
                "| 6. Baselines, conditioning and evaluation | [Concept] | training mean, linear regression, TabDPT on the same holdout | the principal result |\n"
                "| 7. Evaluation report | [Evaluation practice] | the report with every baseline and an interpretation | what the numbers support |\n"
                "| 8. New-data inference and outputs | [Engineering] | eight rows scored; CSV and JSON outputs | the output contract |\n"
                "| 9. Export and fresh reload | [Engineering] | the serving artifact exported and reloaded in a new process | the reload check and the digest |\n"
                "| 10. Optional activity | [Concept] | change `n_ensembles` (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 6, 7 and 9 and the conclusion: they carry the principal "
                "results. The canonical path ends with Section 9; Section 10 changes nothing unless you switch it on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Tabular foundation model** | A model pretrained on many tables so that it can predict on a new table from examples, without training on it. |\n"
                "| **Support rows / context** | The labelled rows TabDPT reads at prediction time; here, the 353-row training split. |\n"
                "| **In-context conditioning** | Registering the support rows (and fitting preprocessing on them) so predictions can use them; the network's weights do not change. What `fit` does here. |\n"
                "| **`context_size`** | The most support rows TabDPT reads at once; with more support rows than this, a seeded random subsample is used. |\n"
                "| **`n_ensembles`** | How many passes, each with a different random view of the context, are averaged into one prediction. |\n"
                "| **Feature ceiling (`max_features`)** | The most features the checkpoint accepts; wider tables are reduced (PCA) first. |\n"
                "| **Feature encoder** | The repository's mapping of columns to numbers: numeric columns pass through, categorical values get fitted integer codes, with dedicated codes for missing and unseen values. |\n"
                "| **Random split** | Holdout rows drawn at random with a fixed seed; it assumes rows are independent. |\n"
                "| **Holdout** | Rows kept out of the support set and used only to measure the result. |\n"
                "| **Training-mean baseline** | Always predict the support rows' mean target; any useful model must beat it. |\n"
                "| **Linear-regression reference** | An ordinary least-squares fit on the standardised features: a one-line classical model. |\n"
                "| **MAE** | Mean absolute error: the average size of a miss, in target units. |\n"
                "| **RMSE** | Root mean squared error, in target units; large misses count more than in MAE. |\n"
                "| **R²** | 1 − (squared error ÷ squared spread of the holdout around its own mean). 0 means no better than predicting the holdout mean; 1 is perfect; it can be negative. |\n"
                "| **Point estimate** | One number per row with no interval around it; this pipeline produces no uncertainty interval. |\n"
                "| **Split-to-split range** | How much a metric moves when the same model is refitted on other seeded splits of the same table. |\n"
                "| **Input manifest** | A JSON record of the validated table: schema, target summary, missing values, verdict and findings. |\n"
                "| **Serving artifact** | `artifact.json` + `training_context.parquet`: everything needed to rebuild the conditioned pipeline elsewhere. |\n"
                "| **No-refit reload** | Rebuilding the pipeline from saved preprocessing statistics instead of fitting them again. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes; one changed byte changes it. |\n"
                "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and its SHA-256 digests. |\n"
                "| **Stage** | One step of the workflow, run as its own process by `run_stage`. |\n"
                "| **BYOD** | Bring Your Own Data: an optional switch to run the same workflow on your own table. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Load the public sample or your own table · [Concept]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. This cell runs the `data` "
                "stage. By default it loads scikit-learn's public diabetes table from the installed package (no download): ten "
                "baseline measurements (age, sex, body-mass index, blood pressure and six blood-serum values, each already centred "
                "and scaled by the dataset's authors) and a quantitative measure of disease progression one year later. With "
                "`USE_BYOD = True` it reads your file instead: set `BYOD_PATH` to a CSV or Parquet file already in the runtime, or, "
                "in Colab, leave `BYOD_PATH` empty to choose the file in an upload dialog (outside Colab an empty `BYOD_PATH` stops "
                "with a message naming it). `TARGET` names your numeric target column. If an identifier-like categorical column "
                "holds numeric-looking strings such as `01`, list it in `CATEGORICAL_COLUMNS` so it is read as text; every named "
                "column must exist.\n\n"
                "The reader refuses — naming the file and the rule — duplicate columns, a `TARGET` absent from the header (the "
                "message lists the header), a declared categorical column that is absent, and any **missing, blank, non-numeric or "
                "infinite target value**, with the number of such rows and examples (a value such as `1,234` is refused with a note "
                "that thousands separators are not accepted). No row is dropped silently. The stage first removes this notebook's "
                "earlier exports from `outputs/`, so a refused file never leaves an older result looking current. It prints the "
                "sample kind, the SHA-256 of the table's CSV serialisation and a summary of the target."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "TARGET = 'target'  # @param {{type:\"string\"}}\n"
                "CATEGORICAL_COLUMNS = []  # @param {{type:\"raw\"}}\n\n"
                "byod_file = ''\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH:\n"
                "        byod_file = BYOD_PATH\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: set BYOD_PATH to a CSV or Parquet file in this runtime.') from None\n"
                "        uploaded = files.upload()\n"
                "        if not uploaded:\n"
                "            raise RuntimeError('The upload was cancelled or empty: no file was received. Run this cell again and choose one CSV or Parquet file, or set BYOD_PATH.')\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one CSV or Parquet file; got {{sorted(uploaded)}}.')\n"
                "        upload_name, payload = next(iter(uploaded.items()))\n"
                "        byod_file = ROOT / 'inputs' / Path(upload_name).name\n"
                "        byod_file.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_file.write_bytes(payload)\n"
                "run_stage('data', use_byod=USE_BYOD, byod_path=str(byod_file), target=TARGET, categorical_columns=CATEGORICAL_COLUMNS)"
            ),
        },
        {
            "md": (
                "**What to notice:** `sample_kind: 'sample'`, 442 rows, 11 columns (10 features and `target`), the CSV digest, and "
                "the target summary: values from 25 to 346 with a mean near 152 and a standard deviation near 77. That spread is "
                "the scale every error below should be read against."
            ),
        },
        {
            "md": (
                "## 5. Validate the table → input manifest, then split · [Evaluation practice]\n\n"
                "The `validate` stage calls `validate_inputs`, the pipeline's public validation stage: it applies exactly the "
                "checks `fit` applies — unique column names, the target column present, numeric, finite and non-missing, at least "
                "`MIN_DISTINCT_TARGETS` distinct values, at least one feature column — and returns an **input manifest** naming the "
                "schema, the observed feature structure (numeric vs categorical columns, missing-value counts) and a target "
                "summary, written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, it also validates a "
                "probe whose target has a missing value and records the pipeline's own error message as a finding. Infinite "
                "numeric features are refused here, before conditioning. The table must have at least `MIN_ROWS` = 10 rows, and the "
                "holdout at least two distinct targets, so that R² can be computed; a smaller table stops here with a message "
                "naming the minimum, not later inside the model.\n\n"
                "Numeric columns are kept as they are (numeric missing values stay NaN through the repository encoder and are "
                "mean-imputed by TabDPT's support-fitted imputer); categorical values are mapped to fitted codes with dedicated "
                "missing and unknown codes. The random 80/20 split (seed 42) assumes rows are sufficiently independent; use "
                "preserved temporal, group or spatial boundaries for leakage-sensitive data. No model selection uses this holdout."
            ),
            "code": "run_stage('validate')",
        },
        {
            "md": (
                "**What to notice:** the ceilings first, then the manifest entry (10 numeric feature columns, no categorical columns, "
                "no missing values, the target summary), one finding — `missing-target-probe` rejected with *Regression target must "
                "be finite and non-missing* — and the split: 353 support rows and 89 holdout rows, with the support mean and the "
                "holdout standard deviation.\n\n"
                "**Checkpoint:** why does the stage refuse a 9-row table here, instead of letting the model run on it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A 20 % holdout of 9 rows is 2 rows, and of 4 rows only 1. R² divides by the holdout's spread around its own mean; "
                "with one row, or rows that share one target value, that spread is zero and R² is undefined. Letting such a table "
                "through means conditioning the model and then failing at evaluation with an error about R², which says nothing "
                "about the real cause. Refusing at validation, with the minimum named, tells you what to fix before any model runs. "
                "Even 10 rows only make R² *defined*, not *informative* — Section 6 shows how much it moves on 89.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines, in-context conditioning, capacity and evaluation · [Concept]\n\n"
                "The `condition` stage compares three regressors on the same 89 holdout rows:\n\n"
                "- `training_mean_baseline` — always predicts the support rows' mean target; its R² is at most about 0;\n"
                "- a **standardised linear regression** fitted on the 353 support rows (the repository's encoder, mean imputation, "
                "standard scaling, ordinary least squares) — a one-line classical reference; the stage also refits it on 20 seeded "
                "80/20 splits to measure how much its R² moves from split to split;\n"
                "- **TabDPT**, conditioned in context on the support rows — `fit()` fits preprocessing and registers the rows; it is "
                "**not gradient training**. The model is loaded from the digest-verified checkpoint of Section 3.\n\n"
                "After conditioning the stage prints the encoded feature count, the model feature ceiling, whether the upstream "
                "feature-reduction path is active, and the missing-value preprocessing in effect. `evaluate` then scores the holdout: "
                "MAE is the average absolute error in target units; RMSE also uses target units but weights large misses more; R² "
                "compares the squared error with that of predicting the holdout's own mean. Seeds control the supported stochastic "
                "paths (split, context subsampling, ensemble permutations); bitwise determinism across hardware kernels is not "
                "promised.\n\n"
                "`N_ENSEMBLES` (1–16) and `CONTEXT_SIZE` (16–4096) are form fields; keep the defaults (2 and 512) for the first "
                "run — the later sections reuse whatever this cell used.\n\n"
                "**Predict before running:** the training-mean baseline will have an RMSE close to the holdout's standard "
                "deviation (about 73) and an R² near 0. Write down your guess for the linear regression's R² and for TabDPT's, and "
                "whether you expect TabDPT to beat the linear model by more than the split-to-split noise."
            ),
            "code": (
                "N_ENSEMBLES = 2  # @param {{type:\"integer\"}}\n"
                "CONTEXT_SIZE = 512  # @param {{type:\"integer\"}}\n"
                "run_stage('condition', n_ensembles=N_ENSEMBLES, context_size=CONTEXT_SIZE)"
            ),
        },
        {
            "md": (
                "**What to notice:** `effective support rows <= 353` (all support rows fit within `context_size` 512, so no "
                "subsampling); the capacity record — 10 encoded features, the checkpoint's feature ceiling, `featureReductionActive: "
                "False`; then three result lines. On this split the training-mean baseline scores MAE 64.01, RMSE 73.22, R² −0.012, "
                "and the linear-regression reference MAE 42.79, RMSE 53.85, R² 0.453, with an R² range of 0.332–0.585 over the 20 "
                "seeded splits. TabDPT's line is the one your run printed; no hosted TabDPT number has been recorded for this "
                "notebook yet.\n\n"
                "**Checkpoint:** suppose TabDPT reaches R² 0.48 and the linear regression 0.453 on this split. What can you "
                "conclude?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Very little about which model is better: 0.03 of R² is far smaller than the 0.25 the linear regression alone moves "
                "between seeded splits of the same 442 rows, so one split cannot rank the two. What the run does show is that "
                "TabDPT, with no training, explains a substantial share of the variance (far better than the mean baseline) and is "
                "in the same range as a strong classical model on this small, noisy table. Read MAE against the target's spread: "
                "an MAE near 43 on a target whose standard deviation is about 73 is a useful but rough predictor, not a precise "
                "one. A ranking would need repeated splits or a much larger independent test set.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report · [Evaluation practice]\n\n"
                "The `report` stage calls `evaluation_report`, the pipeline's public evaluation stage, which always produces a "
                "report. Here it carries the metrics `evaluate` returned (`mae`, `rmse`, `r2` — the repository's own metric ids, "
                "with units), both baselines (the training mean and the linear-regression reference with its split-to-split R² range) "
                "and the verdict `sample-sanity`: a single seeded random holdout with no dispersion estimate for the model, tutorial "
                "evidence rather than a benchmark. Its `interpretation` field states the holdout target's spread and the reference's "
                "range. When no labelled holdout exists the verdict is `not-measurable` and the report states what would make the "
                "task measurable. The report is written to `outputs/{stem}_evaluation_report.json`. Expect TabDPT to beat the "
                "training-mean baseline clearly and to land in the range of the linear reference on the public sample; a BYOD table "
                "can behave very differently."
            ),
            "code": "run_stage('report')",
        },
        {
            "md": (
                "**What to notice:** `verdict: sample-sanity`, `n_holdout: 89`, the three metric entries with their units, two "
                "entries under `baselines`, and the `interpretation` sentence — computed from this run's numbers, so it stays true "
                "for a BYOD table."
            ),
        },
        {
            "md": (
                "## 8. New-data inference and machine-readable outputs · [Engineering]\n\n"
                "The `predict` stage conditions the model again from the saved split and scores eight rows with the target removed. "
                "They are the first eight holdout rows: TabDPT has never seen them as support rows, but Section 6 already scored "
                "them, so this section demonstrates the inference contract, not a new evaluation — and because their true targets "
                "are known, the printed table shows them beside each prediction. Training-fitted repository encoding and upstream "
                "imputation/scaling are reused; no preprocessing is fitted on the new rows. `prediction` is a **continuous point "
                "estimate** in target units, with no uncertainty interval. `row_id` is the row's index in the original table; it "
                "stays outside the model's feature schema and maps predictions back to inputs.\n\n"
                "Outputs: `outputs/{stem}_predictions.csv` (`row_id`, `prediction`); `outputs/{stem}_new_rows.csv` (the same eight "
                "rows, unlabelled, with `row_id` — the input the companion artifact-inference notebook accepts); and "
                "`outputs/{stem}_result.json`, which records the metrics, both baselines, the evaluation report, the input "
                "manifest, the sample identity and digest, the split, the capacity report, the inference controls, the notebook's "
                "source (repository, revision, carried-file digests, generator), the model identifier, the immutable model "
                "revision and licence, and the runtime identity (`python`, `torch`, `tabdpt`, `numpy`, `pandas`, `sklearn`, device, "
                "`use_flash`) as a provenance record. No credentials are recorded."
            ),
            "code": "run_stage('predict')",
        },
        {
            "md": (
                "**What to notice:** eight rows with `row_id` 287, 211, 72, 321, 73, 418, 367 and 354, each with its prediction, "
                "the true `holdout_target` and the absolute error in target units (the printed table only; the CSV holds no "
                "targets). Then the list of files in `outputs/`."
            ),
        },
        {
            "md": (
                "## 9. Export the serving artifact and verify a fresh reload · [Engineering]\n\n"
                "For this in-context model the deployable serving state is not the checkpoint alone: it includes the labelled "
                "support data, the fitted repository/upstream preprocessing state and the exact pinned base-model contract. The "
                "`export` stage writes, with `export_artifact_bundle`, `outputs/artifact/artifact.json` (with the support table's "
                "size and SHA-256 and the base-model identity) and `training_context.parquet`; the latter inherits the source "
                "data's confidentiality, licensing, retention and disclosure obligations. It prints the artifact's **trusted "
                "digest** — the SHA-256 of `artifact.json`, which pins the Parquet file — to hand to the companion notebook "
                "(`EXPECTED_ARTIFACT_SHA256`), and, on the default path, whether the export is byte-identical to the companion's "
                "pinned sample artifact.\n\n"
                "The `reload` stage then runs in a **fresh process**: it copies the artifact to `outputs/artifact-reload/`, validates "
                "it before reconstruction, and loads it through the verified **no-preprocessing-refit** path with the base "
                "checkpoint taken from the digest-verified snapshot of Section 3 (no network fallback). Predictions must match the "
                "exporting process's within `rtol=1e-5`, `atol=1e-6`; bitwise identity is not required. The companion notebook "
                "`tabdpt_regressor_artifact_inference_colab.ipynb` consumes this artifact from a separate execution.\n\n"
                "**Predict before running:** will the reloaded model's predictions be bitwise identical to the exporting "
                "process's? Why might they not be, even though nothing is refitted?"
            ),
            "code": "run_stage('export')\nrun_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** the artifact's two files, its `artifact_sha256`, the `training_context_sha256`, and "
                "`matches_pinned_sample_artifact: True` on the default path; then the largest absolute prediction difference (zero "
                "or a tiny floating-point difference) and the `PASS` line.\n\n"
                "**Checkpoint:** what does a matching `artifact_sha256` prove to the person you send the artifact to, and what "
                "does it not prove?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A matching digest proves that the files they received are byte for byte the files this run exported — nothing "
                "was altered or swapped on the way, including the support targets, because `artifact.json` pins the Parquet file's "
                "size and SHA-256. It does not prove who produced them, that the support data were right or lawful to share, or "
                "that the model is any good: it fixes identity, not quality. That is why the digest has to reach them through a "
                "channel they trust separately from the files. As for the prediction: the outputs need not be bitwise identical "
                "because GPU kernels can sum in a different order in a new process; the stage therefore checks equality within a "
                "stated tolerance.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 10. Optional activity: how many ensemble members does this table need? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** Each ensemble member is one pass with a different random view of "
                "the context; `n_ensembles` averages them. **Predict:** with `ACTIVITY_N_ENSEMBLES = 8` instead of the canonical 2, "
                "will RMSE go down, up, or stay within noise? **Change:** tick `RUN_ACTIVITY` and set `ACTIVITY_N_ENSEMBLES` (1–16). "
                "**Run** this cell. **Observe** the canonical and activity metrics side by side. **Explain** the difference in your "
                "own words.\n\n"
                "The activity conditions the model again on the same support rows, evaluates the same holdout, and writes only to "
                "`outputs/activity/`; it checks the SHA-256 of every canonical output before and after and stops if any changed. "
                "It is off by default and is not part of the canonical path."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_N_ENSEMBLES = 8  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', n_ensembles=ACTIVITY_N_ENSEMBLES)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the `changed` line, then the canonical and activity MAE, RMSE and R² side by "
                "side. No hosted run of the activity has been recorded, so compare with your own prediction rather than with a "
                "stated number. `canonical_outputs_unchanged: True` confirms the activity left Sections 4–9's files alone.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "More members average out the randomness of any single pass, so predictions become steadier; on a noisy 89-row "
                "holdout the resulting change in RMSE or R² is usually smaller than the split-to-split variation measured in "
                "Section 6, so it is not evidence either way. The cost is time: inference grows roughly linearly with "
                "`n_ensembles`.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel. |\n"
        "| `No CUDA GPU detected` | a CPU runtime | It still runs, more slowly. For the documented runtime choose *Runtime → Change runtime type → T4 GPU* before Run all. |\n"
        "| `Not enough free disk` | the isolated environment needs about 7 GB | Start a fresh runtime; an environment built from the same lock earlier in this runtime is reused. |\n"
        "| `Carried file integrity failure` | a carried file was edited in the notebook | Do not edit the infrastructure cells; open a fresh copy from the repository. |\n"
        "| `uv … wheel size/hash mismatch`, a `URLError`, or `CalledProcessError` from `uv` | a network failure or a transient PyPI error | Re-run the Section 2 install cell; `uv` reuses what it already downloaded. Never remove `--require-hashes`, a pin or a hash. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 was run with `NEW_RUN_DIRECTORY` ticked, or the temporary directory was cleared | Run Sections 1, 2 and 3 again in order, or choose *Run all*. |\n"
        "| `RuntimeError: Stage '…' failed (exit 1): …` | the stage raised an error; the text after the colon is its own message, with the traceback above | Find the message below; fix the cause and re-run that cell and the cells after it. |\n"
        "| `… is missing: run the stage that writes it before …` | a learner cell was run before an earlier one | Run the notebook from Section 4 in order. |\n"
        "| A Hub download error in Section 3 | a transient Hugging Face failure | Re-run the Section 3 cell; nothing is used until the digest matches. |\n"
        "| `tabdpt1_2.safetensors: size … != manifest` or `sha256 … != manifest` | a corrupted or substituted download | Delete `weights/tabdpt-1.2/tabdpt1_2.safetensors` and re-run Section 3. Never edit the manifest. |\n"
        "| `CUDA out of memory` in Section 6 or later | a large BYOD table or a large `CONTEXT_SIZE` | Lower `CONTEXT_SIZE` or `N_ENSEMBLES`, or restart on a fresh T4. |\n"
        "| `the target column 'target' is not in the header [...]` | your target column has another name | Set `TARGET` to its name. |\n"
        "| `the target column … has N missing or blank value(s)` | some rows have no target | Remove or fill those rows (the message gives their file lines), then re-run Section 4. |\n"
        "| `the target column … has N non-numeric value(s)` | text, units or thousands separators in the target | Write plain numbers (`1234.5`, not `1,234.5` or `12 kg`). |\n"
        "| `… rows is below the minimum of 10` or `fewer than 2 distinct target values` | a table too small to evaluate | Supply more rows; R² needs a holdout with varied targets. |\n"
        "| `numeric feature … contains infinite values` | `inf` in a numeric column | Replace it with a finite value or leave the cell empty. |\n"
        "| `USE_BYOD is True but BYOD_PATH is empty …` outside Colab, or `The upload was cancelled or empty` | no file was supplied | Set `BYOD_PATH`, or run the cell again and choose a file. |\n"
        "| `matches_pinned_sample_artifact: False` in Section 9 | the export differs from the companion's pinned sample (for example a different library build) | Not an error for this notebook; record both digests and report it, because the companion's default path assumes they match. |\n\n"
        "## Interpretation and limits\n\n"
        "Predictions are continuous point estimates in the target's units with no uncertainty interval; any tolerance or "
        "decision band must be chosen on the caller's own labelled, domain-representative data. The evaluation report's "
        "`sample-sanity` verdict names what it is: one seeded random holdout of the public sample with no dispersion "
        "estimate for the model — tutorial evidence that must not be generalised to a domain, a target range or a "
        "data-collection process. On this small, noisy table a standardised linear regression explains about 45 % of the "
        "holdout variance, and its R² moves from 0.33 to 0.59 across seeded splits, so differences of a few hundredths of R² "
        "between models on one split do not rank them. Rows that are not independent (temporal, grouped, spatial), targets "
        "outside the support range, wide tables that trigger upstream feature reduction, and support sets larger than "
        "`context_size` (subsampling) all change results in ways the metrics above do not measure.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate the demonstrated table, condition on regression "
        "support data, surface missing-value and capacity behaviour, compute sample metrics against a training-mean "
        "baseline and a classical reference, score new rows, emit the shown machine-readable outputs and the DIMER serving "
        "artifact, and reconstruct equivalent predictions in a fresh process from serialised fitted preprocessing without "
        "refitting it — without the repository being reachable. It does **not** establish benchmark superiority, domain "
        "generalisation, fairness, robustness, calibration, production safety, or deployment fitness.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three to five sentences, in your own words, using the numbers your run printed:\n\n"
        "1. **Result:** TabDPT's holdout MAE, RMSE and R², beside the training-mean baseline's and the linear reference's.\n"
        "2. **Reading:** whether the gap to the reference is larger than the reference's split-to-split R² range (Section 6), "
        "and what the MAE means against the target's spread.\n"
        "3. **Reuse:** what the Section 9 reload proved, and what the artifact digest does and does not prove.\n"
        "4. **Limit:** the one limitation you would fix first before using this on your own data.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "On the 89-row holdout of the diabetes table, TabDPT conditioned in context on 353 support rows scored the MAE, RMSE "
        "and R² printed in Section 6, against MAE 64.01 / R² −0.012 for the training mean and MAE 42.79 / R² 0.453 for a "
        "standardised linear regression. The linear regression alone ranged 0.332–0.585 in R² over 20 seeded splits, so a gap "
        "of a few hundredths between the two models is within split-to-split variation: the run shows that TabDPT predicts "
        "useful signal without training, not that it beats a linear model here, and an MAE in the 40s on a target with a "
        "standard deviation near 77 is a rough estimate. The exported artifact reloaded in a fresh process with equivalent "
        "predictions, and its digest lets a recipient check they got exactly these files, though not who made them. Before "
        "using it on my own data I would evaluate on repeated or grouped splits of a larger table and attach an uncertainty "
        "estimate before relying on any single prediction.\n\n"
        "</details>\n\n"
        "**Next experiments:** enable `USE_BYOD` with a labelled CSV from your own domain and compare `evaluate` against "
        "both baselines on a split that respects your data's structure; run the Section 10 activity with `n_ensembles` 1 "
        "and 16; supply a table with more features than the model ceiling to see the upstream feature-reduction path "
        "activate; feed `outputs/artifact/`, `outputs/{stem}_new_rows.csv` and the printed digest to the companion "
        "artifact-inference notebook in a separate session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/weights/README.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream inference code: https://github.com/layer6ai-labs/TabDPT-inference\n"
        "- TabDPT paper: https://arxiv.org/abs/2608.01400\n"
        "- Diabetes data: B. Efron, T. Hastie, I. Johnstone and R. Tibshirani, Least Angle Regression, Annals of Statistics 32(2), 2004, via `sklearn.datasets.load_diabetes`"
    ),
}
