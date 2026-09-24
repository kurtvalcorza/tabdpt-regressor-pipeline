"""Static release-asset validation for the TabDPT regressor DIMER pipeline.

Checks the two STANDALONE tutorial notebooks (DIMER Notebook Specification 2.0 §4) — the `E2E`
tutorial and its `ARTIFACT-INFERENCE` companion — the tutorial registry, model card, README, STATUS.md
and weight documentation for source conformance and cross-document identity consistency, and runs the
generator parity checks (PAR1–PAR3) for every notebook.

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.

Two-notebook variant of the fleet validator (snapshot resnet50 @ 6c77f84, tooling updates 2026-09-13 15:40 + 17:10):
the constants block declares NOTEBOOKS (one entry per generated notebook: template, profile, gates, outputs,
markers), PACKAGE_DIR, IDENTITY_DOCS and EXPECTED_CARD_SPEC; the shared block is the snapshot's (per-module
PAR1 through build.load_context, joined-module digest, own-repo and mutable-git-dependency rules) with the
per-notebook spec threaded through validate_notebooks().
"""
# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "tabdpt_regressor_pipeline"
PACKAGE_DIR = "src/tabdpt_regressor_pipeline"  # the template's package_dir (default src/<package>)
REPO_NAME = "tabdpt-regressor-pipeline"
EXPECTED_MODEL_ID = "Layer6/TabDPT"
PIPELINE_CLASS = "TabDPTRegressionPipeline"
# INF1: the exact load expression the model cell must use (the template's `model_load`).
MODEL_LOAD_EXPR = f"{PIPELINE_CLASS}.from_pretrained(weights_dir=WEIGHTS_DIR, compile_model=False, use_flash=False, seed=42)"
# Additional 40-hex revisions a document may legitimately cite: the upstream TabDPT-inference code commit.
KNOWN_SHAS: frozenset[str] = frozenset(("9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246",))
# Documents that must name the model id and the immutable revision (this repository keeps weight
# provenance in weights/README.md rather than docs/WEIGHTS.md).
IDENTITY_DOCS = ("README.md", "MODEL_CARD.md", "weights/README.md")
# MODEL_CARD_SPEC version the card on this branch declares.
EXPECTED_CARD_SPEC = "1.1"
# Direct-library use that must stay inside the carried module cells (G2: the notebook calls the
# pipeline API, it does not reimplement it). Checked on every code cell except the embedded ones.
FORBIDDEN_OUTSIDE_MODULE = (
    "from huggingface_hub import",
    "import huggingface_hub",
    "hf_hub_download(",
    "from tabdpt import",
    "import tabdpt",
    "TabDPTRegressor(",
    "resolve_tabdpt_weights(",
    "from sklearn.metrics import",
    "mean_absolute_error(",
    "mean_squared_error(",
    "r2_score(",
    "from safetensors",
    "torch.load(",
)
# One entry per generated notebook: its template module (tools/<template>.py), profile, Colab form
# gates that must default to the non-interactive path, the machine-readable artifacts it must write
# (OUT1-OUT3, DAT24, EVAL21), and the profile-specific code / learner-facing markers.
NOTEBOOKS = {
    "tabdpt_regressor_colab.ipynb": {
        "template": "notebook_template",
        "profile": "E2E",
        "byod_gates": ("USE_BYOD",),
        "expected_outputs": (
            "outputs/tabdpt_regressor_input_manifest.json",
            "outputs/tabdpt_regressor_evaluation_report.json",
            "outputs/tabdpt_regressor_result.json",
            "outputs/tabdpt_regressor_predictions.csv",
        ),
        "code_markers": (
            "input_manifest = validate_inputs(frame, target_column=TARGET, names=[data_name])",
            "validate_inputs(probe, target_column=TARGET)",
            "print({'ceilings': {'MIN_DISTINCT_TARGETS': MIN_DISTINCT_TARGETS}",
            "baseline = training_mean_baseline(train[TARGET], test[TARGET])",
            "pipe.fit(train, target_column=TARGET, seed=SEED)",
            "metrics = pipe.evaluate(test, **kw)",
            "report = evaluation_report(metrics, baseline=baseline, n_holdout=len(test), target_column=TARGET, sample_kind=sample_kind",
            "pred = pipe.predict(new_rows, **kw)",
            "train, test = train_test_split(frame, test_size=0.2, random_state=SEED)",
            "'modelFeatureCeiling': model_feature_ceiling",
            "'featureReductionActive': encoded_feature_count > model_feature_ceiling",
            "'numericMissingPolicy': 'support/training-fitted mean imputation'",
            "'categoricalMissingPolicy': 'dedicated fitted missing code'",
            "'unknownCategoryPolicy': 'dedicated fitted unknown code'",
            "'effective support rows <='",
            "manifest_path = export_artifact_bundle(pipe, train, ART)",
            "reloaded = load_verified_artifact(RELOAD / 'artifact.json', model_weight_path=pipe.model_weight_path, compile_model=False, use_flash=False, seed=SEED)",
            "assert reloaded.preprocessing_restored_ is True",
            "np.testing.assert_allclose(pred.to_numpy(), pred2.to_numpy(), rtol=1e-5, atol=1e-6)",
            "'model_revision': MODEL_REVISION",
            "'model_license': MODEL_LICENSE",
            "importlib.metadata.version('tabdpt')",
            "'use_flash': False",
            "from sklearn.datasets import load_diabetes",
        ),
        "markdown_markers": (
            "**Capability:** supervised tabular regression by in-context conditioning",
            "it does **not** gradient-train or fine-tune the pretrained weights",
            "**continuous point estimates only**",
            "`training_mean_baseline` is the trivial baseline",
            "**not gradient training**",
            "MAE is the average absolute error in target units",
            "the verdict is `not-measurable`",
            "`sample-sanity`",
            "**no-preprocessing-refit**",
            "classification, gradient fine-tuning, or calibrated per-prediction uncertainty intervals",
        ),
    },
    "tabdpt_regressor_artifact_inference_colab.ipynb": {
        "template": "notebook_template_artifact_inference",
        "profile": "ARTIFACT-INFERENCE",
        "byod_gates": (),
        "expected_outputs": (
            "outputs/tabdpt_regressor_artifact_inference_input_manifest.json",
            "outputs/tabdpt_regressor_artifact_inference_evaluation_report.json",
            "outputs/tabdpt_regressor_artifact_inference_result.json",
            "outputs/tabdpt_regressor_artifact_inference_predictions.csv",
        ),
        "code_markers": (
            "ARTIFACT_DIR = ''",
            "NEW_DATA_PATH = ''",
            "manifest, context_path = validate_artifact_bundle(manifest_path)",
            "serving = load_verified_artifact(manifest_path, model_weight_path=pipe.model_weight_path, compile_model=False, use_flash=False)",
            "if getattr(serving, 'preprocessing_restored_', False) is not True:",
            "if serving.target_column != target_column:",
            "input_manifest = validate_inputs(new_data, None, feature_columns=expected, names=[input_name])",
            "validate_inputs(new_data.drop(columns=[expected[0]]), None, feature_columns=expected)",
            "print({'ceilings': {'MIN_DISTINCT_TARGETS': MIN_DISTINCT_TARGETS, 'modelFeatureCeiling': model_feature_ceiling}})",
            "pred = serving.predict(new_data, **kw)",
            "report = evaluation_report(None, n_holdout=0, target_column=serving.target_column, sample_kind='BYOD')",
            "'modelFeatureCeiling': model_feature_ceiling",
            "'featureReductionActive': encoded_feature_count > model_feature_ceiling",
            "'numericMissingPolicy': 'restored training-fitted mean imputation'",
            "'categoricalMissingPolicy': 'restored fitted missing code'",
            "'unknownCategoryPolicy': 'restored fitted unknown code'",
            "'effective support rows <='",
            "'model_revision': MODEL_REVISION",
            "'model_license': MODEL_LICENSE",
            "'use_flash': False",
        ),
        "markdown_markers": (
            "**Capability:** serving-state reconstruction from an externally produced DIMER artifact",
            "produced **outside this execution**",
            "**no artifact is created here**",
            "**Trust boundary.**",
            "legacy compatibility",
            "**no network fallback**",
            "**continuous point estimates only**",
            "its verdict is `not-measurable`",
            "artifact creation, in-notebook support fitting, classification, gradient fine-tuning",
        ),
        "forbidden_code": (
            "load_diabetes(",
            "export_artifact_bundle(",
            "pipe.fit(",
            "serving.fit(",
        ),
    },
}

# ---------------------------------------------------------------------------
# Shared checks. Everything below is source/structure validation only. Passing
# these checks is NOT clean-runtime execution evidence under DIMER Notebook
# Specification 2.0; see docs/release-verification.md for the release gate.
# ---------------------------------------------------------------------------

NOTEBOOK_SPEC = "2.0"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
# Markers every standalone DIMER tutorial in this fleet must carry, independent of profile.
# Matched on comment-stripped code, so a commented-out call does not count.
COMMON_CODE_MARKERS = (
    "PINS = [",
    "NOTEBOOK_SOURCE = {",
    "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINS], check=True)",
    "importlib.metadata.packages_distributions()",
    "importlib.invalidate_caches()",
    "platform.python_version()",
    "torch.__version__",
    "MANIFEST = {",
    "if (MANIFEST['modelId'], MANIFEST['revision']) != (MODEL_ID, MODEL_REVISION):",
    "WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR",
    "json.dump(MANIFEST, handle, indent=2)",
    "fetched = stage_missing_files(WEIGHTS_DIR, allow_download=True)",
    "snapshot = verify_snapshot(WEIGHTS_DIR)",
    "'repository_revision': NOTEBOOK_SOURCE['repository_revision']",
    "'notebook_source': NOTEBOOK_SOURCE",
    "os.makedirs('outputs', exist_ok=True)",
    "from google.colab import files",
    "files.upload()",
)
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "- **External access:** the Hugging Face Hub only",
    "## 1. Install the pinned runtime",
    "## 2. Pipeline code (carried verbatim from",
    "## 3. Pin, stage and verify the model",
    "## Interpretation and limits",
    "Successful execution proves that the recorded repository revision",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
# Patterns that must never appear in tutorial code (comment-stripped), in any cell.
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com/kurtvalcorza")),
    ("mutable git dependency (MOD14)", re.compile(r"git\+https?://(?![^\n]*@[0-9a-f]{40}\b)")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("repository package import (ST1)", re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(r"\bpickle\.load|\btorch\.load\s*\(|getattr\(\s*torch\s*,\s*['\"]load['\"]"),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
)
# Worker/clone paths that must never appear outside the generator-owned cells (Kurt 2026-09-13:
# inference is in-notebook; no worker process, no clone of the repository).
FORBIDDEN_OUTSIDE_MODULE_FLEET = (
    "worker.run(",
    "worker_cli(",
    "subprocess.run([",
)


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    """Return the source without comment tokens (string contents are preserved)."""
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _entry_module(template: dict) -> str:
    return template.get("entry_module", "pipeline.py")


def _modules(template: dict) -> list[str]:
    return list(template.get("modules", ["pipeline.py"]))


def _package_identity(template: dict) -> tuple[str, str]:
    """Read MODEL_ID / MODEL_REVISION from the entry module source without importing torch."""
    text = _read(ROOT / PACKAGE_DIR / _entry_module(template))
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        f"{_entry_module(template)} must define MODEL_ID and MODEL_REVISION",
    )
    _check(SHA40.match(revision.group(1)) is not None, "MODEL_REVISION must be a 40-hex immutable commit")
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def _primary_template() -> dict:
    return _load_tool(next(iter(NOTEBOOKS.values()))["template"]).TEMPLATE


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check(f'model_card_spec: "{EXPECTED_CARD_SPEC}"' in front, f"MODEL_CARD.md model_card_spec must be {EXPECTED_CARD_SPEC}")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def validate_identity_consistency() -> None:
    """The immutable upstream identity must be the same string in every document."""
    model_id, revision = _package_identity(_primary_template())
    for name in IDENTITY_DOCS:
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the immutable revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


# --- weight-facts check (fleet rollout 2026-09-24) ---
# Every SHA-256 digest and byte count quoted in the weight prose must come from a committed
# weights/*/dimer-base-manifest.json, or be declared below with a label saying what it describes
# (dataset files, upstream files that are not staged, origin checkpoints, totals). Declared entries
# that no document cites any more are rejected, so the allowlist cannot go stale.
WEIGHT_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
EXTERNAL_WEIGHT_BYTES: dict[int, str] = {}
EXTERNAL_WEIGHT_DIGESTS: dict[str, str] = {}
_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_GROUPED = r"(\d{1,3}(?:[,\u202f\u00a0 ]\d{3})+|\d+)"
_BYTE_COUNT = re.compile(r"(?<![\d,\-])" + _GROUPED + r"\s*bytes\b|totalBytes`?\s*" + _GROUPED)


def _manifest_facts(root: Path = ROOT) -> tuple[set[str], set[int]]:
    digests: set[str] = set()
    sizes: set[int] = set()
    for path in sorted(root.glob("weights/*/dimer-base-manifest.json")):
        manifest = json.loads(_read(path))
        sizes.add(manifest["totalBytes"])
        for entry in manifest["files"]:
            digests.add(entry["sha256"])
            sizes.add(entry["bytes"])
    return digests, sizes


def validate_weight_facts(root: Path = ROOT) -> None:
    """Every SHA-256 and byte count quoted in the weight prose must come from a manifest or a labelled allowlist entry."""
    digests, sizes = _manifest_facts(root)
    _check(bool(digests), "no weights/*/dimer-base-manifest.json found to check weight facts against")
    cited_digests: set[str] = set()
    cited_sizes: set[int] = set()
    for name in WEIGHT_DOCS:
        path = root / name
        if not path.exists():
            continue
        text = _read(path)
        found_digests = set(_DIGEST.findall(text))
        found_sizes = {int(re.sub(r"[,\u202f\u00a0 ]", "", m.group(1) or m.group(2))) for m in _BYTE_COUNT.finditer(text)}
        cited_digests |= found_digests
        cited_sizes |= found_sizes
        bad_digests = sorted(found_digests - digests - set(EXTERNAL_WEIGHT_DIGESTS))
        _check(not bad_digests, f"{name} cites SHA-256 digests absent from every manifest and from EXTERNAL_WEIGHT_DIGESTS: {bad_digests}")
        bad_sizes = sorted(found_sizes - sizes - set(EXTERNAL_WEIGHT_BYTES))
        _check(not bad_sizes, f"{name} cites byte counts absent from every manifest and from EXTERNAL_WEIGHT_BYTES: {bad_sizes}")
    stale = sorted(set(EXTERNAL_WEIGHT_BYTES) - cited_sizes) + sorted(set(EXTERNAL_WEIGHT_DIGESTS) - cited_digests)
    _check(not stale, f"EXTERNAL_WEIGHT_* entries no weight document cites any more: {stale}")


# --- end weight-facts check ---

def validate_release_status() -> None:
    """STATUS.md, README.md and tutorials/README.md must agree on one status token."""
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )


def _validate_notebook_structure(
    path: Path, notebook: dict, spec: dict, template: dict, build
) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == spec["profile"], f"{path.name}: profile {profile!r} != declared {spec['profile']!r}")
    version = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(version == NOTEBOOK_SPEC, f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'")
    _check(dimer.get("notebook_mode") in ("REFERENCE", "GUIDED", "WORKSHOP"), f"{path.name}: metadata.dimer.notebook_mode must declare a §3.3 pedagogical mode")
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    generated = dimer.get("generated_from")
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(generated.get("repository") == REPO_NAME, f"{path.name}: generated_from.repository must be {REPO_NAME}")
    entry_rel = f"{PACKAGE_DIR}/{_entry_module(template)}"
    _check(generated.get("module") == entry_rel, f"{path.name}: generated_from.module must be {entry_rel}")
    order = build._module_order(ROOT / PACKAGE_DIR, _modules(template))
    module_rels = [f"{PACKAGE_DIR}/{m}" for m in order]
    _check(generated.get("modules") == module_rels, f"{path.name}: generated_from.modules must be {module_rels}")
    module_sha = hashlib.sha256("".join(_read(ROOT / PACKAGE_DIR / m) for m in order).encode("utf-8")).hexdigest()
    _check(
        generated.get("module_sha256") == module_sha,
        f"{path.name}: generated_from.module_sha256 does not match {PACKAGE_DIR}/ (PAR4: regenerate the notebook)",
    )
    _check(bool(generated.get("generator")), f"{path.name}: generated_from.generator is required")
    cells = notebook.get("cells", [])
    _check(
        bool(cells) and cells[0].get("cell_type") == "markdown",
        f"{path.name}: first cell must be markdown",
    )
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(
            index > 0 and cells[index - 1].get("cell_type") == "markdown",
            f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell",
        )
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    raw_code = "\n".join(source for _, source, _ in code_cells)
    _check(not PLACEHOLDER.search(raw_code + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _validate_gates(path: Path, code_cells: list[tuple[int, str, ast.Module]], gates: tuple[str, ...]) -> None:
    """Each BYOD gate is assigned exactly once, to the constant False, on a Colab form line."""
    for gate in gates:
        assignments = []
        for index, source, tree in code_cells:
            lines = source.splitlines()
            for node in ast.walk(tree):
                if gate in _assignment_targets(node):
                    line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                    assignments.append((index, node, line))
        _check(
            len(assignments) == 1,
            f"{path.name}: {gate} must be assigned exactly once, found {len(assignments)}",
        )
        index, node, line = assignments[0]
        is_false = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        )
        _check(is_false, f"{path.name}: {gate} must be assigned the constant False (cell {index})")
        _check("# @param" in line, f"{path.name}: {gate} must be a Colab form parameter (`# @param`)")
    for index, _source, tree in code_cells:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                _check(
                    not any(alias.name.startswith("google.colab") for alias in node.names),
                    f"{path.name}: google.colab must only be imported inside the BYOD gate (cell {index})",
                )


def _validate_embedded_modules(path: Path, notebook: dict, build, template: dict) -> set[int]:
    """PAR1: one tagged cell per carried module, in dependency order, each equal to its module after
    the documented rewrites (generator /2 multi-module carrier; ST2 applied per module)."""
    tagged = [
        (index, cell)
        for index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code" and cell.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    context = build.load_context(ROOT, template, recorded)
    expected_rels = context["module_rels"]
    _check(
        [cell["metadata"]["dimer"]["embedded_module"] for _, cell in tagged] == expected_rels,
        f"{path.name}: the cells tagged metadata.dimer.embedded_module must be exactly {expected_rels}, in order (ST2)",
    )
    for (index, cell), module in zip(tagged, context["modules"], strict=True):
        rel = f"{context['pkg_rel']}/{module}"
        _check(
            cell["metadata"]["dimer"].get("module_sha256") == context["per_module_sha256"][rel],
            f"{path.name}: cell {index} module_sha256 tag does not match {rel}",
        )
        _check(
            _cell_source(cell).rstrip("\n") + "\n" == context["embedded"][module],
            f"{path.name}: embedded module cell {index} differs from {rel} (PAR1); regenerate the notebook",
        )
    return {index for index, _ in tagged}


def _validate_identity(
    path: Path, code_cells: list[tuple[int, str, ast.Module]], embedded: set[int], revision: str
) -> None:
    """Identity constants are bound in the carried module cells only; nothing outside rebinds them."""
    for index, _source, tree in code_cells:
        if index in embedded:
            continue
        for node in ast.walk(tree):
            rebound = [name for name in _assignment_targets(node) if name in IDENTITY_NAMES]
            _check(not rebound, f"{path.name}: {rebound} must not be rebound outside the module cells (cell {index})")
    outside = "\n".join(source for index, source, _ in code_cells if index not in embedded)
    manifest_block = re.search(r"^MANIFEST = (\{.*?^\})$", outside, re.M | re.S)
    _check(manifest_block is not None, f"{path.name}: model cell must carry an inline MANIFEST literal (ST3)")
    outside_without_manifest = outside.replace(manifest_block.group(0), "")
    _check(
        revision not in outside_without_manifest,
        f"{path.name}: the model revision may appear only in the carried modules and the inline manifest",
    )


def _validate_parity(
    path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], build, template: dict
) -> None:
    """PAR2/PAR3: inline manifest and pins equal the repository's; the generator reproduces the file."""
    code = "\n".join(source for _, source, _ in code_cells)
    manifest = json.loads(_read(ROOT / "weights" / template["weights_key"] / "dimer-base-manifest.json"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    _check(inline is not None and json.loads(inline.group(1)) == manifest, f"{path.name}: inline MANIFEST != committed manifest (PAR2)")
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    _check(pins_block is not None, f"{path.name}: install cell must carry PINS = [...] (ENV2)")
    _check(re.findall(r"'([^']+)'", pins_block.group(1)) == build._pins(ROOT, template), f"{path.name}: inline PINS != declared runtime pins (PAR2)")
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    _check(current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate")


def _validate_bootstrap_guard(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    """The stale-import guard must actually raise: `if stale:` whose body raises RuntimeError."""
    raises = False
    for _, _, tree in code_cells:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "stale":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call):
                        func = sub.exc.func
                        if isinstance(func, ast.Name) and func.id == "RuntimeError":
                            raises = True
    _check(raises, f"{path.name}: install cell must raise RuntimeError when already-imported packages change")


def _validate_notebook_content(
    path: Path,
    code_cells: list[tuple[int, str, ast.Module]],
    markdown: str,
    embedded: set[int],
    spec: dict,
    template: dict,
) -> None:
    model_id, _revision = _package_identity(template)
    stripped = {index: _strip_comments(source) for index, source, _ in code_cells}
    code = "\n".join(stripped.values())
    install_index = min(stripped)  # the generator-owned install cell is the first code cell
    outside = "\n".join(text for index, text in stripped.items() if index not in embedded)
    outside_after_install = "\n".join(
        text for index, text in stripped.items() if index not in embedded and index != install_index
    )
    missing = [marker for marker in COMMON_CODE_MARKERS + spec["code_markers"] if marker not in code]
    _check(not missing, f"{path.name}: missing required source markers: {missing}")
    present = [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(code)]
    _check(not present, f"{path.name}: forbidden/insecure source: {present}")
    leaked = [marker for marker in FORBIDDEN_OUTSIDE_MODULE if marker in outside]
    _check(not leaked, f"{path.name}: direct library use outside the carried module cells (G2): {leaked}")
    worker = [marker for marker in FORBIDDEN_OUTSIDE_MODULE_FLEET if marker in outside_after_install]
    _check(not worker, f"{path.name}: worker/subprocess path outside the generator-owned cells: {worker}")
    forbidden = [marker for marker in spec.get("forbidden_code", ()) if marker in outside]
    _check(not forbidden, f"{path.name}: profile-forbidden code outside the carried module cells: {forbidden}")
    _check(
        f"pipe = {MODEL_LOAD_EXPR}" in outside,
        f"{path.name}: must load through {MODEL_LOAD_EXPR} (INF1)",
    )
    _validate_gates(path, code_cells, spec["byod_gates"])
    _validate_bootstrap_guard(path, code_cells)
    for filename in spec["expected_outputs"]:
        _check(filename in code, f"{path.name}: must export {filename}")
    missing_md = [marker for marker in COMMON_MARKDOWN_MARKERS + spec["markdown_markers"] if marker not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check(f"**Profile:** `{spec['profile']}`" in markdown, f"{path.name}: markdown must state the profile")
    ref = template.get("model_host", {}).get("reference_url", f"https://huggingface.co/{model_id}")
    _check(ref in markdown, f"{path.name}: references must link {ref}")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    notebooks = sorted(tutorials.glob("*.ipynb"))
    names = sorted(NOTEBOOKS)
    _check(
        [p.name for p in notebooks] == names,
        f"tutorial notebooks must be exactly {names}, found {[p.name for p in notebooks]}",
    )
    build = _load_tool("build_notebook")
    registry = _read(tutorials / "README.md")
    for path in notebooks:
        spec = NOTEBOOKS[path.name]
        template = _load_tool(spec["template"]).TEMPLATE
        _check(template["notebook_name"] == path.name, f"tools/{spec['template']}.py must name {path.name}")
        _check(template["profile"] == spec["profile"], f"tools/{spec['template']}.py profile must be {spec['profile']}")
        notebook = json.loads(_read(path))
        code_cells, markdown = _validate_notebook_structure(path, notebook, spec, template, build)
        embedded = _validate_embedded_modules(path, notebook, build, template)
        _model_id, revision = _package_identity(template)
        _validate_identity(path, code_cells, embedded, revision)
        _validate_parity(path, notebook, code_cells, build, template)
        _validate_notebook_content(path, code_cells, markdown, embedded, spec, template)
        _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
        _check(f"`{spec['profile']}`" in registry, f"tutorials/README.md must record `{spec['profile']}`")
    _check(
        f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry,
        "tutorials/README.md must name the notebook spec version",
    )
    _check("standalone" in registry.lower(), "tutorials/README.md must record that the notebooks are standalone")


def validate_all() -> list[str]:
    validate_model_card()
    validate_identity_consistency()
    validate_weight_facts()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "identity-consistency", "weight-facts", "release-status", "notebooks+parity"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
