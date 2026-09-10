#!/usr/bin/env python3
"""Statically validate DIMER Colab tutorials against notebook-spec v1 source contracts.

This validator deliberately does not claim runtime execution evidence. It checks notebook JSON,
profile declarations, Python-cell syntax, portable TabDPT entrypoints, source hygiene, and a small
set of profile-specific structural invariants that are falsifiable without downloading the model.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = ROOT / "tutorials"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
EXPECTED_PROFILES = {
    "tabdpt_regressor_colab.ipynb": "E2E",
    "tabdpt_regressor_artifact_inference_colab.ipynb": "ARTIFACT-INFERENCE",
}

# Disallow developer/author workstation paths like C:\Users or /home/username.
# Colab runtime paths (/content/...) are intentionally permitted.
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[a-zA-Z]:[\\/]"),
    re.compile(r"/(?:Users|home|root)/"),
]
PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|TBD|FIXME)\b", re.IGNORECASE)


def clean_code_for_ast(code: str) -> str:
    """Comment out IPython magic and shell commands for Python AST parsing."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("%", "!")):
            lines.append(f"# {line}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _requires_use_flash_false(node: ast.Call, filename: str, cell_idx: int) -> None:
    has_use_flash_false = any(
        kw.arg == "use_flash"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
        for kw in node.keywords
    )
    if not has_use_flash_false:
        raise AssertionError(
            f"{filename} (cell {cell_idx}): TabDPT runtime entrypoints must pass "
            "`use_flash=False` explicitly for Tesla T4 portability."
        )


def check_pipeline_use_flash(tree: ast.AST, filename: str, cell_idx: int) -> bool:
    """Validate all user-facing TabDPT construction/reload calls for portable FlashAttention settings."""
    found_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in {"TabDPTRegressionPipeline", "load_verified_artifact"}:
            found_call = True
            _requires_use_flash_false(node, filename, cell_idx)
    return found_call


def check_absolute_paths(code: str, filename: str, cell_idx: int) -> None:
    for line in code.splitlines():
        if "http://" in line or "https://" in line:
            continue
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(line):
                raise AssertionError(
                    f"{filename} (cell {cell_idx}): Absolute filesystem path detected: {line.strip()}"
                )


def _source_text(nb: dict) -> str:
    parts: list[str] = []
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        parts.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(parts)


def _require_markers(nb_path: Path, profile: str, source_text: str, required: dict[str, str]) -> None:
    for label, marker in required.items():
        if marker not in source_text:
            raise AssertionError(f"{nb_path.name}: {profile} contract missing {label} ({marker!r})")


def _validate_profile_contract(nb_path: Path, nb: dict, source_text: str) -> None:
    dimer = nb.get("metadata", {}).get("dimer", {})
    profile = dimer.get("notebook_profile")
    if profile not in ALLOWED_PROFILES:
        raise AssertionError(f"{nb_path.name}: missing/invalid metadata.dimer.notebook_profile")
    if dimer.get("notebook_spec") != "1.0":
        raise AssertionError(f"{nb_path.name}: metadata.dimer.notebook_spec must be '1.0'")
    expected = EXPECTED_PROFILES.get(nb_path.name)
    if expected and profile != expected:
        raise AssertionError(f"{nb_path.name}: expected profile {expected}, got {profile}")

    if PLACEHOLDER_PATTERN.search(source_text):
        raise AssertionError(f"{nb_path.name}: unresolved TODO/TBD/FIXME placeholder found")

    common = {
        "supported Python floor": "Python 3.11+",
        "runtime Python guard": "sys.version_info < (3, 11)",
        "point-prediction semantics": "point estimate",
    }
    _require_markers(nb_path, profile, source_text, common)

    if profile == "E2E":
        required = {
            "BYOD path": "USE_BYOD",
            "meaningful regression baseline": "training-mean baseline",
            "new-data inference": ".predict(",
            "machine-readable predictions": "to_csv(",
            "machine-readable provenance": "provenance",
            "artifact export": "export_artifact_bundle(",
            "fresh artifact reload": "load_verified_artifact(",
            "no-refit reload assertion": "preprocessing_restored_",
            "reload equivalence": "assert_allclose(",
            "regression MAE": "mae",
            "regression RMSE": "rmse",
            "regression R2": "r2",
            "missing-value policy": "numericMissingPolicy",
            "categorical missing policy": "categoricalMissingPolicy",
            "unknown-category policy": "unknownCategoryPolicy",
            "feature ceiling": "modelFeatureCeiling",
            "feature reduction visibility": "featureReductionActive",
            "context visibility": "effective support rows",
        }
        _require_markers(nb_path, profile, source_text, required)
        if "smoke tutorial" in source_text.lower():
            raise AssertionError(f"{nb_path.name}: E2E notebook must not identify itself as smoke")

    if profile == "ARTIFACT-INFERENCE":
        required = {
            "external upload": "files.upload(",
            "pre-load artifact validation": "validate_artifact_bundle(",
            "verified serving reconstruction": "load_verified_artifact(",
            "no-refit reload assertion": "preprocessing_restored_",
            "new-data inference": ".predict(",
            "machine-readable predictions": "to_csv(",
            "provenance export": "provenance",
            "trust-boundary explanation": "Trust boundary",
            "legacy compatibility distinction": "legacy compatibility",
            "missing-value policy": "numericMissingPolicy",
            "categorical missing policy": "categoricalMissingPolicy",
            "unknown-category policy": "unknownCategoryPolicy",
            "feature ceiling": "modelFeatureCeiling",
            "feature reduction visibility": "featureReductionActive",
            "context visibility": "effective support rows",
        }
        _require_markers(nb_path, profile, source_text, required)
        forbidden = {
            "self-created sample dataset": "load_diabetes(",
            "artifact creation": "export_artifact_bundle(",
            "in-notebook support fitting": ".fit(",
        }
        for label, marker in forbidden.items():
            if marker in source_text:
                raise AssertionError(
                    f"{nb_path.name}: ARTIFACT-INFERENCE must not perform {label} ({marker!r})"
                )


def validate_notebook(nb_path: Path) -> None:
    if not nb_path.exists():
        raise AssertionError(f"Missing notebook: {nb_path}")
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{nb_path.name}: invalid notebook JSON: {exc}") from exc
    if nb.get("nbformat") != 4:
        raise AssertionError(f"{nb_path.name}: must use nbformat 4")
    if not isinstance(nb.get("cells"), list) or not nb["cells"]:
        raise AssertionError(f"{nb_path.name}: must contain notebook cells")

    # Preserve syntax/path failures as the first-order source check. Profile metadata and
    # higher-level semantic assertions run only after every code cell parses successfully.
    runtime_entrypoint_found = False
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("execution_count") is not None:
            raise AssertionError(f"{nb_path.name} (cell {idx}): execution_count must be cleared")
        if cell.get("outputs") not in (None, []):
            raise AssertionError(f"{nb_path.name} (cell {idx}): persisted outputs must be cleared")
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        code = "".join(src) if isinstance(src, list) else str(src)
        if not code.strip():
            continue

        check_absolute_paths(code, nb_path.name, idx)
        clean_code = clean_code_for_ast(code)
        try:
            tree = ast.parse(clean_code, filename=f"{nb_path.name}:cell_{idx}")
        except SyntaxError as exc:
            raise AssertionError(f"Syntax error in {nb_path.name} (cell {idx}): {exc}") from exc

        if check_pipeline_use_flash(tree, nb_path.name, idx):
            runtime_entrypoint_found = True

    source_text = _source_text(nb)
    _validate_profile_contract(nb_path, nb, source_text)

    if not runtime_entrypoint_found:
        raise AssertionError(f"{nb_path.name}: supported TabDPT runtime entrypoint was never exercised")
    print(f"[PASS] Validated {nb_path.name}")


def main() -> None:
    notebooks = sorted(TUTORIALS_DIR.glob("*.ipynb"))
    if not notebooks:
        raise AssertionError(f"No notebooks found in {TUTORIALS_DIR}")
    for nb_path in notebooks:
        validate_notebook(nb_path)
    print(f"All {len(notebooks)} tutorial notebooks passed notebook-spec v1 static validation.")
    print("NOTE: static validation is not clean-runtime execution evidence.")


if __name__ == "__main__":
    main()
