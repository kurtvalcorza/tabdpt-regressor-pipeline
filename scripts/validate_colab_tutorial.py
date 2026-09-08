#!/usr/bin/env python3
"""Validates Colab tutorial notebooks for AST syntax, pipeline arguments, and path safety."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = ROOT / "tutorials"

# Disallow developer/author workstation paths like C:\Users or /home/username
# Permit Colab paths (/content/...) and web URLs
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[a-zA-Z]:[\\/]"),  # Windows drive paths like C:\ or C:/
    re.compile(r"/(?:Users|home|root)/"),  # Workstation home directories
]


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


def check_pipeline_use_flash(tree: ast.AST, filename: str, cell_idx: int) -> bool:
    """Check if TabDPTRegressionPipeline is called, and verify use_flash=False is passed."""
    found_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == "TabDPTRegressionPipeline":
                found_call = True
                has_use_flash_false = any(
                    kw.arg == "use_flash"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                    for kw in node.keywords
                )
                if not has_use_flash_false:
                    raise AssertionError(
                        f"{filename} (cell {cell_idx}): TabDPTRegressionPipeline must pass "
                        f"`use_flash=False` explicitly for Tesla T4 portability."
                    )
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


def validate_notebook(nb_path: Path) -> None:
    if not nb_path.exists():
        raise AssertionError(f"Missing notebook: {nb_path}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        raise AssertionError(f"{nb_path.name}: must use nbformat 4")

    pipeline_instantiated = False
    for idx, cell in enumerate(nb.get("cells", [])):
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
            pipeline_instantiated = True

    if not pipeline_instantiated:
        raise AssertionError(f"{nb_path.name}: TabDPTRegressionPipeline was never instantiated.")
    print(f"[PASS] Validated {nb_path.name}")


def main() -> None:
    notebooks = sorted(TUTORIALS_DIR.glob("*.ipynb"))
    if not notebooks:
        raise AssertionError(f"No notebooks found in {TUTORIALS_DIR}")
    for nb_path in notebooks:
        validate_notebook(nb_path)
    print(f"All {len(notebooks)} tutorial notebooks validated successfully.")


if __name__ == "__main__":
    main()
