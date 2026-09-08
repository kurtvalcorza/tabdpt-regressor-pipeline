import ast
import json
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_colab_tutorial import (
    clean_code_for_ast,
    check_pipeline_use_flash,
    check_absolute_paths,
    validate_notebook,
)


def test_validator_accepts_clean_pipeline_call():
    code = "pipe = TabDPTRegressionPipeline(compile_model=False, use_flash=False)"
    tree = ast.parse(clean_code_for_ast(code))
    assert check_pipeline_use_flash(tree, "test.ipynb", 1) is True


def test_validator_rejects_missing_use_flash():
    code = "pipe = TabDPTRegressionPipeline(compile_model=False)"
    tree = ast.parse(clean_code_for_ast(code))
    with pytest.raises(AssertionError, match="must pass `use_flash=False` explicitly"):
        check_pipeline_use_flash(tree, "test.ipynb", 1)


def test_validator_rejects_use_flash_true():
    code = "pipe = TabDPTRegressionPipeline(use_flash=True)"
    tree = ast.parse(clean_code_for_ast(code))
    with pytest.raises(AssertionError, match="must pass `use_flash=False` explicitly"):
        check_pipeline_use_flash(tree, "test.ipynb", 1)


def test_validator_rejects_syntax_error(tmp_path):
    nb_content = {
        "cells": [
            {"cell_type": "code", "source": ["def broken_syntax(\n"]}
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_file = tmp_path / "broken.ipynb"
    nb_file.write_text(json.dumps(nb_content), encoding="utf-8")
    with pytest.raises(AssertionError, match="Syntax error in broken.ipynb"):
        validate_notebook(nb_file)


def test_validator_rejects_absolute_path():
    code = 'dataset_path = "C:\\\\Users\\\\Kurt\\\\data.csv"'
    with pytest.raises(AssertionError, match="Absolute filesystem path detected"):
        check_absolute_paths(code, "test.ipynb", 1)
