"""Concise package and user-surface checks."""

from __future__ import annotations

import json
from pathlib import Path


CUDA_ROOT = Path(__file__).parents[1]


def test_package_metadata_and_license_agree():
    metadata = (CUDA_ROOT / "pyproject.toml").read_text()
    build, project = metadata.split("[project]", maxsplit=1)
    project = project.split("[project.optional-dependencies]", maxsplit=1)[0]
    assert 'requires = ["setuptools>=77", "wheel"]' in build
    assert 'version = "0.2.0"' in project
    assert 'license = "MIT"' in project
    assert 'license-files = ["LICENSE"]' in project
    assert '"torch>=2.11.0"' in project
    assert "matplotlib" not in project
    assert (CUDA_ROOT / "LICENSE").read_text().startswith("MIT License\n")


def test_readme_and_notebooks_are_compact_and_portable():
    readme = (CUDA_ROOT / "README.md").read_text()
    assert 100 <= len(readme.splitlines()) <= 150
    assert "estimation-calibration-cuda train example" in readme
    assert "contact_process_covariance" in readme
    assert "about **0.8–0.9 ms/step**" in readme

    notebooks = sorted((CUDA_ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "covariance_calibration_run.ipynb",
        "covariance_tuning_tutorial.ipynb",
    ]
    documents = {path.name: json.loads(path.read_text()) for path in notebooks}
    assert all(document["nbformat"] == 4 for document in documents.values())
    tutorial = json.dumps(documents["covariance_tuning_tutorial.ipynb"])
    benchmark = json.dumps(documents["covariance_calibration_run.ipynb"])
    assert "_SgScalar" in tutorial
    assert "cuda-graph-compile" in benchmark
    assert "ms_per_step" in benchmark
    assert "/home/" not in tutorial + benchmark
