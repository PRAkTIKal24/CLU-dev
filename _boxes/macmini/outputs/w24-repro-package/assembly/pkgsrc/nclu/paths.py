"""Locations of the dataset, the feature caches, the frozen weights and the run outputs.

Every location is read from an environment variable and otherwise defaults to a path
relative to the package root (the directory that holds ``pyproject.toml``):

  NCLU_DATA_ROOT      dataset root, holds ``shapes3d/3dshapes.h5`` and the split files
                      (default ``data``)
  NCLU_FEATURES_ROOT  frozen-encoder feature caches, ``shapes3d_recon/seed_<s>/features.npy``
                      (default ``data/features``)
  NCLU_WEIGHTS_DIR    frozen decoder and judge classifier (default ``weights``)
  NCLU_OUT            every run output (default ``runs``)
"""

from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _loc(var: str, default: str) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else ROOT / default


def data_root() -> Path:
    return _loc("NCLU_DATA_ROOT", "data")


def features_root() -> Path:
    return _loc("NCLU_FEATURES_ROOT", "data/features")


def weights_dir() -> Path:
    return _loc("NCLU_WEIGHTS_DIR", "weights")


def out_root() -> Path:
    return _loc("NCLU_OUT", "runs")


def run_dir(*parts: str) -> Path:
    return out_root().joinpath(*parts)


def data_manifest() -> dict:
    with open(Path(__file__).resolve().parent / "data_manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


def partition_file(where: str) -> Path:
    """The fit pool and dead-channel mask of a condition (``val`` or ``test``), written by
    ``python -m nclu.experiments.factor_wells --stage partition``."""
    return run_dir("partitions", where, "partitions.npz")
