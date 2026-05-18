"""Beam and Earth Engine helpers for running data pipelines"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("geebeam")
except PackageNotFoundError:
    __version__ = "unknown"    

from . import pipeline
from . import sampler
from .pipeline import run_pipeline, sample_and_run_pipeline, grid_and_run_pipeline

__all__ = [
    "pipeline",
    "sampler",
    "run_pipeline",
    "sample_and_run_pipeline",
    "grid_and_run_pipeline"
]
