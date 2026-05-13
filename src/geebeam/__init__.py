"""Beam and Earth Engine helpers for running data pipelines"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("geebeam")
except PackageNotFoundError:
    __version__ = "unknown"    
    
from . import sampler
from ._pipeline import run_pipeline, sample_and_run_pipeline, grid_and_run_pipeline

__all__ = [
    "sampler",
    "run_pipeline",
    "sample_and_run_pipeline",
    "grid_and_run_pipeline"
]
