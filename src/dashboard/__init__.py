"""dashboard - experiment tracking and ablation tooling over a small MLP."""

from .data import get_dataloaders
from .model import MLP
from .experiment import ExperimentConfig, run_experiment

__all__ = ["get_dataloaders", "MLP", "ExperimentConfig", "run_experiment"]
__version__ = "1.0.0"
