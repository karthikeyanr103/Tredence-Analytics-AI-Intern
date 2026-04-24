"""
Self-Pruning Neural Network  –  source package
"""
from .model         import PrunableLinear, SelfPruningNet
from .data          import get_loaders
from .trainer       import train_one_epoch, evaluate, run_experiment
from .visualize     import plot_gate_distributions, make_training_gif
from .logging_setup import setup_logging

__all__ = [
    "PrunableLinear", "SelfPruningNet",
    "get_loaders",
    "train_one_epoch", "evaluate", "run_experiment",
    "plot_gate_distributions", "make_training_gif",
    "setup_logging",
]
