"""Shared utilities."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all random number generators used in this project.

    Args:
        seed: The seed value to apply to random, numpy and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
