"""Unit tests for the seeding utility."""

import random

import numpy as np
import torch

from src.utils import set_seed


def test_same_seed_gives_same_torch_values() -> None:
    """Two runs with the same seed should produce identical torch tensors."""
    set_seed(0)
    a = torch.rand(5)
    set_seed(0)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_different_seeds_give_different_torch_values() -> None:
    """Different seeds should produce different torch tensors."""
    set_seed(0)
    a = torch.rand(5)
    set_seed(1)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_seeds_numpy_and_random_too() -> None:
    """set_seed should cover numpy and the random module, not just torch."""
    set_seed(42)
    a_np, a_py = np.random.rand(3), random.random()
    set_seed(42)
    b_np, b_py = np.random.rand(3), random.random()

    assert np.array_equal(a_np, b_np)
    assert a_py == b_py
