"""Tests for the reproducibility helpers."""

import random

from alcohol_detector.seed import DEFAULT_SEED, set_global_seed


def test_default_seed_is_project_seed() -> None:
    assert DEFAULT_SEED == 1998


def test_set_global_seed_makes_python_random_repeatable() -> None:
    set_global_seed(DEFAULT_SEED)
    first = [random.random() for _ in range(5)]
    set_global_seed(DEFAULT_SEED)
    second = [random.random() for _ in range(5)]
    assert first == second
