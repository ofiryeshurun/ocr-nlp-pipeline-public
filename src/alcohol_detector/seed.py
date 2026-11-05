"""Reproducibility helpers: one place to seed every source of randomness."""

from __future__ import annotations

import os
import random

DEFAULT_SEED: int = 1998


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and (if installed) PyTorch random generators.

    Optional libraries are imported lazily so this works in any environment.

    Args:
        seed: The seed to apply everywhere.

    Returns:
        The seed that was applied, for logging.
    """
    # Reaches subprocesses only: this interpreter's string hashing was seeded when it started.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return seed
