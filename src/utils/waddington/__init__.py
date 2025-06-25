"""Top-level *utils/waddington* package initializer
=====================================

* Re-exports the two most commonly used landscape functions so callers can
  write shorter import statements:

    >>> from utils import V_total, grad_V_total

* Provides ``seed_everything`` – a convenience wrapper that seeds Python,
  NumPy, PyTorch (if present) and JAX (if present) for deterministic runs.
"""
from __future__ import annotations

import os
import random
from typing import Dict, Any

import numpy as np

# -----------------------------------------------------------------------------
#  Optional big-dependency availability checks ---------------------------------
# -----------------------------------------------------------------------------
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.random as jax_random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# -----------------------------------------------------------------------------
#  Re-export most used landscape API -------------------------------------------
# -----------------------------------------------------------------------------
from src.utils.waddington.landscape_core import V_total, grad_V_total  # noqa: E402

__all__ = [
    "V_total",
    "grad_V_total",
    "seed_everything",
]

# -----------------------------------------------------------------------------
#  Repro helper ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42, verbose: bool = True) -> Dict[str, Any]:
    """Seed Python, NumPy (+optional PyTorch, JAX) for reproducibility.

    Returns a dict of interesting state objects (e.g. the JAX key) so callers
    can stash them if needed.
    """

    if verbose:
        print(f"🌱 Setting global seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    result: Dict[str, Any] = {"seed": seed}

    # PyTorch ---------------------------------------------------------------
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if verbose:
                print("  ✓ PyTorch + CUDA seeded")
        elif verbose:
            print("  ✓ PyTorch seeded")
    elif verbose:
        print("  ⚠ PyTorch not available")

    # JAX -------------------------------------------------------------------
    if JAX_AVAILABLE:
        jax_key = jax_random.PRNGKey(seed)
        globals()["JAX_KEY"] = jax_key  # make accessible module-wide
        result["jax_key"] = jax_key
        if verbose:
            print("  ✓ JAX PRNG key created")
    elif verbose:
        print("  ⚠ JAX not available")

    if verbose:
        print("🌱 All available RNGs seeded.")

    return result