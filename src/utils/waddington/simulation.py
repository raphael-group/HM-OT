"""utils.waddington.simulation
================================
Numerical integration of Langevin dynamics on the Goldilocks landscape
and small helper utilities (track construction, I/O).

Public surface (import via `from utils.waddington.simulation import …`):

* ``simulate_langevin_with_snapshots`` – run a simple Euler–Maruyama
  scheme and return full trajectories **plus** selected snapshot arrays.
* ``build_tracks``  – convert (Xs, Ys) into a Napari-compatible Tracks array.
* ``save_simulation_data`` – persist minima and snapshot data under a
  common prefix.

Only depends on ``utils.waddington.landscape_core``; no other project
modules are imported.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from utils.waddington.landscape_core import V_total, grad_V_total

__all__: list[str] = [
    "simulate_langevin_with_snapshots",
    "build_tracks",
    "save_simulation_data",
]

# -----------------------------------------------------------------------------
#  Langevin dynamics (Euler–Maruyama) -----------------------------------------
# -----------------------------------------------------------------------------

def simulate_langevin_with_snapshots(
    *,
    n_particles: int = 250,
    n_steps: int = 250,
    dt: float = 0.2,
    diffusion: float = 0.01,
    snap_times: Sequence[int] | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Run Langevin dynamics in 2-D.

    Parameters
    ----------
    n_particles: int
        Number of independent trajectories.
    n_steps: int
        Number of discrete time steps (including *t = 0*).
    dt: float
        Euler time-step size.
    diffusion: float
        Diffusion constant *D* (noise strength = :math:`\sqrt{2Ddt}`).
    snap_times: Iterable[int] | None
        Indices at which to record a snapshot array *(N×2)*.
        Defaults to ``[0, n_steps//2, n_steps-1]``.
    rng: numpy.random.Generator | None
        Optional pre-initialised random‐number generator for reproducibility.

    Returns
    -------
    Xs, Ys : ndarray
        Trajectories; each shape ``(n_particles, n_steps)``.
    snapshots : dict[str, ndarray]
        Keys like ``"step_125"`` → snapshot array of shape *(n_particles, 2)*.
    """

    if rng is None:
        rng = np.random.default_rng()

    if snap_times is None:
        snap_times = [0, n_steps // 2, n_steps - 1]
    snap_times = sorted(set(snap_times))

    sigma = np.sqrt(2.0 * diffusion * dt)

    # initial positions ~ N(0, Σ)
    X0 = rng.multivariate_normal([0.0, 0.0], [[0.15, 0.0], [0.0, 0.15]], n_particles)

    Xs = np.zeros((n_particles, n_steps), dtype=np.float32)
    Ys = np.zeros_like(Xs)
    Xs[:, 0], Ys[:, 0] = X0[:, 0], X0[:, 1]

    for t in range(n_steps - 1):
        gx, gy = grad_V_total(Xs[:, t], Ys[:, t])
        Xs[:, t + 1] = Xs[:, t] - gx * dt + sigma * rng.standard_normal(n_particles)
        Ys[:, t + 1] = Ys[:, t] - gy * dt + sigma * rng.standard_normal(n_particles)

    # collect requested snapshots ------------------------------------------------
    snapshots: dict[str, np.ndarray] = {
        f"step_{t}": np.stack([Xs[:, t], Ys[:, t]], axis=1).astype(np.float32)
        for t in snap_times
    }
    return Xs.astype(np.float32), Ys.astype(np.float32), snapshots


# -----------------------------------------------------------------------------
#  Track construction (for Napari) --------------------------------------------
# -----------------------------------------------------------------------------

def build_tracks(Xs: np.ndarray, Ys: np.ndarray) -> np.ndarray:
    """Convert *Traj* arrays into Napari “tracks” (T, Y, X, value) format."""
    n, T = Xs.shape
    rows = [
        [i, t, V_total(Xs[i, t], Ys[i, t]), Ys[i, t], Xs[i, t]]
        for i in range(n)
        for t in range(T)
    ]
    return np.asarray(rows, dtype=np.float32)


# -----------------------------------------------------------------------------
#  I/O helper ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_simulation_data(
    minima_points: Sequence[Tuple[float, float]],
    snapshots: Dict[str, np.ndarray],
    *,
    out_dir: str | os.PathLike = "simulation_data",
    filename_prefix: str = "goldilocks_data",
) -> Tuple[str, str]:
    """Persist minima locations and snapshot arrays as ``.npy`` / ``.npz`` files."""

    os.makedirs(out_dir, exist_ok=True)

    minima_path = os.path.join(out_dir, f"{filename_prefix}_minima.npy")
    np.save(minima_path, np.asarray(minima_points, dtype=np.float32))

    snap_path = os.path.join(out_dir, f"{filename_prefix}_snapshots.npz")
    np.savez(snap_path, **snapshots)

    print(
        f"Saved {len(minima_points)} minima → {minima_path}\n"
        f"Saved {len(snapshots)} snapshot arrays → {snap_path}"
    )
    return minima_path, snap_path