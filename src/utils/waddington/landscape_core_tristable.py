"""
Self-contained definition of the Goldilocks potential landscape
and Langevin simulation utilities.

Expose only two public functions:
    * V_total(x, y)        – scalar potential value
    * grad_V_total(x, y)   – analytical gradient (∂V/∂x, ∂V/∂y)

All other symbols are considered internal.
Import elsewhere with::

    from utils.waddington.landscape_core import V_total, grad_V_total
"""


from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Iterable

# -----------------------------------------------------------------------------
#  ─── Core potential parameters (internal) ───────────────────────────────────
# -----------------------------------------------------------------------------

B = 2.0
sx = sy = 0.6
well_depth = 5.0
s_well = 0.7
wells: list[Tuple[float, float]] = [(-3.0, 0.0), (2.0, 2.0), (2.0, -2.0)]
_two_sx2 = 2 * sx**2
_two_sy2 = 4 * sy**2   # note: original had 4*sy**2 here
_two_sw2 = 2 * s_well**2

# -----------------------------------------------------------------------------
#  ─── Elementary potential and gradient ──────────────────────────────────────
# -----------------------------------------------------------------------------

def V_orig(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    b1 = B * np.exp(-(x**2/_two_sx2 + (y+2)**2/_two_sy2))
    b2 = B * np.exp(-(x**2/_two_sx2 + (y-2)**2/_two_sy2))
    wells_sum = sum(
        -well_depth * np.exp(-((x-xi)**2 + (y-yi)**2)/_two_sw2)
        for xi, yi in wells
    )
    return b1 + b2 + wells_sum

    
def grad_V_orig(x: np.ndarray | float, y: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
    e1 = np.exp(-(x**2/_two_sx2 + (y+2)**2/_two_sy2))
    e2 = np.exp(-(x**2/_two_sx2 + (y-2)**2/_two_sy2))
    dVdx = B*(e1 + e2)*(-2*x/_two_sx2)
    dVdy = B*e1*(-2*(y+2)/_two_sy2) + B*e2*(-2*(y-2)/_two_sy2)  # note: (y-2)
    for xi, yi in wells:
        e = np.exp(-((x-xi)**2 + (y-yi)**2)/_two_sw2)
        dVdx += well_depth * e * (2*(x-xi)/_two_sw2)
        dVdy += well_depth * e * (2*(y-yi)/_two_sw2)
    return dVdx, dVdy

# Public aliases
V_total = V_orig
grad_V_total = grad_V_orig


'''
def simulate_langevin(
    D: float,
    num_traj: int = 300,
    N_steps: int = 400,
    dt: float = 0.02,
    snap_times: tuple[int, ...] = (0, 200, 399),
    seed: int | None = None,
    plot_original: bool = True
) -> Dict[str, Any]:
    """
    Stochastic Euler–Maruyama on the original 2-barrier/3-well potential.
    """
    if seed is not None:
        np.random.seed(seed)

    sqrt2Ddt = np.sqrt(2 * D * dt)
    mean = [0.0, 0.0]
    cov = [[0.1, 0], [0, 0.1]]
    X0 = np.random.multivariate_normal(mean, cov, size=num_traj)

    Xs = np.zeros((num_traj, N_steps))
    Ys = np.zeros((num_traj, N_steps))
    Xs[:, 0], Ys[:, 0] = X0[:, 0], X0[:, 1]

    for t in range(N_steps - 1):
        x, y = Xs[:, t], Ys[:, t]
        gx, gy = grad_V_orig(x, y)
        Xs[:, t + 1] = x - gx * dt + sqrt2Ddt * np.random.randn(num_traj)
        Ys[:, t + 1] = y - gy * dt + sqrt2Ddt * np.random.randn(num_traj)

    snapshots = {t: np.stack([Xs[:, t], Ys[:, t]], axis=1)
                 for t in snap_times}

    if plot_original:
        plot_langevin_contours(Xs, Ys)

    return {
        'Xs': Xs,
        'Ys': Ys,
        'snapshots': snapshots,
        'Vfun': V_orig
    }

def generate_datasets_for_noises(
    noise_levels: Iterable[float],
    **sim_params: Any
) -> dict[float, dict[str, Any]]:
    return {D: simulate_langevin(D, **sim_params) for D in noise_levels}
'''
# -----------------------------------------------------------------------------
#  ─── Module exports ─────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
__all__ = ["V_total", "grad_V_total"]

