"""utils.waddington.landscape_core
=================================

Expose only two public functions:
    * ``V_total(x, y)``  – scalar potential value
    * ``grad_V_total(x, y)`` – analytical gradient (∂V/∂x, ∂V/∂y)
All other symbols are considered internal.  Import elsewhere with::

    from utils.waddington.landscape_core import V_total, grad_V_total
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

# -----------------------------------------------------------------------------
#  ─── Global shape constants ──────────────────────────────────────────────────
# -----------------------------------------------------------------------------
HEIGHT_SCALE = 0.75  # global gain factor on the entire landscape

# Peak (zygote) parameters
peak_height: float = 8.0
peak_center: Tuple[float, float] = (0.0, 0.0)
peak_width: float = 3.0

# First ring (T1) wells – equilateral triangle
T1_radius: float = 2.4
T1_angles: Tuple[float, float, float] = (
    0.0,
    2 * math.pi / 3,
    4 * math.pi / 3,
)
T1_positions: list[Tuple[float, float]] = [
    (T1_radius * math.cos(a), T1_radius * math.sin(a)) for a in T1_angles
]

# Second ring (T2) – two offspring per T1 sector
T2_radius: float = 4.8
_sector_spread: float = math.pi / 3
T2_positions: list[Tuple[float, float]] = []
for base in T1_angles:
    for delta in (-_sector_spread / 2, +_sector_spread / 2):
        T2_positions.append(
            (T2_radius * math.cos(base + delta), T2_radius * math.sin(base + delta))
        )

# Third ring (T3) – one per T2
T3_radius: float = 8.8
T3_positions: list[Tuple[float, float]] = []
for x_t2, y_t2 in T2_positions:
    theta = math.atan2(y_t2, x_t2)
    T3_positions.append((T3_radius * math.cos(theta), T3_radius * math.sin(theta)))

# Well depths (more differentiated ⇒ deeper)
T1_well_depth: float = 3.0
T2_well_depth: float = 3.2
T3_well_depth: float = 3.6

# Pathway & miscellaneous widths/depths
well_width: float = 1.1
pathway_depth: float = 0.5
pathway_width: float = 0.7
BACKBOARD_radius: float = 12.0
boundary_slope: float = 5.0

# -----------------------------------------------------------------------------
#  ─── Derived constants & helper geometry ─────────────────────────────────────
# -----------------------------------------------------------------------------
_two_well_w2 = 2.0 * well_width**2
_two_pathway_w2 = 2.0 * pathway_width**2

# Lineage pathway centre points (T0→T1, T1→T2, T2→T3)
pathway_centers: list[Tuple[float, float]] = []

# T0 → T1
for x1, y1 in T1_positions:
    dx, dy = x1 - peak_center[0], y1 - peak_center[1]
    for j in range(1, 5):
        f = j / 5.0
        pathway_centers.append((peak_center[0] + f * dx, peak_center[1] + f * dy))

# T1 → T2 (each T1 has two daughters)
for k, (x1, y1) in enumerate(T1_positions):
    for x2, y2 in T2_positions[2 * k : 2 * k + 2]:
        dx, dy = x2 - x1, y2 - y1
        for j in range(1, 5):
            f = j / 5.0
            pathway_centers.append((x1 + f * dx, y1 + f * dy))

# T2 → T3 (one‑to‑one)
for (x2, y2), (x3, y3) in zip(T2_positions, T3_positions):
    dx, dy = x3 - x2, y3 - y2
    for j in range(1, 4):
        f = j / 4.0
        pathway_centers.append((x2 + f * dx, y2 + f * dy))

# -----------------------------------------------------------------------------
#  ─── Elementary potential components ─────────────────────────────────────────
# -----------------------------------------------------------------------------

def _mountain_peak(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    return peak_height * np.exp(
        -((x - peak_center[0]) ** 2 + (y - peak_center[1]) ** 2) / (2.0 * peak_width**2)
    )


def _wells(
    x: np.ndarray | float,
    y: np.ndarray | float,
    centres: Iterable[Tuple[float, float]],
    depth: float,
) -> np.ndarray:
    return sum(
        -depth * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / _two_well_w2)
        for cx, cy in centres
    )


def _paths(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    return sum(
        -pathway_depth * np.exp(-((x - px) ** 2 + (y - py) ** 2) / _two_pathway_w2)
        for px, py in pathway_centers
    )


def _boundary(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    return -boundary_slope * (np.hypot(x, y) / BACKBOARD_radius) ** 2


# -----------------------------------------------------------------------------
#  ─── Public API ──────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------

def V_total(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Total potential evaluated at (x, y).  Accepts scalars or NumPy arrays."""

    return HEIGHT_SCALE * (
        _mountain_peak(x, y)
        + _wells(x, y, T1_positions, T1_well_depth)
        + _wells(x, y, T2_positions, T2_well_depth)
        + _wells(x, y, T3_positions, T3_well_depth)
        + _paths(x, y)
        + _boundary(x, y)
    )


def grad_V_total(x: np.ndarray | float, y: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical gradient (∂V/∂x, ∂V/∂y) evaluated element‑wise."""

    # Gradient of the mountain peak
    peak_factor = -((x - peak_center[0]) ** 2 + (y - peak_center[1]) ** 2) / (2.0 * peak_width**2)
    exp_peak = np.exp(peak_factor)
    gx_peak = peak_height * exp_peak * (-(x - peak_center[0]) / peak_width**2)
    gy_peak = peak_height * exp_peak * (-(y - peak_center[1]) / peak_width**2)

    # Gradient helpers
    def _wells_grad(centres: Iterable[Tuple[float, float]], depth: float):
        gx = gy = 0.0
        for cx, cy in centres:
            factor = -((x - cx) ** 2 + (y - cy) ** 2) / _two_well_w2
            exp_ = np.exp(factor)
            gx += -depth * exp_ * (-(x - cx) / well_width**2)
            gy += -depth * exp_ * (-(y - cy) / well_width**2)
        return gx, gy

    gx_t1, gy_t1 = _wells_grad(T1_positions, T1_well_depth)
    gx_t2, gy_t2 = _wells_grad(T2_positions, T2_well_depth)
    gx_t3, gy_t3 = _wells_grad(T3_positions, T3_well_depth)

    # Pathway gradient
    gx_paths = gy_paths = 0.0
    for px, py in pathway_centers:
        factor = -((x - px) ** 2 + (y - py) ** 2) / _two_pathway_w2
        exp_ = np.exp(factor)
        gx_paths += -pathway_depth * exp_ * (-(x - px) / pathway_width**2)
        gy_paths += -pathway_depth * exp_ * (-(y - py) / pathway_width**2)

    # Boundary gradient (quadratic radial bowl)
    boundary_factor = -2.0 * boundary_slope / BACKBOARD_radius**2
    gx_boundary = boundary_factor * x
    gy_boundary = boundary_factor * y

    gx = HEIGHT_SCALE * (gx_peak + gx_t1 + gx_t2 + gx_t3 + gx_paths + gx_boundary)
    gy = HEIGHT_SCALE * (gy_peak + gy_t1 + gy_t2 + gy_t3 + gy_paths + gy_boundary)

    return gx, gy