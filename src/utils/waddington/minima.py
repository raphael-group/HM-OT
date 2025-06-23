"""utils.waddington.minima
==========================
Local-minima detection on the Goldilocks landscape **with canonical labels**.

Canonical ring codes
--------------------
* **A** – innermost ring (3 minima)
* **B** – middle ring   (6 minima)
* **C** – outer ring    (6 minima)

Public helpers
--------------
* ``find_local_minima`` – optionally returns *index → label* mapping.
* ``label_minima`` – quick accessor for the same mapping.
* ``classify_minima_by_ring`` – ring assignment (returns **A/B/C** codes).
* ``mark_minima_regions`` – boolean masks for colouring.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import string

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize

from utils.waddington.landscape_core import (
    V_total,
    grad_V_total,
    T1_positions,
    T2_positions,
    T3_positions,
)

__all__ = [
    "find_local_minima",
    "label_minima",
    "classify_minima_by_ring",
    "mark_minima_regions",
]

# -----------------------------------------------------------------------------
# 1) Core detector -------------------------------------------------------------
# -----------------------------------------------------------------------------

def find_local_minima(
    grid_res: int = 400,
    min_distance: float = 0.2,
    tolerance: float = 1e-6,
    *,
    return_mapping: bool = False,
    extra_candidates: Iterable[Tuple[float, float]] | None = None,
):
    """Return minima in deterministic radius–angle order.

    If *return_mapping* is *True* an **index → label** dict is returned. Labels
    follow the *A/B/C + angle* scheme (e.g. ``A240``).
    """

    # 1) coarse grid --------------------------------------------------------
    x = np.linspace(-12.0, 12.0, grid_res)
    y = np.linspace(-12.0, 12.0, grid_res)
    X, Y = np.meshgrid(x, y)
    Z = V_total(X, Y)

    is_local = minimum_filter(Z, size=3) == Z
    candidates = [(X[i, j], Y[i, j]) for i, j in zip(*np.where(is_local))]

    # 2) domain-knowledge seeds -------------------------------------------
    jitter: List[Tuple[float, float]] = []
    for wx, wy in (*T1_positions, *T2_positions, *T3_positions):
        jitter.append((wx, wy))
        for dx in (-0.1, 0.0, 0.1):
            for dy in (-0.1, 0.0, 0.1):
                if dx or dy:
                    jitter.append((wx + dx, wy + dy))

    radial = [
        (r * np.cos(a), r * np.sin(a))
        for r in np.linspace(1.5, 10.0, 20)
        for a in np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    ]

    all_cand = candidates + jitter + radial
    if extra_candidates:
        all_cand += list(extra_candidates)

    # 3) refinement ---------------------------------------------------------
    refined: List[Tuple[float, float]] = []
    bounds = [(-15.0, 15.0), (-15.0, 15.0)]

    def obj(p):
        return V_total(p[0], p[1])

    def grad(p):
        return grad_V_total(p[0], p[1])

    for x0, y0 in all_cand:
        if x0 * x0 + y0 * y0 > 225.0:
            continue
        res = minimize(
            obj,
            (x0, y0),
            jac=grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"gtol": tolerance, "maxiter": 200},
        )
        if not res.success:
            continue
        xm, ym = res.x
        gx, gy = grad_V_total(xm, ym)
        if (
            abs(xm) < 14.5
            and abs(ym) < 14.5
            and np.hypot(gx, gy) < tolerance
            and all(np.hypot(xm - px, ym - py) >= min_distance for px, py in refined)
        ):
            refined.append((xm, ym))

    # 4) deterministic ordering -------------------------------------------
    tagged = [(x, y, np.hypot(x, y), np.arctan2(y, x)) for x, y in refined]
    tagged.sort(key=lambda t: (round(t[2], 12), round(t[3], 12)))
    ordered = [(x, y) for x, y, _, _ in tagged]

    if not return_mapping:
        return ordered

    mapping = _build_label_mapping(tagged)
    return ordered, mapping

# -----------------------------------------------------------------------------
# 2) Canonical label helpers ---------------------------------------------------
# -----------------------------------------------------------------------------

def _build_label_mapping(tagged: List[Tuple[float, float, float, float]]) -> Dict[int, str]:
    radii: List[float] = []
    for _, _, r, _ in tagged:
        if all(abs(r - rr) > 1e-2 for rr in radii):
            radii.append(r)
    ring_letter = {r: string.ascii_uppercase[i] for i, r in enumerate(radii)}

    mapping: Dict[int, str] = {}
    for idx, (_, _, r, th) in enumerate(tagged):
        letter = ring_letter[next(rr for rr in radii if abs(r - rr) <= 1e-2)]
        ang = int(round(((np.degrees(th) + 360) % 360) / 10.0)) * 10
        if ang == 360:
            ang = 0
        mapping[idx] = f"{letter}{ang}"
    return mapping


def label_minima(
    minima_points: Iterable[Tuple[float, float]] | None = None,
) -> Dict[int, str]:
    """Return the canonical mapping without re-running optimisation."""
    if minima_points is None:
        _, mapping = find_local_minima(return_mapping=True)
        return mapping

    tagged = [(x, y, np.hypot(x, y), np.arctan2(y, x)) for x, y in minima_points]
    tagged.sort(key=lambda t: (round(t[2], 12), round(t[3], 12)))
    return _build_label_mapping(tagged)

# -----------------------------------------------------------------------------
# 3) Ring classification & region masks ---------------------------------------
# -----------------------------------------------------------------------------

def classify_minima_by_ring(minima_points):
    """Classify minima into rings **A/B/C** by radius."""
    minima_points = np.asarray(minima_points)
    radii = np.linalg.norm(minima_points, axis=1)
    order = np.argsort(radii)
    labels = np.empty(len(minima_points), dtype=object)
    for rank, idx in enumerate(order):
        if rank < 3:
            labels[idx] = "A"
        elif rank < 9:
            labels[idx] = "B"
        else:
            labels[idx] = "C"
    return [(float(x), float(y), lab) for (x, y), lab in zip(minima_points, labels)]


def mark_minima_regions(X, Y, classified_minima, ball_radius: float = 1.4):
    """Return boolean masks selecting the catchment basin of each ring."""
    a_mask = np.zeros_like(X, dtype=bool)
    b_mask = np.zeros_like(X, dtype=bool)
    c_mask = np.zeros_like(X, dtype=bool)
    for x_min, y_min, ring in classified_minima:
        dist = np.hypot(X - x_min, Y - y_min)
        mask = dist < ball_radius
        if ring == "A":
            a_mask |= mask
        elif ring == "B":
            b_mask |= mask
        elif ring == "C":
            c_mask |= mask
    return a_mask, b_mask, c_mask
