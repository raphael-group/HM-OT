"""utils.waddington.differentiation
===================================
Cell-type assignment and differentiation-transition analysis on top of
Langevin trajectories (see :pymod:`utils.waddington.simulation`).

Canonical cell-type naming
--------------------------
* *U* – unassigned / progenitor (catch-all)
* *Axxx* – inner ring minima (3 points)
* *Bxxx* – middle ring minima (6 points)
* *Cxxx* – outer ring minima (6 points)
  where *xxx* is the polar angle **rounded to the nearest 10 deg**.

Public surface
--------------
* ``assign_cell_types``
* ``build_differentiation_map``
* ``print_differentiation_summary``
* ``save_differentiation_data``
* ``generate_Q_matrices_from_clusters``
* ``setup_point_clouds_for_waddington_ot``
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from anndata import AnnData
from src.utils.waddington.minima import assign_cell_types

try:
    import torch
except ImportError:  # no-torch environment is fine for most workflows
    torch = None  # type: ignore

__all__ = [
    "assign_cell_types",
    "build_differentiation_map",
    "print_differentiation_summary",
    "save_differentiation_data",
    "generate_Q_matrices_from_clusters",
    "setup_point_clouds_for_waddington_ot",
]

# -----------------------------------------------------------------------------
# Helper – canonical label creation -------------------------------------------
# -----------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Helper – canonical label creation
# -------------------------------------------------------------------------
def _canonical_labels(classified_minima):
    """Return labels like A240, B30… aligned with minima index order."""
    labels = []
    for x, y, ring in classified_minima:
        # ring is already "A", "B", or "C" after the minima refactor
        letter = ring if ring in {"A", "B", "C"} else {"T1": "A", "T2": "B", "T3": "C"}.get(ring, "?")
        angle = int(round((np.degrees(np.arctan2(y, x)) + 360) % 360 / 10.0)) * 10
        if angle == 360:
            angle = 0
        labels.append(f"{letter}{angle}")
    return labels



# -----------------------------------------------------------------------------
# 2) Differentiation map -------------------------------------------------------
# -----------------------------------------------------------------------------

def build_differentiation_map(
    Xs: np.ndarray,  # shape (N×T)
    Ys: np.ndarray,
    minima_points: Sequence[Tuple[float, float]],
    *,
    timepoints: Sequence[int] | None = None,
    assignment_radius: float = 1.4,
    return_transition: str | None = None,
):
    """Create per-timepoint assignments and transition matrices."""

    N, T = Xs.shape
    if timepoints is None:
        timepoints = [0, T // 2, T - 1]
    timepoints = list(timepoints)

    cell_type_assignments: Dict[int, np.ndarray] = {}
    classified_minima = None

    for t in timepoints:
        pos = np.stack((Xs[:, t], Ys[:, t]), axis=1)
        ctype, classified_minima = assign_cell_types(
            pos, minima_points, assignment_radius=assignment_radius
        )
        cell_type_assignments[t] = ctype

    # --- canonical labels ---------------------------------------------------
    type_labels: List[str] = ["U"]  # index 0 = unassigned / progenitor
    type_labels.extend(_canonical_labels(classified_minima))
    n_types = len(type_labels)

    # --- transition matrices ------------------------------------------------
    transition_maps: Dict[str, np.ndarray] = {}
    for t1, t2 in zip(timepoints[:-1], timepoints[1:]):
        mat = np.zeros((n_types, n_types), dtype=int)
        s1, s2 = cell_type_assignments[t1], cell_type_assignments[t2]
        for src, dst in zip(s1, s2):
            mat[src + 1, dst + 1] += 1  # shift −1→0
        transition_maps[f"t{t1}_to_t{t2}"] = mat

    if return_transition is None:
        return transition_maps, cell_type_assignments, type_labels

    if return_transition not in transition_maps:
        raise KeyError(
            f"'{return_transition}' not a valid key.\n"
            f"Available: {list(transition_maps)}"
        )
    return transition_maps[return_transition], cell_type_assignments, type_labels

# -----------------------------------------------------------------------------
# 3) Pretty console output -----------------------------------------------------
# -----------------------------------------------------------------------------

def print_differentiation_summary(
    transition_maps: Dict[str, np.ndarray],
    type_labels: List[str],
):
    for name, mat in transition_maps.items():
        print(f"\n=== {name} ===")
        hdr = "Source".ljust(6) + " " + " ".join(lbl.ljust(6) for lbl in type_labels)
        print(hdr)
        for i, src_lbl in enumerate(type_labels):
            row = src_lbl.ljust(6) + " " + " ".join(str(mat[i, j]).ljust(6) for j in range(len(type_labels)))
            print(row)
        total = mat.sum()
        print(f"Total particles: {total}\n")

# -----------------------------------------------------------------------------
# 4) Persistence --------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_differentiation_data(
    transition_maps: Dict[str, np.ndarray],
    cell_type_assignments: Dict[int, np.ndarray],
    type_labels: List[str],
    *,
    out_dir: str | os.PathLike = "simulation_data",
    filename_prefix: str = "goldilocks_differentiation",
) -> Dict[str, str]:
    """Save transitions, assignments and labels to *out_dir*."""

    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    paths["transitions"] = os.path.join(out_dir, f"{filename_prefix}_transitions.npz")
    np.savez(paths["transitions"], **transition_maps)

    paths["assignments"] = os.path.join(out_dir, f"{filename_prefix}_assignments.npz")
    np.savez(paths["assignments"], **{f"t{t}": a for t, a in cell_type_assignments.items()})

    paths["labels"] = os.path.join(out_dir, f"{filename_prefix}_labels.npy")
    np.save(paths["labels"], np.asarray(type_labels, dtype=object))

    print("Saved differentiation data →", out_dir)
    return paths



