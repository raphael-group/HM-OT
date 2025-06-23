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
from utils.waddington.minima import classify_minima_by_ring

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
# 1) Cell-type assignment ------------------------------------------------------
# -----------------------------------------------------------------------------

def assign_cell_types(
    positions: np.ndarray,  # shape (N, 2)
    minima_points: Sequence[Tuple[float, float]],
    *,
    assignment_radius: float = 1.4,
) -> Tuple[np.ndarray, List[Tuple[float, float, str]]]:
    """Nearest-minimum assignment with an *unassigned* fallback (−1)."""

    n = positions.shape[0]
    cell_types = np.full(n, -1, dtype=int)
    classified_minima = classify_minima_by_ring(minima_points)

    for p_idx, (x, y) in enumerate(positions):
        best_idx: int | None = None
        best_dist = assignment_radius
        for m_idx, (mx, my, _) in enumerate(classified_minima):
            d = np.hypot(x - mx, y - my)
            if d < best_dist:
                best_dist = d
                best_idx = m_idx
        if best_idx is not None:
            cell_types[p_idx] = best_idx

    return cell_types, classified_minima

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

# -----------------------------------------------------------------------------
# 5) Q-matrices & OT preparation ----------------------------------------------
# -----------------------------------------------------------------------------

def generate_Q_matrices_from_clusters(
    snapshots: dict[str, np.ndarray],
    minima_points: list[tuple[float, float]],
    assignment_radius: float,
    *,
    device: str = "cpu",
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Return a Q matrix per snapshot; last column is *U* (unassigned)."""
    minima = np.asarray(minima_points)
    n_min = minima.shape[0]
    n_types = n_min + 1

    Qs: list[np.ndarray] = []
    assigns: dict[str, np.ndarray] = {}

    for key in sorted(snapshots):
        X = snapshots[key]
        N = X.shape[0]
        dists = np.linalg.norm(X[:, None, :] - minima[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        within = dists <= assignment_radius

        Q = np.zeros((N, n_types), dtype=float)
        lab = np.full(N, -1, dtype=int)
        for i in range(N):
            if within[i, nearest[i]]:
                Q[i, nearest[i]] = 1.0
                lab[i] = nearest[i]
            else:
                Q[i, n_min] = 1.0
        Qs.append(Q)
        assigns[key] = lab
    return Qs, assigns


def setup_point_clouds_for_waddington_ot(
    pc1: np.ndarray,
    pc2: np.ndarray,
    *,
    time_1: int = 0,
    time_2: int = 1,
    ct_labels_1: np.ndarray | None = None,
    ct_labels_2: np.ndarray | None = None,
) -> AnnData:
    """Combine two point clouds into an AnnData suitable for OT pipelines."""
    X = np.vstack([pc1, pc2])
    t_lbls = np.concatenate([np.full(pc1.shape[0], time_1), np.full(pc2.shape[0], time_2)])
    ids = np.arange(X.shape[0])
    adata = AnnData(X=X)
    adata.obs["time_point"] = t_lbls
    adata.obs["cell_id"] = ids
    adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    adata.var_names = ["dim_1", "dim_2"]
    if ct_labels_1 is not None and ct_labels_2 is not None:
        adata.obs["celltype"] = np.concatenate([ct_labels_1, ct_labels_2])
    return adata