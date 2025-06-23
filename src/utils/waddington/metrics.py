# utils/waddington/metrics.py
"""
Clustering-quality and co-cluster evaluation utilities
======================================================

Public functions
----------------
* ``compute_clustering_metrics`` – AMI / ARI per time-point
* ``compute_centroids``          – mass-weighted centroid helper
* ``evaluate_coclusters``        – compare HM-OT vs. W-OT predictions
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
except ImportError:
    torch = None  # type: ignore

__all__ = [
    "compute_clustering_metrics",
    "compute_centroids",
    "evaluate_coclusters",
]

# --------------------------------------------------------------------------- #
# 1) basic clustering metrics                                                 #
# --------------------------------------------------------------------------- #
def compute_clustering_metrics(
    true_Q_matrices: Sequence[np.ndarray | "torch.Tensor"],
    pred_Q_matrices: Sequence[np.ndarray | "torch.Tensor"],
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    Return AMI and ARI for each time-point (length = len(true_Q_matrices)).
    """
    if len(true_Q_matrices) != len(pred_Q_matrices):
        raise ValueError("Number of time-points must match between true & predicted")

    ami, ari = [], []

    for t, (Q_true, Q_pred) in enumerate(zip(true_Q_matrices, pred_Q_matrices)):
        if torch and torch.is_tensor(Q_true):
            Q_true = Q_true.cpu().detach().numpy()
        if torch and torch.is_tensor(Q_pred):
            Q_pred = Q_pred.cpu().detach().numpy()

        y_true = np.argmax(Q_true, axis=1)
        y_pred = np.argmax(Q_pred, axis=1)

        ami_t = adjusted_mutual_info_score(y_true, y_pred)
        ari_t = adjusted_rand_score(y_true, y_pred)
        ami.append(ami_t)
        ari.append(ari_t)

        if verbose:
            print(
                f"t={t:<3}  AMI={ami_t:6.4f}  ARI={ari_t:6.4f}  "
                f"N={len(y_true):5d}  clusters(true/pred)="
                f"{len(np.unique(y_true))}/{len(np.unique(y_pred))}"
            )

    return ami, ari


# --------------------------------------------------------------------------- #
# 2) centroid helper                                                          #
# --------------------------------------------------------------------------- #
def compute_centroids(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """g(Q)⁻¹ · Qᵀ · X   (rows = clusters, cols = spatial dims)."""
    gQ = Q.sum(axis=0)                 # (K,)
    return (Q.T @ X) / gQ[:, None]     # broadcasting divides each row


# --------------------------------------------------------------------------- #
# 3) co-cluster evaluation (HM-OT vs. W-OT)                                   #
# --------------------------------------------------------------------------- #
def evaluate_coclusters(
    Qs_kmeans: Sequence[np.ndarray],
    Qs_hmot: Sequence[np.ndarray],
    Ts_wot: Tuple[np.ndarray, np.ndarray],
    Ts_hmot: Tuple[np.ndarray, np.ndarray],
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """
    Return two 2-element lists with **weighted cosine-similarity scores**
    for the 1→2 and 2→3 transitions, comparing predictions from

    * HM-OT (hierarchical mediation OT)
    * W-OT  (Waddington OT paper baseline)
    """
    # unpack
    Q1_k, Q2_k, Q3_k = Qs_kmeans
    Q1_h, Q2_h, Q3_h = Qs_hmot
    T12_w, T23_w = Ts_wot
    T12_h, T23_h = Ts_hmot

    # centroids
    C1_k, C2_k, C3_k = map(compute_centroids, (X1, X2, X3), (Q1_k, Q2_k, Q3_k))
    C1_h, C2_h, C3_h = map(compute_centroids, (X1, X2, X3), (Q1_h, Q2_h, Q3_h))

    # predictions via transport maps
    C2_pred_h = (T12_h.T @ C1_h) / T12_h.sum(axis=0)[:, None]
    C2_pred_w = (T12_w.T @ C1_k) / T12_w.sum(axis=0)[:, None]
    C3_pred_h = (T23_h.T @ C2_h) / T23_h.sum(axis=0)[:, None]
    C3_pred_w = (T23_w.T @ C2_k) / T23_w.sum(axis=0)[:, None]

    # weighted cos-sim
    w_h12 = (Q2_h.sum(axis=0) * cosine_similarity(C2_h,  C2_pred_h).diagonal()).sum()
    w_h23 = (Q3_h.sum(axis=0) * cosine_similarity(C3_h,  C3_pred_h).diagonal()).sum()
    w_w12 = (Q2_k.sum(axis=0) * cosine_similarity(C2_k,  C2_pred_w).diagonal()).sum()
    w_w23 = (Q3_k.sum(axis=0) * cosine_similarity(C3_k,  C3_pred_w).diagonal()).sum()

    return [w_h12, w_h23], [w_w12, w_w23]