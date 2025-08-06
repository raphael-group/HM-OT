import numpy as np
import torch
from anndata import AnnData

# -------- utilities --------

def coarsen_with_Q(X: torch.Tensor, Q: torch.Tensor):
    """
    X: (n_i, 2) float tensor (coords)
    Q: (n_i, r_i) nonnegative soft/hard assignments, columns sum to masses
    returns:
        X_bar: (r_i, 2) barycenters
        g:     (r_i,) masses (column sums of Q)
    """
    g = Q.sum(dim=0)                       # (r_i,)
    # avoid divide-by-zero in empty clusters
    g_safe = torch.clamp(g, min=1e-12)
    # Take LC-projection ( see Definition 4.1 in https://openreview.net/pdf?id=hGgkdFF2hR )
    X_bar = torch.diag( 1 / g_safe) @ (Q.T @ X)    # r_i x 2 low-rank'd spatial coordinates
    return X_bar, g

@torch.no_grad()
def weighted_procrustes(S: torch.Tensor, T: torch.Tensor,
                        Pi: torch.Tensor, partial: bool=False):
    """
    Weighted Procrustes using coupling Pi between rows of S and rows of T.
    S:  (r_s, 2)
    T:  (r_t, 2)
    Pi: (r_s, r_t) nonnegative weights (not necessarily normalized)
    returns:
        R:  (2,2) rotation (det=+1)
        t:  (2,)  translation such that  S ≈ (T @ R.T) + t
        mu_s, mu_t: weighted centroids (2,), (2,)
    """
    device = S.device
    one_s = torch.ones(S.shape[0], device=device)
    one_t = torch.ones(T.shape[0], device=device)

    a = Pi @ one_t           # (r_s,)
    b = Pi.T @ one_s         # (r_t,)
    m = Pi.sum()             # total mass

    if partial:
        m = torch.clamp(m, min=1e-12)
        mu_s = (S.T @ a) / m           # (2,)
        mu_t = (T.T @ b) / m
    else:
        # if Pi already has marginals summing to 1, this is the same
        m = torch.clamp(m, min=1e-12)
        mu_s = (S.T @ a) / m
        mu_t = (T.T @ b) / m

    S0 = S - mu_s[None, :]
    T0 = T - mu_t[None, :]

    # cross-covariance (2x2)
    C = T0.T @ Pi.T @ S0 / m

    # SVD for rotation
    U, _, Vt = torch.linalg.svd(C, full_matrices=False)
    R = Vt.T @ U.T
    # enforce det(R)=+1 (no reflection)
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # translation so that (T @ R.T) + t ≈ S
    t = mu_s - (R @ mu_t)
    return R, t, mu_s, mu_t

def apply_transform_full(X: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mu_t: torch.Tensor):
    """
    Apply target->source rigid transform to all points X (n x 2):
      X' = (X - mu_t) R^T + mu_s
    We pass t, but use mu_t (and t) consistently via: X' = X @ R.T + t
    Both forms are equivalent if t = mu_s - R mu_t.
    """
    return (X @ R.T) + t[None, :]

# -------- main adapters (pairwise and center) --------

def stack_slices_pairwise_lowrank(
    slices: list[AnnData],
    Qs: list[torch.Tensor],   # each (n_i, r_i)
    Ts: list[torch.Tensor],   # each (r_i, r_{i+1}) transition/coupling
    is_partial: bool = False,
    dtype=torch.float32,
    device='cpu'
):
    """
    Pairwise chain: slice0 -> slice1 -> ... using low-rank (Q,T).
    Returns aligned slices + per-edge (R,t, mus, muts).
    """
    assert len(slices) == len(Qs) == (len(Ts) + 1)

    # coarsen all slices once
    Xbars, masses = [], []
    for i, sl in enumerate(slices):
        Xi = torch.tensor(sl.obsm["spatial"], dtype=dtype, device=device)
        Qi = Qs[i].to(device=device, dtype=dtype)
        Xbar, g = coarsen_with_Q(Xi, Qi)
        Xbars.append(Xbar)
        masses.append(g)

    transforms = []   # list of dicts with R,t,mu_s,mu_t per edge
    aligned_coords = [torch.tensor(slices[0].obsm["spatial"], dtype=dtype, device=device).clone()]

    for i in range(len(Ts)):
        # low-dim Procrustes from centroids and T_i
        Pi = Ts[i].to(device=device, dtype=dtype)
        R, t, mu_s, mu_t = weighted_procrustes(Xbars[i], Xbars[i+1], Pi, partial=is_partial)

        # apply to FULL target coordinates of slice i+1
        X_next = torch.tensor(slices[i+1].obsm["spatial"], dtype=dtype, device=device)
        X_next_aligned = apply_transform_full(X_next, R, t, mu_t)
        aligned_coords.append(X_next_aligned)

        transforms.append(dict(R=R, t=t, mu_s=mu_s, mu_t=mu_t))

    # build new AnnData list
    new_slices = []
    for i, sl in enumerate(slices):
        _sl = sl.copy()
        _sl.obsm["spatial"] = aligned_coords[i].cpu().numpy()
        new_slices.append(_sl)

    return new_slices, transforms

def stack_slices_center_lowrank(
    center_slice: AnnData,
    slices: list[AnnData],
    Q_center: torch.Tensor,
    Qs: list[torch.Tensor],
    Ts_to_center: list[torch.Tensor],   # each (r_center, r_i) or (r_i, r_center); see below
    to_center: bool = True,             # True if Ts map center->slice; False if slice->center
    dtype=torch.float32,
    device='cpu'
):
    """
    Align every slice to a fixed center slice via low-rank Q/T.
    """
    # coarsen center and others
    Xc = torch.tensor(center_slice.obsm["spatial"], dtype=dtype, device=device)
    Xc_bar, gc = coarsen_with_Q(Xc, Q_center.to(device=device, dtype=dtype))

    new_slices = []
    transforms = []

    for i, sl in enumerate(slices):
        Xi = torch.tensor(sl.obsm["spatial"], dtype=dtype, device=device)
        Xi_bar, gi = coarsen_with_Q(Xi, Qs[i].to(device=device, dtype=dtype))
        Ti = Ts_to_center[i].to(device=device, dtype=dtype)

        if to_center:
            # Pi couples center (rows) to slice i (cols): shape rc x ri
            S, T, Pi = Xc_bar, Xi_bar, Ti
        else:
            # Pi couples slice i (rows) to center (cols)
            S, T, Pi = Xi_bar, Xc_bar, Ti

        R, t, mu_s, mu_t = weighted_procrustes(S, T, Pi, partial=False)
        # If S is center and T is slice, we want to map slice -> center:
        # X_i_aligned = X_i @ R^T + t
        X_aligned = apply_transform_full(Xi, R, t, mu_t)
        _sl = sl.copy()
        _sl.obsm["spatial"] = X_aligned.cpu().numpy()
        new_slices.append(_sl)
        transforms.append(dict(R=R, t=t, mu_s=mu_s, mu_t=mu_t))

    new_center = center_slice.copy()
    new_center.obsm["spatial"] = Xc.cpu().numpy()  # unchanged
    return new_center, new_slices, transforms