from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patheffects as path_effects

from src.platting.palette_utils import get_diffmap_inputs
from src.utils.clustering import (
    max_likelihood_clustering,
    reference_clustering,
    reference_clustering_prime,
)

__all__ = [
    "plot_labeled_differentiation",
    "diffmap_from_QT",
    "plot_diffmap_clusters",
    "plot_diffmap_clusters_prime",
]

# ────────────────────────────────────────────────────────────────────────────
#  Helper
# ────────────────────────────────────────────────────────────────────────────
def _centres_to_array(centres_t: Dict[int, np.ndarray], local_ids: np.ndarray) -> np.ndarray:
    """Convert ``{global_id: (x,y)}`` to a (k×2) array aligned with *local_ids*."""
    global_sorted = sorted(centres_t.keys())
    return np.vstack([centres_t[global_sorted[i]] for i in local_ids])


# ────────────────────────────────────────────────────────────────────────────
#  Core differentiation‑map scatter + arrows
# ────────────────────────────────────────────────────────────────────────────
def plot_labeled_differentiation(
    population_list: List[List[int]],
    transition_list: List[np.ndarray],
    label_list: List[List[int]],
    color_dict: Dict[int, Tuple[float, float, float, float]],
    *,
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    row_stochastic: bool = False,
    deg_threshold: float = 0.0,
    dotsize_factor: float = 1.0,
    linethick_factor: float = 10.0,
    save_name: Optional[str] = None,
    title: Optional[str] = None,
    stretch: float = 1.0,
    outline: float = 3.0,
    fontsize: int = 12,
) -> None:
    # TODO: missing save directory argument, docstring 
    """Draw nodes per slice and arrows according to *transition_list*."""
    sns.set(style="white")
    N = len(population_list)
    Ts = transition_list

    # normalise rows if requested
    if row_stochastic:
        Ts = [T / np.where(T.sum(1, keepdims=True) == 0, 1, T.sum(1, keepdims=True)) for T in Ts]

    x_pos = [np.full(len(pop), i) for i, pop in enumerate(population_list)]
    y_pos = [np.arange(len(pop)) for pop in population_list]

    plt.figure(figsize=(stretch * 5 * (N - 1), 10))

    for t, T in enumerate(Ts):
        # current slice ↓
        pops_cur = np.asarray(population_list[t])
        keep_cur = pops_cur >= deg_threshold
        x_cur = x_pos[t][keep_cur]
        y_cur = np.arange(keep_cur.sum())
        lbl_cur = np.asarray(label_list[t])[keep_cur]
        sizes_cur = dotsize_factor * pops_cur[keep_cur]
        plt.scatter(x_cur, y_cur, c=[color_dict[l] for l in lbl_cur], s=sizes_cur, edgecolor="b", lw=1, zorder=1)
        row2y = {r: y_cur[i] for i, r in enumerate(np.where(keep_cur)[0])}

        # next slice ↓
        pops_next = np.asarray(population_list[t + 1])
        keep_next = pops_next >= deg_threshold
        x_next = x_pos[t + 1][keep_next]
        y_next = np.arange(keep_next.sum())
        lbl_next = np.asarray(label_list[t + 1])[keep_next]
        plt.scatter(x_next, y_next, c=[color_dict[l] for l in lbl_next], s=dotsize_factor * pops_next[keep_next], edgecolor="b", lw=1, zorder=1)
        col2y = {c: y_next[j] for j, c in enumerate(np.where(keep_next)[0])}

        # arrows
        if clustering_type == "ml":
            r1, r2 = T.shape
            for i_row in range(r1):
                if i_row not in row2y:
                    continue
                for j_col in range(r2):
                    if j_col not in col2y or T[i_row, j_col] == 0:
                        continue
                    plt.plot([x_cur[0], x_next[0]], [row2y[i_row], col2y[j_col]], "k-", lw=T[i_row, j_col] * linethick_factor, zorder=0)

    # optional text labels
    if cell_type_labels is not None:
        for s in range(N):
            if cell_type_labels[s] is None:
                continue
            for j, txt_label in enumerate(cell_type_labels[s]):
                txt = plt.text(x_pos[s][j], y_pos[s][j], txt_label, fontsize=fontsize, ha="right", va="bottom")
                txt.set_path_effects([path_effects.Stroke(linewidth=outline, foreground="white"), path_effects.Normal()])

    plt.axis("off")
    if title:
        plt.title(title, fontsize=36)
    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor="white")
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
#  Wrappers that derive clusters then call the above
# ────────────────────────────────────────────────────────────────────────────

def diffmap_from_QT(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    *,
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dsf: float = 1.0,
    stretch: float = 1.0,
    outline: float = 2.0,
    fontsize: int = 12,
    linethick_factor: int = 10,
    global_Qs: bool = False,
) -> None:

    _, pop_list, index_list, color_dict = get_diffmap_inputs(Qs=Qs,
                                                        Ts=Ts, 
                                                        clustering_type=clustering_type,
                                                        reference_index=reference_index,
                                                        global_Qs=global_Qs)

    plot_labeled_differentiation(
        population_list=pop_list,
        transition_list=Ts,
        label_list=index_list,
        color_dict=color_dict,
        cell_type_labels=cell_type_labels,
        clustering_type=clustering_type,
        dotsize_factor=dsf,
        linethick_factor=linethick_factor,
        title=title,
        save_name=save_name,
        stretch=stretch,
        outline=outline,
        fontsize=fontsize,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Legacy barycentre arrow plots (no label‑shift)
# ────────────────────────────────────────────────────────────────────────────

def plot_diffmap_clusters(
    X: np.ndarray,
    time_labels: np.ndarray,
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    df: "pd.DataFrame",
    *,
    cluster_key: str = "cluster_pred",
    mode: str = "standard",
    min_thresh: float = 1e-5,
    max_lw: float = 8.0,
    figsize: Tuple[int, int] = (16, 16),
):
    import pandas as pd
    labels_list = max_likelihood_clustering(Qs, mode=mode)
    labels_list = [np.asarray(l) for l in labels_list]
    offset, labels_flat = 0, []
    for l in labels_list:
        labels_flat.append(l + offset)
        offset += l.max() + 1
    df[cluster_key] = np.concatenate(labels_flat)

    _, _, color_dict = get_diffmap_inputs(labels_list, "ml")

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x="x", y="y", hue=cluster_key, palette=color_dict, s=60, lw=0.3, edgecolor="k", ax=ax)
    ax.set_aspect("equal", adjustable="box")
    sns.despine()

    for t, T in enumerate(Ts):
        X_t, X_tp1 = X[time_labels == t], X[time_labels == t + 1]
        Q_src, Q_tgt = Qs[t], Qs[t + 1]
        bary_src = (Q_src.T @ X_t) / Q_src.sum(0)[:, None]
        bary_tgt = (Q_tgt.T @ X_tp1) / Q_tgt.sum(0)[:, None]
        norm = T.max() or 1.0
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                w = T[i, j]
                if w <= min_thresh:
                    continue
                ax.annotate("", xy=bary_tgt[j], xytext=bary_src[i], arrowprops=dict(arrowstyle="->", lw=(w / norm) * max_lw, alpha=0.7, color="gray"))
    plt.tight_layout()
    return fig, ax


def plot_diffmap_clusters_prime(
    X: np.ndarray,
    time_labels: np.ndarray,
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    df: "pd.DataFrame",
    *,
    cluster_key: str = "cluster_pred",
    mode: str = "standard",
    min_thresh: float = 1e-5,
    max_lw: float = 8.0,
    figsize: Tuple[int, int] = (16, 16),
    centres: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
):
    import pandas as pd
    labels_list = max_likelihood_clustering(Qs, mode=mode)
    labels_list = [np.asarray(l) for l in labels_list]

    offset, labels_flat = 0, []
    for lbl in labels_list:
        labels_flat.append(lbl + offset)
        offset += lbl.max() + 1
    df[cluster_key] = np.concatenate(labels_flat)

    _, _, color_dict = get_diffmap_inputs(labels_list, "ml")

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x="x", y="y", hue=cluster_key, palette=color_dict, s=60, lw=0.3, edgecolor="k", alpha=0.5, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    sns.despine()

    for t, T in enumerate(Ts):
        src_ids = np.unique(labels_list[t])
        tgt_ids = np.unique(labels_list[t + 1])

        if centres is not None:
            bary_src = _centres_to_array(centres[t], src_ids)
            bary_tgt = _centres_to_array(centres[t + 1], tgt_ids)
        else:
            X_t, X_tp1 = X[time_labels == t], X[time_labels == t + 1]
            Q_src, Q_tgt = Qs[t], Qs[t + 1]
            bary_src = (Q_src.T @ X_t) / Q_src.sum(0)[:, None]
            bary_tgt = (Q_tgt.T @ X_tp1) / Q_tgt.sum(0)[:, None]

        norm = T.max() or 1.0
        for i in range(len(src_ids)):
            for j in range(len(tgt_ids)):
                w = T[i, j]
                if w <= min_thresh:
                    continue
                ax.annotate("", xy=bary_tgt[j], xytext=bary_src[i], arrowprops=dict(arrowstyle="-|>", lw=(w / norm) * max_lw, alpha=0.7, color="black"))

    plt.tight_layout()
    return fig, ax
