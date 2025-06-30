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

RGBA = Tuple[float, float, float, float]
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
    Ts: List[np.ndarray],
    index_lists: List[List[int]],
    *,
    ind_to_str_dict: Optional[Dict[int, str]] = None,
    str_to_color_dict: Optional[Dict[str, RGBA]] = None,
    row_stochastic: bool = False,
    deg_threshold: float = 0.0,
    dotsize_factor: float = 1.0,
    linethick_factor: float = 10.0,
    title: Optional[str] = None,
    save_directory: Optional[str] = None,
    save_name: Optional[str] = None,
    stretch: float = 1.0,
    outline: float = 3.0,
    fontsize: int = 12,
) -> None:
    """Draw nodes per slice and arrows according to *transition_list*."""
    sns.set(style="white")
    N = len(population_list)

    # normalise rows if requested
    if row_stochastic:
        Ts = [T / np.where(T.sum(1, keepdims=True) == 0, 1, T.sum(1, keepdims=True)) for T in Ts]

    x_pos = [np.full(len(pop), i) for i, pop in enumerate(population_list)]
    y_pos = [np.arange(len(pop)) for pop in population_list]

    plt.figure(figsize=(stretch * 5 * (N - 1), 10))

    for t, T in enumerate(Ts):
        # current slice ↓
        ind_list_cur = index_lists[t]
        label_list_cur = [ind_to_str_dict[i] for i in ind_list_cur]
        pops_cur = np.asarray(population_list[t])
        keep_cur = pops_cur >= deg_threshold
        x_cur = x_pos[t][keep_cur]
        y_cur = np.arange(keep_cur.sum())
        lbl_cur = [label_list_cur[i] for i in range(len(label_list_cur)) if keep_cur[i]]
        sizes_cur = dotsize_factor * pops_cur[keep_cur]
        plt.scatter(x_cur, y_cur, c=[str_to_color_dict[l] for l in lbl_cur], s=sizes_cur, edgecolor="black", lw=1, zorder=1)
        row2y = {r: y_cur[i] for i, r in enumerate(np.where(keep_cur)[0])}

        # next slice ↓
        ind_list_next = index_lists[t + 1]
        label_list_next = [ind_to_str_dict[i] for i in ind_list_next]
        pops_next = np.asarray(population_list[t + 1])
        keep_next = pops_next >= deg_threshold
        x_next = x_pos[t + 1][keep_next]
        y_next = np.arange(keep_next.sum())
        lbl_next = [label_list_next[i] for i in range(len(label_list_next)) if keep_next[i]]
        plt.scatter(x_next, y_next, c=[str_to_color_dict[l] for l in lbl_next], s=dotsize_factor * pops_next[keep_next], edgecolor="black", lw=1, zorder=1)
        col2y = {c: y_next[j] for j, c in enumerate(np.where(keep_next)[0])}

        # arrows
        r1, r2 = T.shape
        for i_row in range(r1):
            if i_row not in row2y:
                continue
            for j_col in range(r2):
                if j_col not in col2y or T[i_row, j_col] == 0:
                    continue
                plt.plot([x_cur[0], x_next[0]], [row2y[i_row], col2y[j_col]], "k-", lw=T[i_row, j_col] * linethick_factor, zorder=0)

    labels_list = [ [ind_to_str_dict[i] for i in ind_list] for ind_list in index_lists]
    # optional text labels
    # if cell_type_labels is not None:
    for s in range(N):
        for j, txt_label in enumerate(labels_list[s]):
            txt = plt.text(x_pos[s][j], y_pos[s][j], txt_label, fontsize=fontsize, ha="right", va="bottom")
            txt.set_path_effects([path_effects.Stroke(linewidth=outline, foreground="white"), path_effects.Normal()])

    plt.axis("off")
    if title:
        plt.title(title, fontsize=36)
    if save_name:
        plt.savefig(save_directory + save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor="white")
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
#  Wrappers that derive clusters then call the above
# ────────────────────────────────────────────────────────────────────────────

def diffmap_from_QT(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    *,
    global_Qs: bool = False,
    clustering_type: str = "ml",
    ind_to_str_dict: Optional[dict] = None,
    str_to_color_dict: Optional[dict] = None,
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_directory: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    stretch: float = 1.0,
    outline: float = 2.0,
    fontsize: int = 12,
    linethick_factor: int = 10,
    full_P: bool = True,
) -> None:
    _, pop_list, index_lists, cdict = get_diffmap_inputs(Qs=Qs,
                                                        Ts=Ts, 
                                                        clustering_type=clustering_type,
                                                        reference_index=reference_index,
                                                        global_Qs=global_Qs,
                                                        full_P=full_P)

    if str_to_color_dict is None:
        str_to_color_dict = cdict

    plot_labeled_differentiation(
        population_list=pop_list,
        Ts=Ts,
        index_lists=index_lists,
        ind_to_str_dict=ind_to_str_dict,
        str_to_color_dict=str_to_color_dict,
        dotsize_factor=dotsize,
        linethick_factor=linethick_factor,
        title=title,
        save_directory=save_directory,
        save_name=save_name,
        stretch=stretch,
        outline=outline,
        fontsize=fontsize,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Legacy barycenter arrow plots (no label‑shift)
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