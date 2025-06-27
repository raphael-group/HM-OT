from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.platting.palette_utils import get_diffmap_inputs
from src.utils.clustering import (
    max_likelihood_clustering,
    reference_clustering,
    reference_clustering_prime,
)

__all__ = [
    "plot_all_sc_clusters",
    "plot_clustering_list",
    "plot_clusters_from_QT",
    "plot_all_sc_from_QT",
]

# ────────────────────────────────────────────────────────────────────────────
#  Core scatter helpers (Matplotlib / Seaborn)
# ────────────────────────────────────────────────────────────────────────────

def plot_all_sc_clusters(
    spatial_list: List[np.ndarray],
    clustering_list: List[np.ndarray],
    *,
    clustering_type: str = "ml",
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[List[str]]]] = None,
) -> None:
    """Overlay **all** slices on a single axis.

    Parameters
    ----------
    spatial_list
        List of (n_t × 2) arrays of spatial coordinates.
    clustering_list
        Matching list of hard cluster labels for each slice.
    clustering_type
        "ml" (default) → shift labels so they are unique over time;
        "reference" → leave labels as‑is.
    cell_type_labels
        Optional text labels per slice.  If supplied, a legend is drawn –
        but note that on a single axis this can get crowded quickly.
    dotsize
        Marker size passed to seaborn.
    flip
        Mirror x‑axis (useful if original spatial coords have inverted axis).
    """
    n_slices = len(spatial_list)
    cell_type_labels = cell_type_labels or [None] * n_slices

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(20 * n_slices, 20), facecolor="white")

    # recentre each slice around its mean for nicer overlay
    centred = [S - np.mean(S, axis=0) for S in spatial_list]
    _, _, color_dict = get_diffmap_inputs(clustering_list, clustering_type)

    # common axis limits for neat aspect ratio
    all_pts = np.vstack(centred)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    cumulative_shift = 0  # only used for ML style
    for coords, labels in zip(centred, clustering_list):
        lbls = labels + cumulative_shift if clustering_type == "ml" else labels
        if flip:
            coords = coords @ np.array([[-1, 0], [0, 1]])
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": lbls})
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="value",
            palette=color_dict,
            s=dotsize,
            legend=False,
            ax=ax,
        )
        if clustering_type == "ml":
            cumulative_shift += len(np.unique(labels))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if flip:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=36, color="black")
    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor="white")
    plt.tight_layout()
    plt.show()


def plot_clustering_list(
    spatial_list: List[np.ndarray],
    clustering_list: List[np.ndarray],
    *,
    clustering_type: str = "ml",
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    key_dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[str]]] = None,
    color_dict: Optional[dict] = None,
    global_Qs: bool = False,
) -> None:
    """One subplot *per* slice with consistent color scheme."""
    n_slices = len(spatial_list)
    if cell_type_labels is None:
        cell_type_labels = [None] * n_slices
    # cell_type_labels = cell_type_labels or [None] * n_slices

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5)

    fig, axes = plt.subplots(1, n_slices, figsize=(20 * n_slices, 20), facecolor="white")
    axes = np.asarray(axes).reshape(-1)  # ensures iterable even if only one slice

    centred = [S - np.mean(S, axis=0) for S in spatial_list]
    #. _, _, color_dict = get_diffmap_inputs(clustering_list, clustering_type)

    all_pts = np.vstack(centred)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    cumulative_shift = 0
    for i, (coords, labels) in enumerate(zip(centred, clustering_list)):
        ax = axes[i]
        if clustering_type == "ml" and global_Qs==False:
            lbls = labels + cumulative_shift
        else:
            lbls = labels 
        # lbls = labels + cumulative_shift if clustering_type == "ml" else labels
        if flip:
            coords = coords @ np.array([[-1, 0], [0, 1]])
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": lbls})
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="value",
            palette=color_dict,
            s=dotsize,
            legend="brief",
            ax=ax,
        )
        handles, labels_leg = ax.get_legend_handles_labels()
        for h in handles:
            h.set_markersize(50 * key_dotsize)

        if cell_type_labels[i] is not None:
            # map numeric → custom label
            mapping = {
                n: cell_type_labels[i][j]
                for j, n in enumerate(sorted(np.unique(lbls)))
                if j < len(cell_type_labels[i])
            }
            new_lbls = [mapping.get(int(float(l)), f"Cluster {l}") for l in labels_leg]
            ax.legend(
                handles=handles,
                labels=new_lbls,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
                labelspacing=4 * key_dotsize,
                fontsize=20,
            )
        else:
            ax.legend(
                handles=handles,
                labels=labels_leg,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
                labelspacing=4 * key_dotsize,
                fontsize=20,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if flip:
            ax.invert_yaxis()
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        if subplot_labels is not None:
            ax.set_title(subplot_labels[i] or "", color="black")
        else:
            ax.set_title(f"Slice {i + 1}", color="black")

        if clustering_type == "ml":
            cumulative_shift += len(np.unique(labels))

    if title:
        plt.suptitle(title, fontsize=36, color="black")
    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor="white")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
#  Convenience wrappers that derive clusters from (Qs, Ts) then plot
# ────────────────────────────────────────────────────────────────────────────

def plot_clusters_from_QT(
    Ss: List[np.ndarray],
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    *,
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    key_dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[str]]] = None,
    full_P: bool = True,
    color_dict: Optional[dict] = None,
    global_Qs: bool = False,
) -> None:
    
    raw_labels_list, _, _, color_dict = get_diffmap_inputs(
        Qs=Qs,
        Ts=Ts,
        clustering_type=clustering_type,
        reference_index=reference_index,
        global_Qs=global_Qs,
    )

    plot_clustering_list(
        spatial_list=Ss,
        clustering_list=raw_labels_list,
        clustering_type=clustering_type,
        cell_type_labels=cell_type_labels,
        title=title,
        save_name=save_name,
        dotsize=dotsize,
        key_dotsize=key_dotsize,
        flip=flip,
        subplot_labels=subplot_labels,
        color_dict=color_dict,
        global_Qs=global_Qs,
    )


def plot_all_sc_from_QT(
    Ss: List[np.ndarray],
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    *,
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[str]]] = None,
) -> None:
    """Derive clusters then overlay all slices (single axis)."""
    if clustering_type == "ml":
        clustering_list = max_likelihood_clustering(Qs)
    elif clustering_type == "reference":
        if reference_index is None:
            raise ValueError("Reference index required for reference clustering.")
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError(f"Invalid clustering_type '{clustering_type}'.")

    plot_all_sc_clusters(
        spatial_list=Ss,
        clustering_list=clustering_list,
        clustering_type=clustering_type,
        cell_type_labels=cell_type_labels,
        title=title,
        save_name=save_name,
        dotsize=dotsize,
        flip=flip,
        subplot_labels=subplot_labels,
    )
