from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.platting.palette_utils import get_diffmap_inputs

__all__ = [
    "plot_clustering_list",
    "plot_clusters_from_QT",
]

def plot_clustering_list(
    Ss: List[np.ndarray],
    clustering_list: List[np.ndarray], # list of raw labels 
    *,
    ind_to_str_dict: Optional[dict] = None,
    str_to_color_dict: Optional[dict] = None,
    title: Optional[str] = None,
    save_directory: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    key_dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[str]]] = None,
) -> None:
    """One subplot *per* slice with consistent color scheme."""
    n_slices = len(Ss)

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5)

    fig, axes = plt.subplots(1, n_slices, figsize=(20 * n_slices, 20), facecolor="white")
    axes = np.asarray(axes).reshape(-1)  # ensures iterable even if only one slice

    Ss_ = [S - np.mean(S, axis=0) for S in Ss] # center coordinates

    all_pts = np.vstack(Ss_)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    for i, (coords, labels) in enumerate(zip(Ss_, clustering_list)):
        if ind_to_str_dict is None:
            ind_to_str_dict = {int(i): str(int(i)) for i in set(list(labels))}
        else:
            pass
        lbls = [ind_to_str_dict[int(i)] for i in labels]
        
        ax = axes[i]
        if flip:
            coords = coords @ np.array([[-1, 0], [0, 1]])
        df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": lbls})
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="value",
            palette=str_to_color_dict,
            s=dotsize,
            legend="brief",
            ax=ax,
        )
        handles, labels_leg = ax.get_legend_handles_labels()
        for h in handles:
            h.set_markersize(50 * key_dotsize)

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

        """if subplot_labels is not None:
            ax.set_title(subplot_labels[i] or "", color="black")
        else:
            pass
            ax.set_title(f"Slice {i + 1}", color="black")"""

    if title:
        plt.suptitle(title, fontsize=36, color="black")
    if save_name:
        plt.savefig(save_directory + save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor="white")
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
    global_Qs: bool = False,
    clustering_type: str = "ml",
    ind_to_str_dict: Optional[dict] = None, #NOTE: crucial change for flexibility: this is now a dict
    str_to_color_dict: Optional[dict] = None,
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_directory: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    key_dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[str]]] = None,
    full_P: bool = True, 
) -> None:
    """
    Args:
        * Ss: List[np.ndarray],
            List of spatial (or synthetic expression) coordinates for each timepoint.
        * Qs: List[np.ndarray],
            List of latent representation matrices (coupling spots to cell types), 
            at each timepoint.
        * Ts: List[np.ndarray],
            List of transition matrices (coupling cell types across consecutive timepoints).
        * global_Qs: bool,
            If True, Qs are treated as global population matrices,
            meaning all timepoints share all cell types,
            otherwise they are treated as local population matrices.
        * clustering_type: str,
            "ml" for max likelihood (ARGMAX of columns of Qs), or
            "reference" for reference co-clustering.
        * reference_index: Optional[int],
            If clustering_type is "reference",
            this is the index of the reference timepoint.
        * ind_to_str_dict: Optional[dict],
            Dictionary mapping cluster labels to cell type names.
            If None, default numeric labels will be used.
            This is only used if clustering_type is "ml" or "reference".
        * str_to_color_dict: Optional[dict],
            Dictionary mapping cluster labels to colors.
            If None, a default color palette will be used.
        * title: Optional[str],
            Title for the plot.
        * save_directory: Optional[str],
            Directory to save the plot.
        * save_name: Optional[str],
            Name of the saved plot file.
        * dotsize: float,
            Size of the dots in the scatter plot.
        * key_dotsize: float,
            Size of the legend dots.
        * flip: bool,
            If True, flips the y-axis of the plot.
        * subplot_labels: Optional[List[Optional[str]]],
            List of labels for each subplot.
            If None, default labels will be used.
        * full_P: bool,
            If True, uses the full transition matrix P for clustering.
            If False, uses low-memory reference clustering, slower
    """
    
    raw_labels_list, _, _, cdict = get_diffmap_inputs(
        Qs=Qs,
        Ts=Ts,
        clustering_type=clustering_type,
        reference_index=reference_index,
        global_Qs=global_Qs,
        full_P=full_P,
    )

    if str_to_color_dict is None:
        str_to_color_dict = cdict
    else:
        pass

    plot_clustering_list(
        Ss=Ss,
        clustering_list=raw_labels_list,
        ind_to_str_dict=ind_to_str_dict,
        str_to_color_dict=str_to_color_dict,
        title=title,
        save_directory=save_directory,
        save_name=save_name,
        dotsize=dotsize,
        key_dotsize=key_dotsize,
        flip=flip,
        subplot_labels=subplot_labels,
    )