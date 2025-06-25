from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Union

import numpy as np
try:
    import plotly.graph_objects as go  # type: ignore
except ImportError:                    # pragma: no cover
    raise ImportError(
        "Plotly is required for Sankey visualisation. "
        "Install it with `pip install plotly`."
    )

# ─── internal imports ─────────────────────────────────────────────────────────
from src.platting.diffmap import get_diffmap_inputs  # colour + population helpers
from src.utils.clustering import (
    max_likelihood_clustering,
    reference_clustering,
)
from src.platting.color_utils import rgba_to_plotly_string
from src.platting.string_utils import alphabetic_key

__all__ = [
    "plot_labeled_differentiation_sankey",
    "plot_labeled_differentiation_sankey_sorted",
    "diffmap_from_QT_sankey",
    "make_sankey",
]

# -----------------------------------------------------------------------------
# Core utilities (kept local so we don't add more tiny helper modules)
# -----------------------------------------------------------------------------

def _validate_threshold(threshold: float) -> float:
    """Ensure a non-negative transition threshold."""
    if threshold < 0:
        raise ValueError("threshold must be ≥ 0.0")
    return threshold


# -----------------------------------------------------------------------------
# 1) Sankey for multi-slice differentiation maps
# -----------------------------------------------------------------------------

def plot_labeled_differentiation_sankey(
    population_list: List[List[int]],
    transition_list: List[np.ndarray],
    label_list: List[List[int]],
    color_dict: Dict[int, Tuple[float, float, float, float]],
    *,
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    linethick_factor: float = 10.0,
    plot_height: int = 600,
    plot_width: int = 1000,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    save_as_svg: bool = True,
    threshold: float = 0.0,
) -> None:
    """Plain left-to-right Sankey diagram without manual ordering."""

    threshold = _validate_threshold(threshold)
    node_labels, node_colors, link_sources, link_targets, link_values = [], [], [], [], []
    node_idx_map: Dict[Tuple[int, int], int] = {}
    cur_idx = 0

    # ─── build nodes ────────────────────────────────────────────────────────
    for slice_idx, slice_labels in enumerate(label_list):
        for j, cluster_label in enumerate(slice_labels):
            label_str = (
                cell_type_labels[slice_idx][j]
                if cell_type_labels and cell_type_labels[slice_idx]
                else str(cluster_label)
            )
            node_labels.append(label_str)
            node_colors.append(rgba_to_plotly_string(color_dict[cluster_label]))
            node_idx_map[(slice_idx, j)] = cur_idx
            cur_idx += 1

    # ─── build links ────────────────────────────────────────────────────────
    for t, T in enumerate(transition_list):
        rows, cols = T.shape
        for i in range(rows):
            for j in range(cols):
                val = T[i, j]
                if val <= threshold:
                    continue
                link_sources.append(node_idx_map[(t, i)])
                link_targets.append(node_idx_map[(t + 1, j)])
                link_values.append(val * linethick_factor)

    # ─── figure ----------------------------------------------------------------
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
            ),
            link=dict(source=link_sources, target=link_targets, value=link_values),
        )
    )
    fig.update_layout(
        title_text=title or "Differentiation Map",
        font_size=24,
        height=plot_height,
        width=plot_width,
    )

    if save_as_svg and save_name:
        fig.write_image(f"{save_name}.svg")
    fig.show()


# -----------------------------------------------------------------------------
# 2) Ordered variant – manual (x, y) placement + alphabetical sorting
# -----------------------------------------------------------------------------

def plot_labeled_differentiation_sankey_sorted(
    population_list: List[List[int]],
    transition_list: List[np.ndarray],
    label_list: List[List[int]],
    color_dict: Dict[int, Tuple[float, float, float, float]],
    *,
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    linethick_factor: float = 10.0,
    plot_height: int = 600,
    plot_width: int = 800,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    save_as_svg: bool = False,
    threshold: float = 0.0,
) -> None:
    """Alphabetically-sorted vertical layout to reduce edge crossings."""

    threshold = _validate_threshold(threshold)
    N = len(population_list)

    node_labels, node_colors, node_x, node_y = [], [], [], []
    link_sources, link_targets, link_values = [], [], []
    node_idx_map: Dict[Tuple[int, int], int] = {}
    cur_idx = 0
    sorted_indices_per_slice: List[List[int]] = []

    # ─── nodes with manual positions ────────────────────────────────────────
    for slice_idx, labels_slice in enumerate(label_list):
        def _order_key(k: int) -> str:
            if cell_type_labels and cell_type_labels[slice_idx]:
                return alphabetic_key(cell_type_labels[slice_idx][k])
            return alphabetic_key(str(labels_slice[k]))

        sorted_idx = sorted(range(len(labels_slice)), key=_order_key)
        sorted_indices_per_slice.append(sorted_idx)

        y_positions = np.linspace(1.0, 0.0, len(sorted_idx)) if len(sorted_idx) > 1 else [0.5]
        for rank, orig_j in enumerate(sorted_idx):
            lbl_val = labels_slice[orig_j]
            lbl_str = (
                cell_type_labels[slice_idx][orig_j]
                if cell_type_labels and cell_type_labels[slice_idx]
                else str(lbl_val)
            )
            node_labels.append(lbl_str)
            node_colors.append(rgba_to_plotly_string(color_dict[lbl_val]))
            node_x.append(slice_idx / (N - 1) if N > 1 else 0.5)
            node_y.append(y_positions[rank])
            node_idx_map[(slice_idx, orig_j)] = cur_idx
            cur_idx += 1

    # ─── links (use original indices → mapped positions) ────────────────────
    for t, T in enumerate(transition_list):
        rows, cols = T.shape
        for i in range(rows):
            for j in range(cols):
                val = T[i, j]
                if val <= threshold:
                    continue
                link_sources.append(node_idx_map[(t, i)])
                link_targets.append(node_idx_map[(t + 1, j)])
                link_values.append(val * linethick_factor)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                x=node_x,
                y=node_y,
            ),
            link=dict(source=link_sources, target=link_targets, value=link_values),
        )
    )
    fig.update_layout(
        title_text=title or "Differentiation Map (Sorted)",
        font_size=24,
        height=plot_height,
        width=plot_width,
    )

    if save_as_svg and save_name:
        fig.write_image(f"{save_name}.svg")
    fig.show()


# -----------------------------------------------------------------------------
# 3) Pipeline wrapper from (Qs, Ts)
# -----------------------------------------------------------------------------

def diffmap_from_QT_sankey(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    *,
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dsf: float = 1.0,
    plot_height: int = 600,
    plot_width: int = 1000,
    save_as_svg: bool = True,
    threshold: float = 0.0,
    order: bool = False,
) -> None:
    """High-level convenience wrapper for Sankey diff-maps."""

    # 1) get hard clustering per slice
    if clustering_type == "ml":
        clustering_list = max_likelihood_clustering(Qs)
    elif clustering_type == "reference":
        if reference_index is None:
            raise ValueError("reference_index required for 'reference' mode")
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError(f"Unknown clustering_type '{clustering_type}'.")

    # 2) colours & populations
    population_list, labels_list, color_dict = get_diffmap_inputs(
        clustering_list, clustering_type
    )

    # 3) delegate to plotting helper
    if order:
        plot_labeled_differentiation_sankey_sorted(
            population_list,
            Ts,
            labels_list,
            color_dict,
            cell_type_labels=cell_type_labels,
            linethick_factor=10,
            plot_height=plot_height,
            plot_width=plot_width,
            title=title,
            save_name=save_name,
            save_as_svg=save_as_svg,
            threshold=threshold,
        )
    else:
        plot_labeled_differentiation_sankey(
            population_list,
            Ts,
            labels_list,
            color_dict,
            cell_type_labels=cell_type_labels,
            linethick_factor=10,
            plot_height=plot_height,
            plot_width=plot_width,
            title=title,
            save_name=save_name,
            save_as_svg=save_as_svg,
            threshold=threshold,
        )


# -----------------------------------------------------------------------------
# 4) Pairwise GT ↔︎ predicted helper – retained for legacy notebooks
# -----------------------------------------------------------------------------

def make_sankey(
    gt_clustering: Union[List[int], np.ndarray],
    pred_clustering: Union[List[int], np.ndarray],
    gt_labels: List[str],
    *,
    save_format: str = "jpg",
    save_name: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Simple two-column Sankey for predicted vs ground-truth clusters."""

    df = np.vstack([gt_clustering, pred_clustering]).T
    gt_vals = np.unique(df[:, 0])
    pred_vals = np.unique(df[:, 1])

    labels = gt_labels + [f"Predicted {int(j)}" for j in pred_vals]
    num_gt = len(gt_vals)

    # build transition counts
    source, target, value = [], [], []
    for i, g in enumerate(gt_vals):
        for j, p in enumerate(pred_vals):
            count = int(((df[:, 0] == g) & (df[:, 1] == p)).sum())
            if count == 0:
                continue
            source.append(i)
            target.append(num_gt + j)
            value.append(count)

    # simple colour scheme
    node_colors = ["rgba(31,119,180,0.8)"] * num_gt + ["rgba(255,127,14,0.8)"] * len(pred_vals)

    fig = go.Figure(
        go.Sankey(
            node=dict(label=labels, pad=15, thickness=20, color=node_colors),
            link=dict(source=source, target=target, value=value),
        )
    )
    fig.update_layout(title_text=title or "Cluster Transition Sankey", font_size=12)

    if save_name:
        fmt = save_format.lower()
        fig.write_image(f"{save_name}.{fmt}")
    fig.show()
