from __future__ import annotations

from typing import List, Union, Dict, Tuple

import numpy as np
import scanpy as sc

from src.platting.color_utils import hex_to_rgba

from src.utils.clustering import (
    max_likelihood_clustering,
    reference_clustering
)

# public API
__all__ = [
    "get_scanpy_color_dict",
    "get_diffmap_inputs",
]

RGBA = Tuple[float, float, float, float]
# ────────────────────────────────────────────────────────────────────────────
# Color dictionary helper
# ────────────────────────────────────────────────────────────────────────────
def get_scanpy_color_dict(
    labels_list: List[Union[np.ndarray, List[int]]],
    *,
    alpha: float = 1.0,
) -> Dict[int, RGBA]:
    unique_values = np.unique(np.concatenate([np.asarray(l) for l in labels_list]))
    scanpy_colors = sc.pl.palettes.default_102  # 102 distinct hex codes
    rgba_colors = [
        hex_to_rgba(scanpy_colors[i % len(scanpy_colors)], alpha=alpha)
        for i in range(len(unique_values))
    ]
    return dict(zip(unique_values, rgba_colors))

# ────────────────────────────────────────────────────────────────────────────
# Get differentiation map inputs
# ────────────────────────────────────────────────────────────────────────────
def get_diffmap_inputs(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    clustering_type: str = "ml",
    reference_index: int = None,
    global_Qs: bool = False,
    full_P: bool = True,
) -> Tuple[List[List[int]], List[List[int]], Dict[int, RGBA]]:
    """
    Prepare inputs for plotting differentiation maps.
    Arguments:
        (*) Qs: List of arrays Q,
            where each Q is a matrix of shape (N, K),
            with N the total population at given timepoint,
            and K the number of clusters at that timepoint. 
        (*) Ts: List of arrays T,
            where each T is a matrix of shape (K1, K2),
            with K1 the number of clusters at earlier timepoint t1,
            and K2 the number of clusters at later timepoint t2.
            NOTE: Ts are needed for reference clustering only.
            NOTE: max likelihood clustering works when passing Ts=None
        (*) clustering_type: str, 
            "ml" for max likelihood(NOTE: ARGMAX of columns of Qs), or 
            "reference" for reference co-clustering.
        (*) reference_index: int,
            if clustering_type is "reference", 
            this is the index of the reference timepoint.
        (*) global_Qs: bool,
            if True, Qs are treated as global population matrices,
            meaning all timepoints share all cell types,
            otherwise they are treated as local population matrices.
    """
    raw_labels_list = []
    if global_Qs==True: # treat Qs as global population matrices (one cell type pool)
        if clustering_type == 'ml':
            raw_labels_list = max_likelihood_clustering(Qs)# [np.argmax(Q, axis=1) for Q in Qs]
            indices = list(range(Qs[0].shape[1]))          # 0…15
            color_dict = get_scanpy_color_dict([indices])
            # color_dict = get_scanpy_color_dict(raw_labels_list)
        elif clustering_type == 'reference':
            raw_labels_list = reference_clustering(Qs, Ts, reference_index, full_P=full_P)
            if reference_index is None:
                raise ValueError("reference_index needed for 'reference' clustering_type")
            raw_labels_list = [np.argmax(Q, axis=1) for Q in Qs]
            full_color_dict = get_scanpy_color_dict(raw_labels_list)
            reference_labels = raw_labels_list[reference_index]
            color_dict = { label : full_color_dict[label] for label in reference_labels}
        population_list = []
        for Q in Qs:
            pops = Q.sum(axis=0).astype(int).tolist()
            population_list.append(pops)
        indices = list(np.arange(Qs[0].shape[1]))  # assuming all Qs have same number of clusters
        index_list = [indices] * len(Qs)  # same labels for all timepoints
        
    else: # treat Qs as local population matrices (cell types change over time)
        if clustering_type == "ml":
            raw_labels_list = max_likelihood_clustering(Qs)
        elif clustering_type == "reference":
            if reference_index is None:
                raise ValueError("reference_index needed for 'reference' clustering_type")
            raw_labels_list = reference_clustering(Qs, Ts, reference_index, full_P=full_P)
        else:
            raise ValueError(f"Invalid clustering_type '{clustering_type}'.")

        population_list = []
        for raw_labels in raw_labels_list:
            labels = np.asarray(raw_labels)
            pops   = [int((labels == lbl).sum()) for lbl in np.unique(labels)]
            population_list.append(pops)

        # 2) label_list (optionally shifted) 
        index_list = []
        cumulative_shift = 0
        if clustering_type == "ml" and global_Qs==False:
            shifted_labels_list = []
            for raw_label in raw_labels_list:
                unique_labels = np.unique(raw_label)
                label_map = { old : old + cumulative_shift for old in unique_labels }
                shifted = np.vectorize(label_map.get)(raw_label)
                shifted_labels_list.append(shifted)
                index_list.append(list(np.unique(shifted)))
                cumulative_shift += len(unique_labels)
            raw_labels_list = shifted_labels_list
        elif clustering_type == "reference":
            for raw_label in raw_labels_list:
                index_list.append(list(np.unique(raw_label)))
        else:
            raise ValueError(f"Unknown clustering_type '{clustering_type}'.")

        # 3) color mapping
        color_dict = get_scanpy_color_dict(index_list)

    return raw_labels_list, population_list, index_list, color_dict