import numpy as np
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.patheffects as path_effects  
import re
from typing import List, Union, Dict, Tuple, Optional

import scanpy as sc

from clustering import max_likelihood_clustering, reference_clustering

################################################################################################
#   plotting helper functions
################################################################################################


def get_color_dict(
    labels_list: List[np.ndarray],
    cmap: str = "tab"
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Create a dictionary mapping unique cluster labels to distinct colors.

    This function collects all unique values from a list of label arrays
    (e.g., cluster assignments) and assigns each a color. By default, it uses
    the combined 'tab20', 'tab20b', and 'tab20c' colormaps, cycling through
    them as needed. If `cmap` is not 'tab', it uses a simple HSV-based
    generation scheme instead.

    Args:
        labels_list (List[np.ndarray]):
            A list of 1D arrays (e.g., cluster label assignments). Each array can
            have an arbitrary length. The function will take the union of all unique
            labels across these arrays.
        cmap (str, optional):
            Which colormap scheme to use. 
            - 'tab': combine 'tab20', 'tab20b', and 'tab20c' (good for up to ~60 clusters).
            - otherwise: generate colors using a simple HSV approach.
            Defaults to 'tab'.

    Returns:
        Dict[int, Tuple[float, float, float, float]]:
            A dictionary where keys are the unique labels (integers), and values
            are RGBA color tuples (floats in [0, 1]) suitable for Matplotlib plotting.

    Example:
        >>> import numpy as np
        >>> labels_list = [np.array([0, 1, 2]), np.array([2, 3, 4])]
        >>> color_map = get_color_dict(labels_list, cmap="tab")
        >>> print(color_map)
        {0: (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), 
         1: (...), 
         ... }
    """
    # Collect all unique labels across the list of label arrays
    unique_values = np.unique(np.concatenate(labels_list))

    # Handle 'tab' vs. fallback colormap
    if cmap == "tab":
        # Combine tab20, tab20b, tab20c
        cmap_tab20 = plt.get_cmap("tab20")
        cmap_tab20b = plt.get_cmap("tab20b")
        cmap_tab20c = plt.get_cmap("tab20c")

        colors_tab20 = [cmap_tab20(i) for i in range(cmap_tab20.N)]
        colors_tab20b = [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
        colors_tab20c = [cmap_tab20c(i) for i in range(cmap_tab20c.N)]

        combined_colors = colors_tab20 + colors_tab20b + colors_tab20c

        print(f"Total 'tab' colors available: {len(combined_colors)}")
        print(f"Number of unique labels: {len(unique_values)}")

        # Cycle through combined_colors if we have more unique labels than colors
        colors = [
            combined_colors[i % len(combined_colors)]
            for i in range(len(unique_values))
        ]
    else:
        # Simple HSV-based color generation
        num_labels = len(unique_values)
        colors = [
            mcolors.hsv_to_rgb((i / num_labels, 1, 1)) + (1.0,)  # (R, G, B, A=1)
            for i in range(num_labels)
        ]

    # Construct the dictionary from label -> color
    color_dict = dict(zip(unique_values, colors))
    return color_dict


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Convert a hex color code (e.g., '#1f77b4') to an RGBA tuple.

    Args:
        hex_color (str):
            A string representing a color in hex format, starting with '#'.
        alpha (float, optional):
            The alpha (opacity) channel value in the range [0, 1].
            Defaults to 1.0 (fully opaque).

    Returns:
        Tuple[float, float, float, float]:
            RGBA color components, each in the range [0, 1].

    Example:
        >>> rgba = hex_to_rgba("#1f77b4", alpha=0.8)
        >>> print(rgba)
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.8)
    """
    # Convert hex to an (R, G, B) tuple in [0,1]
    rgb = mcolors.hex2color(hex_color)
    return (*rgb, alpha)  # Append alpha and return


def get_scanpy_color_dict(
    labels_list: List[Union[np.ndarray, List[int]]],
    alpha: float = 1.0
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Create a color dictionary for cluster labels using Scanpy's predefined palette.

    Args:
        labels_list (List[Union[np.ndarray, List[int]]]):
            A list of arrays or lists of integers, where each sub-array/list contains
            label assignments (e.g., cluster IDs). The function will gather all unique
            labels across these inputs.
        alpha (float, optional):
            Alpha (transparency) value in the range [0, 1] for the RGBA colors.
            Defaults to 1.0.

    Returns:
        Dict[int, Tuple[float, float, float, float]]:
            A dictionary where each key is a unique cluster label (int) and each
            value is an RGBA color tuple. Colors are drawn from Scanpy’s
            `default_102` palette; if there are more labels than 102, the palette
            is cycled through again.

    Example:
        >>> labels_list = [
        ...     [0, 1, 2, 2],
        ...     [1, 2, 3],
        ... ]
        >>> color_map = get_scanpy_color_dict(labels_list, alpha=0.9)
        >>> color_map[0]
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.9)
    """
    # Gather all unique labels
    unique_values = np.unique(np.concatenate(labels_list))

    # Use Scanpy's predefined palette (default_102 has 102 distinct hex colors)
    scanpy_colors = sc.pl.palettes.default_102

    # Convert each hex color to RGBA (with the specified alpha), cycling if needed
    rgba_colors = [
        hex_to_rgba(scanpy_colors[i % len(scanpy_colors)], alpha=alpha)
        for i in range(len(unique_values))
    ]

    # Map unique labels to RGBA colors
    color_dict = dict(zip(unique_values, rgba_colors))
    return color_dict


def get_diffmap_inputs(
    clustering_list: List[np.ndarray],
    clustering_type: str,
    cmap: str = "tab"
) -> Tuple[
    List[List[int]],
    List[List[int]],
    Dict[int, Tuple[float, float, float, float]]
]:
    """
    Prepare population counts, label lists, and a color dictionary for 
    diffusion map or embedding visualizations.

    This function:
      1. Computes the number of spots in each cluster per slice (population_list).
      2. Builds a list of cluster labels for each slice (label_list). If
         `clustering_type == 'ml'`, it shifts labels slice-by-slice so that each
         slice has a unique label range. If `clustering_type == 'reference'`,
         it keeps labels as-is.
      3. Calls `get_scanpy_color_dict` to generate a color dictionary that maps
         labels to RGBA color tuples.

    Args:
        clustering_list (List[np.ndarray]):
            A list of 1D arrays. Each array represents cluster labels for 
            the spots in one slice (e.g., shape (n_t,)).
        clustering_type (str):
            Determines label-offset handling:
              - 'ml': merges slices by shifting each subsequent slice's labels 
                      so they won't overlap with earlier slices.
              - 'reference': preserves the raw labels per slice.
        cmap (str, optional):
            Colormap scheme passed down to color-generation logic. 
            Defaults to "tab".

    Returns:
        Tuple[
            List[List[int]], 
            List[List[int]], 
            Dict[int, Tuple[float, float, float, float]]
        ]: 
            A 3-item tuple consisting of:
              1. population_list (List[List[int]]): 
                 The counts of spots per cluster for each slice.
              2. label_list (List[List[int]]): 
                 A list of lists of cluster labels, possibly shifted 
                 (depending on `clustering_type`).
              3. color_dict (Dict[int, (float, float, float, float)]): 
                 Maps labels to RGBA color tuples.

    Example:
        >>> import numpy as np
        >>> clustering_list = [
        ...     np.array([0, 0, 1]), 
        ...     np.array([1, 2, 2, 3])
        ... ]
        >>> pop_list, lbl_list, color_dict = get_diffmap_inputs(
        ...     clustering_list, 
        ...     'ml', 
        ...     cmap='tab'
        ... )
        >>> print(pop_list)
        [[2, 1], [1, 2, 1]]
        >>> print(lbl_list)
        [[0, 1], [4, 5, 6, 7]] 
        # second slice got shifted to ensure no overlap
        >>> print(color_dict) 
        # e.g. {0: (R, G, B, A), 1: (...), 4: (...), 5: (...), ...}
    """

    # 1) population_list: spots count in each cluster per slice
    population_list: List[List[int]] = []
    for clustering in clustering_list:
        labels_in_slice = set(clustering)
        counts = [np.sum(clustering == lbl) for lbl in labels_in_slice]
        population_list.append(counts)

    # 2) label_list, optionally shifting labels for 'ml'
    label_list: List[List[int]] = []
    cumulative_shift = 0

    if clustering_type == "ml":
        for clustering in clustering_list:
            slice_labels_shifted = [lbl + cumulative_shift for lbl in clustering]
            unique_labels_shifted = list(set(slice_labels_shifted))
            label_list.append(unique_labels_shifted)
            # Increase shift by how many unique labels were in this slice
            cumulative_shift += len(set(clustering))
    elif clustering_type == "reference":
        for clustering in clustering_list:
            label_list.append(list(set(clustering)))
    else:
        # optionally raise an error or do nothing
        pass

    # 3) color_dict for all labels (flattened from label_list)
    color_dict = get_scanpy_color_dict(label_list)

    return population_list, label_list, color_dict


################################################################################################
#   plotting: core functions
################################################################################################

def plot_clustering_list(
    spatial_list: List[np.ndarray],
    clustering_list: List[np.ndarray],
    clustering_type: str = "ml",
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    cmap: str = "tab",
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[List[str]]]] = None
) -> None:
    """
    Plot a row of scatterplots for each slice in spatial_list, colored by cluster assignments.

    This function displays each slice in a separate subplot, showing the spatial coordinates 
    (x, y) of spots colored by their cluster labels. If 'clustering_type' is 'ml', labels in 
    consecutive slices are shifted to ensure they do not overlap numerically. If it's 
    'reference', labels are used as-is.

    Args:
        spatial_list (List[np.ndarray]):
            A list of 2D arrays, each of shape (n_spots, 2). Each array contains the
            (x, y) coordinates for spots in one slice.
        clustering_list (List[np.ndarray]):
            A list of 1D arrays, each of shape (n_spots,). Each array contains the 
            cluster labels for spots in the corresponding slice.
        clustering_type (str, optional):
            One of:
              - 'ml': (default) shift labels slice by slice to ensure unique label ranges.
              - 'reference': keep labels as-is.
        cell_type_labels (List[Optional[List[str]]] | None, optional):
            A list of label lists or None, one per slice. If provided, each subplot's legend
            is replaced with the corresponding cell_type_labels[i]. Defaults to None.
        cmap (str, optional):
            Name of the colormap passed into downstream color-dict generation. Default is 'tab'.
        title (str, optional):
            A global figure title. Defaults to None (no title).
        save_name (str, optional):
            If provided, saves the plot to the specified filename (e.g., 'figure.png') 
            with DPI=300. Defaults to None (no file saved).
        dotsize (float, optional):
            Marker size for the scatterplot. Defaults to 1.0.
        flip (bool, optional):
            If True, flips the y-axis (multiplying y by -1). Helpful for coordinate systems
            with reversed orientation. Defaults to False.
        subplot_labels ((List[str]), optional)
            Labels for each sub-plot, if none default to numbered labeling.

    Returns:
        None. Displays a row of subplots, one per slice, and optionally saves them.

    Notes:
        - It uses `get_diffmap_inputs` internally to obtain a color dictionary, 
          so ensure that function is accessible.
        - The background of each subplot is set to black, and the default style is 
          `sns.set_style("white")` with `sns.set_context("notebook")`.

    Example:
        >>> import numpy as np
        >>> spatial_0 = np.random.rand(100, 2)  # 100 spots, 2D coords
        >>> spatial_1 = np.random.rand(120, 2)
        >>> clust_0 = np.random.randint(0, 3, size=100)  # 3 clusters
        >>> clust_1 = np.random.randint(0, 4, size=120)  # 4 clusters
        >>> plot_clustering_list(
        ...     [spatial_0, spatial_1],
        ...     [clust_0, clust_1],
        ...     clustering_type='ml',
        ...     title='My Clustering Visualization'
        ... )
    """
    # Number of slices
    N_slices = len(spatial_list)

    # Provide a default list of cell_type_labels if none was supplied
    if cell_type_labels is None:
        cell_type_labels = [None] * N_slices

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5)

    # Create subplots: one per slice
    fig, axes = plt.subplots(1, N_slices, figsize=(20 * N_slices, 20), facecolor="white")

    # Center each slice around its mean
    slices = [S - np.mean(S, axis=0) for S in spatial_list]

    # Obtain color dict from get_diffmap_inputs (ignoring population_list, label_list)
    _, _, color_dict = get_diffmap_inputs(clustering_list, clustering_type, cmap)

    # Determine global x/y bounds so all subplots share the same limits
    all_spatial = np.vstack(slices)
    x_min, x_max = np.min(all_spatial[:, 0]), np.max(all_spatial[:, 0])
    y_min, y_max = np.min(all_spatial[:, 1]), np.max(all_spatial[:, 1])

    # Only used if clustering_type == 'ml'
    cumulative_shift = 0

    # Plot each slice in its own axis
    for i, (coords, labels) in enumerate(zip(slices, clustering_list)):
        ax = axes[i] if N_slices > 1 else axes  # If only one subplot, axes is single
        ax.set_facecolor("black")

        # Shift labels if 'ml' mode
        if clustering_type == "ml":
            shifted_labels = labels + cumulative_shift
        else:
            shifted_labels = labels

        # Optionally flip the y-axis
        if flip:
            coords = coords @ np.array([[-1, 0], [0, 1]])

        # Build a DataFrame for plotting
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "value": shifted_labels
        })

        # If 'ml', increment the label space for the next slice
        if clustering_type == "ml":
            cumulative_shift += len(set(labels))

        # Scatter plot
        sns.scatterplot(
            x="x", y="y", hue="value",
            palette=color_dict,
            data=df, ax=ax,
            s=dotsize, legend=True
        )

        # Set consistent axes limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Optionally invert y-axis if you prefer conventional top->bottom orientation
        if flip:
            ax.invert_yaxis()

        # Remove axis boundaries / ticks
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        if subplot_labels is not None:
            ax.set_title(subplot_labels[i], color="black")
        else:
            ax.set_title(f"Slice {i+1}", color="black")

        # If cell_type_labels[i] is provided, override the legend labels
        if cell_type_labels[i] is not None:
            handles, lbls = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=cell_type_labels[i], title="")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)  # horizontal spacing

    if title:
        plt.suptitle(title, fontsize=36, color="black")

    # Save figure if requested
    if save_name is not None:
        plt.savefig(save_name, dpi=300, transparent=True,
                    bbox_inches="tight", facecolor="black")
    plt.show()


def plot_labeled_differentiation(
    population_list: List[List[int]],
    transition_list: List[np.ndarray],
    label_list: List[List[int]],
    color_dict: Dict[int, Union[str, tuple]],
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,  # Not used within the function, for future
    dotsize_factor: float = 1.0,
    linethick_factor: float = 10.0,
    save_name: Optional[str] = None,
    title: Optional[str] = None,
    stretch: float = 1.0,
    outline: float = 3.0,
    fontsize: int = 12
) -> None:
    """
    Plot a 2D "differentiation map" showing transitions between clusters
    across multiple slices of data. Each cluster is represented by a node
    (dot) whose size is proportional to the population count, and edges
    (lines) represent the transition flow between clusters in consecutive slices.

    Args:
        population_list (List[List[int]]):
            A list of length N_slices. Each element is a list of population counts
            for that slice’s clusters.
        transition_list (List[np.ndarray]):
            A list of length (N_slices - 1). Each element T is a 2D array
            (shape (r_t, r_{t+1})) representing the cell-type coupling matrix
            between consecutive slices. T[i, j] often denotes transition mass
            or flow from cluster i in slice t to cluster j in slice t+1.
        label_list (List[List[int]]):
            A list of length N_slices, where each sub-list contains the cluster
            labels (integers) for that slice. Used to look up node colors
            in color_dict.
        color_dict (Dict[int, Union[str, tuple]]):
            A dictionary mapping each cluster label (int) to a color specification
            (e.g., an (R, G, B) tuple or a Matplotlib-recognized color string).
        cell_type_labels (List[Optional[List[str]]] | None, optional):
            A list of length N_slices. Each element is either None or a list of
            strings. If provided, the text is rendered next to each node. Defaults
            to None (no text labels).
        clustering_type (str, optional):
            One of:
              - 'ml': (default) plot lines with thickness proportional to T[i, j].
              - 'reference': logic not implemented for lines; the code
                effectively skips the line plotting block if not 'ml'.
        reference_index (int | None, optional):
            Unused in this function’s logic. Provided for consistency with
            external workflows. Defaults to None.
        dotsize_factor (float, optional):
            A scalar that multiplies each cluster’s population to determine
            the scatter marker size. Defaults to 1.0.
        linethick_factor (float, optional):
            A scalar multiplied by T[i, j] to set the edge thickness. Defaults to 10.0.
        save_name (str, optional):
            If provided, saves the final figure to the given path or filename
            (e.g., "plot.png"). Defaults to None (no saving).
        title (str, optional):
            A title at the top of the figure. Defaults to None.
        stretch (float, optional):
            Horizontal stretch factor for the figure size. Defaults to 1.0.
        outline (float, optional):
            Width of a white outline drawn around text labels, for clarity on 
            dark backgrounds. Defaults to 3.0.

    Returns:
        None. Displays the figure (and optionally saves it).

    Notes:
        - The function arranges slices along the x-axis, with x-positions at
          slice indices. The y-positions are spread vertically for each cluster
          in that slice.
        - Edges (lines) are only drawn if `clustering_type == 'ml'` and
          `T[i, j] > 0`.
        - A black background is used for better contrast, and each node has a
          simple blue edge color for distinction.

    Example:
        >>> # Suppose we have 3 slices, each with cluster populations:
        >>> population_list = [[100, 50], [60, 90, 10], [150, 25]]
        >>> # Coupling between slice 0->1 and slice 1->2
        >>> T01 = np.array([[0.1, 0.9, 0.0],
        ...                 [0.4, 0.5, 0.1]])
        >>> T12 = np.array([[0.3, 0.7],
        ...                 [0.2, 0.8],
        ...                 [0.0, 1.0]])
        >>> transition_list = [T01, T12]
        >>> label_list = [[0, 1], [2, 3, 4], [5, 6]]
        >>> color_dict = {0: "red", 1: "blue", 2: "green",
        ...               3: "orange", 4: "purple", 5: "cyan", 6: "yellow"}
        >>> plot_labeled_differentiation(
        ...     population_list, transition_list, label_list, color_dict,
        ...     dotsize_factor=5, linethick_factor=10
        ... )
    """
    # Set seaborn style
    sns.set(style="white")

    dsf = dotsize_factor
    ltf = linethick_factor

    N_slices = len(population_list)

    # Build x-positions and y-positions for each slice’s clusters
    x_positions = []
    y_positions = []
    for i, population in enumerate(population_list):
        y_positions.append(np.arange(len(population)))
        x_positions.append(np.ones(len(population)) * i)
    
    # Configure figure size
    plt.figure(figsize=(stretch * 5 * (N_slices - 1), 10))

    # Plot each slice’s nodes and lines between consecutive slices
    for pair_ind, T in enumerate(transition_list):
        # Current slice
        plt.scatter(
            x_positions[pair_ind],
            y_positions[pair_ind],
            c=[color_dict[label] for label in label_list[pair_ind]],
            s=dsf * np.array(population_list[pair_ind]),
            edgecolor="b",
            linewidth=1,
            zorder=1,
        )
        # Next slice
        plt.scatter(
            x_positions[pair_ind + 1],
            y_positions[pair_ind + 1],
            c=[color_dict[label] for label in label_list[pair_ind + 1]],
            s=dsf * np.array(population_list[pair_ind + 1]),
            edgecolor="b",
            linewidth=1,
            zorder=1,
        )

        r1, r2 = T.shape[0], T.shape[1]
        
        # Draw lines only for 'ml' (plus T[i, j] > 0)
        if clustering_type == "ml":
            for i_row in range(r1):
                for j_col in range(r2):
                    if T[i_row, j_col] > 0.0:
                        plt.plot(
                            [x_positions[pair_ind][i_row], x_positions[pair_ind + 1][j_col]],
                            [y_positions[pair_ind][i_row], y_positions[pair_ind + 1][j_col]],
                            "k-",
                            lw=T[i_row, j_col] * ltf,
                            zorder=0,
                        )
        else:
            # 'reference' or other type: skip line plotting
            pass

    # Optionally add text labels for each node (cluster)
    if cell_type_labels is not None:
        for i in range(N_slices):
            if cell_type_labels[i] is not None:
                for j, label_text in enumerate(cell_type_labels[i]):
                    txt = plt.text(
                        x_positions[i][j],
                        y_positions[i][j],
                        label_text,
                        fontsize=fontsize,
                        ha="right",
                        va="bottom",
                    )
                    # Add a white outline behind text for contrast
                    txt.set_path_effects([
                        path_effects.Stroke(linewidth=outline, foreground="white"),
                        path_effects.Normal()
                    ])

    # Title settings
    if title:
        plt.suptitle(title, fontsize=36, color="black")
    else:
        plt.title("Differentiation Map")

    # Remove tick marks and spines
    plt.yticks([])
    plt.xticks([])
    plt.axis("off")
    sns.despine()

    # Optional: save figure
    if save_name is not None:
        plt.savefig(save_name, dpi=300, transparent=True,
                    bbox_inches="tight", facecolor="black")
    
    plt.rcParams['figure.dpi'] = 300
    plt.show()

################################################################################################
#   plotting: more directly from from output Qs, Ts
################################################################################################

def diffmap_from_QT(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dsf: float = 1.0,
    stretch: float = 1.0,
    outline: float = 2.0,
    fontsize: int = 12,
    linethick_factor: int = 10
) -> None:
    """
    Create and plot a diffusion map (or differentiation map) from sets of Q and T matrices,
    optionally using either max-likelihood clustering or reference clustering.

    The Q matrices specify per-slice distributions between spots and clusters, while
    the T matrices specify transitions between consecutive slices. Based on the chosen
    clustering type, this function:
      1. Builds a clustering_list using either `max_likelihood_clustering` or
         `reference_clustering`.
      2. Derives the population list, labels list, and color dictionary from
         `get_diffmap_inputs`.
      3. Calls `plot_labeled_differentiation` to display a high-level flow
         or "differentiation" diagram of how clusters evolve across slices.

    Args:
        Qs (List[np.ndarray]):
            A list of length N, where each element Qs[t] is a 2D array (shape (n_t, r_t))
            representing the distribution for slice t.
        Ts (List[np.ndarray]):
            A list of length N-1, where each element Ts[t] is a 2D array
            (shape (r_t, r_{t+1})) specifying transitions between consecutive slices.
        cell_type_labels (List[Optional[List[str]]] | None, optional):
            A list of length N. Each element is either None or a list of 
            textual labels for each cluster in the slice. If provided, these
            labels are rendered in the final plot. Defaults to None.
        clustering_type (str, optional):
            One of:
              - 'ml' (default): uses `max_likelihood_clustering`.
              - 'reference': uses `reference_clustering`, requiring reference_index.
        reference_index (int | None, optional):
            Required if clustering_type == 'reference'. Specifies which slice
            to treat as a reference. Ignored otherwise.
        title (str, optional):
            A title for the final figure. Defaults to None.
        save_name (str, optional):
            Filename (or path) to save the final figure. If None, no file is saved.
        dsf (float, optional):
            "Dotsize factor" for node sizes in the final plot. Default is 1.0.
        stretch (float, optional):
            Horizontal stretch factor of the final figure. Default is 1.0.
        outline (float, optional):
            The width of the white outline around text labels in the final plot.
            Default is 2.0.
        fontsize (int, optional):
            Fontsize of labels.
        linethick_factor (int, optional):
            Thickness of lines.

    Returns:
        None. This function displays (and optionally saves) a plot showing
        cluster transitions across slices.
    """
    # 1) Build clustering_list using chosen approach
    if clustering_type == "ml":
        clustering_list = max_likelihood_clustering(Qs)
    elif clustering_type == "reference":
        if reference_index is None:
            raise ValueError("reference_index is required for 'reference' clustering_type.")
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError(f"Invalid clustering_type: '{clustering_type}'.")

    # 2) Get population_list, labels_list, and color_dict
    for i in range(len(clustering_list)):
        list_size = len(np.unique(clustering_list[i]))
        rank = Qs[i].shape[1]
        if list_size != rank:
            raise ValueError(f"Degenerate clusters, rank '{rank}' not equal to number of clusters '{list_size}'.")
    
    population_list, labels_list, color_dict = get_diffmap_inputs(clustering_list, clustering_type)

    # 3) Plot the differentiation map
    plot_labeled_differentiation(
        population_list=population_list,
        transition_list=Ts,
        label_list=labels_list,
        color_dict=color_dict,
        cell_type_labels=cell_type_labels,
        clustering_type=clustering_type,
        dotsize_factor=dsf,
        linethick_factor=linethick_factor,
        title=title,
        save_name=save_name,
        stretch=stretch,
        outline=outline,
        fontsize=fontsize
    )


def plot_clusters_from_QT(
    Ss: List[np.ndarray],
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
    dotsize: float = 1.0,
    flip: bool = False,
    subplot_labels: Optional[List[Optional[List[str]]]] = None
) -> None:
    """
    Generate a scatterplot visualization of cluster assignments for one or 
    more slices, given spatial coordinates (Ss), per-slice probability 
    matrices (Qs), and transitions (Ts).

    Depending on clustering_type, the function either:
      1. Uses max-likelihood clustering ('ml'), or
      2. Uses reference clustering ('reference'), requiring reference_index.

    The final plot is produced by calling `plot_clustering_list`, showing
    each slice’s coordinates colored by cluster labels.

    Args:
        Ss (List[np.ndarray]):
            A list of length N, where each element Ss[t] has shape (n_t, 2),
            giving the (x, y) spatial coordinates of spots in slice t.
        Qs (List[np.ndarray]):
            A list of length N, where Qs[t] has shape (n_t, r_t), representing 
            the distribution of spots vs. clusters in slice t.
        Ts (List[np.ndarray]):
            A list of length (N-1), where each element has shape (r_t, r_{t+1}),
            specifying transitions between consecutive slices.
        cell_type_labels (List[Optional[List[str]]] | None, optional):
            A list of length N, each element is either None or a list of strings
            labeling each cluster in the slice. Defaults to None (no text labels).
        clustering_type (str, optional):
            The clustering mode:
              - 'ml': (default) use `max_likelihood_clustering(Qs)`.
              - 'reference': use `reference_clustering(Qs, Ts, reference_index)`;
                requires reference_index. 
        reference_index (int | None, optional):
            Index of the reference slice if clustering_type == 'reference'.
            Defaults to None.
        title (str, optional):
            An optional title displayed above the entire figure. Defaults to None.
        save_name (str, optional):
            If provided, saves the resulting figure to this file path. Defaults to None.
        dotsize (float, optional):
            Size factor for scatterplot markers. Defaults to 1.0.
        flip (bool, optional):
            If True, flips (mirrors) the coordinate system along the y-axis. Defaults to False.
        subplot_labels ((List[str]), optional)
            Labels for each sub-plot, if none default to numbered labeling.
    Returns:
        None. Displays (and optionally saves) a scatterplot of cluster assignments
        for each slice.

    Example:
        >>> # Suppose we have 2 slices of data:
        >>> Ss = [np.random.rand(100, 2), np.random.rand(120, 2)]
        >>> Qs = [np.random.rand(100, 3), np.random.rand(120, 4)]
        >>> T01 = np.random.rand(3, 4)  # transitions between slice 0 and 1
        >>> Ts = [T01]
        >>> plot_clusters_from_QT(Ss, Qs, Ts, clustering_type='ml', dotsize=2)
    """
    # Build the clustering_list
    if clustering_type == "ml":
        clustering_list = max_likelihood_clustering(Qs)
    elif clustering_type == "reference":
        if reference_index is None:
            raise ValueError("Reference index is required for reference clustering.")
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError(f"Invalid clustering_type: '{clustering_type}'.")

    # Call the plotting function
    plot_clustering_list(
        spatial_list=Ss,
        clustering_list=clustering_list,
        cell_type_labels=cell_type_labels,
        clustering_type=clustering_type,
        cmap="tab",
        title=title,
        save_name=save_name,
        dotsize=dotsize,
        flip=flip,
        subplot_labels=subplot_labels
    )


################################################################################################
#   Sankey plotting utils
################################################################################################

def rgba_to_plotly_string(rgba: Tuple[float, float, float, float]) -> str:
    """
    Convert an RGBA tuple (floats in [0,1]) to a Plotly-compatible RGBA string.

    Args:
        rgba (Tuple[float, float, float, float]):
            A 4-tuple of (red, green, blue, alpha), each in [0, 1].

    Returns:
        str: A string of the form 'rgba(r, g, b, a)' where r, g, b are in [0, 255].

    Example:
        >>> rgba_str = rgba_to_plotly_string((0.5, 0.2, 0.1, 0.8))
        >>> print(rgba_str)
        'rgba(127, 51, 25, 0.8)'
    """
    r, g, b, a = rgba
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})"

def alphabetic_key(label: str) -> str:
    """
    A custom key function for sorting strings alphabetically,
    ignoring non-alphabetic characters (e.g., digits or punctuation).

    Args:
        label (str): The label or string to be processed.

    Returns:
        str: The label with all non-alphabetic characters removed,
             suitable for alphabetical sorting.

    Example:
        >>> labels = ["Cell 2", "Cell_10", "Cell-1"]
        >>> labels.sort(key=alphabetic_key)
        >>> print(labels)
        ['Cell-1', 'Cell 2', 'Cell_10']
    """
    return re.sub("[^a-zA-Z]", "", label)


################################################################################################
#   Sankey plotting, main functions
################################################################################################

def plot_labeled_differentiation_sankey(
    population_list: List[List[int]],
    transition_list: List[np.ndarray],
    label_list: List[List[int]],
    color_dict: Dict[int, Tuple[float, float, float, float]],
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    dotsize_factor: float = 1.0,
    linethick_factor: float = 10.0,
    plot_height: int = 600,
    plot_width: int = 1000,
    save_name: Optional[str] = None,
    title: Optional[str] = None,
    save_as_svg: bool = True,
    threshold: float = 0.0
) -> None:
    """
    Create a Sankey diagram representing multi-slice differentiation or transitions between clusters.

    This function arranges clusters from each slice as nodes in a horizontal sequence,
    then draws directed links (edges) between slices based on the values in transition_list.
    Colors are defined by color_dict, with optional thresholding to remove very small flows.

    Args:
        population_list (List[List[int]]):
            A list of length N_slices. Each element is a list of population counts per cluster.
            (Currently not used directly in the Sankey, but may be useful if adjusting thickness 
             by population.)
        transition_list (List[np.ndarray]):
            A list of length (N_slices-1). Each element is an array T of shape (r_t, r_{t+1}),
            describing transition intensities from slice t to slice t+1.
        label_list (List[List[int]]):
            A list of length N_slices, each sub-list containing the numeric cluster labels in that slice.
        color_dict (Dict[int, Tuple[float, float, float, float]]):
            A dictionary mapping cluster labels to RGBA tuples in [0,1].
        cell_type_labels (List[Optional[List[str]]] | None, optional):
            A list of length N_slices. Each element is either None or a list of 
            strings for naming each cluster in that slice.
        clustering_type (str, optional):
            For reference only in this function; by default "ml" or "reference".
            Not actively used in plotting logic beyond storing or printing.
        reference_index (int | None, optional):
            Also for reference if needed externally; not used here.
        dotsize_factor (float, optional):
            Currently unused, but could scale node sizes in a future extension.
        linethick_factor (float, optional):
            Scales the numeric values in T for the link "value", controlling line thickness.
            Defaults to 10.0.
        plot_height (int, optional):
            Height of the resulting Sankey diagram in pixels. Defaults to 600.
        plot_width (int, optional):
            Width of the resulting Sankey diagram in pixels. Defaults to 1000.
        save_name (str, optional):
            Base filename for saving the figure. If None, no file is written.
        title (str, optional):
            Title for the diagram. Defaults to None.
        save_as_svg (bool, optional):
            If True, saves the plot as an SVG file. Defaults to True.
        threshold (float, optional):
            Minimum transition value to include as an edge. Values below threshold are omitted.
            Defaults to 0.0 (include all).

    Returns:
        None. Displays the Sankey diagram in a Plotly figure window and optionally saves it.
    """
    # Number of slices
    N_slices = len(population_list)
    
    # Prepare lists for Sankey node/link definitions
    node_labels = []
    link_sources = []
    link_targets = []
    link_values = []
    node_colors = []

    # Map from (slice_idx, cluster_idx_within_slice) -> global node index
    node_idx_map = {}
    current_node_idx = 0

    # 1) Build Sankey nodes from label_list
    for slice_idx, pop_slice in enumerate(population_list):
        for i, cluster_label in enumerate(label_list[slice_idx]):
            # Use cell_type_labels if available; fallback to numeric label as string
            if cell_type_labels and cell_type_labels[slice_idx] is not None:
                label_str = cell_type_labels[slice_idx][i] or str(cluster_label)
            else:
                label_str = str(cluster_label)

            node_labels.append(label_str)
            node_idx_map[(slice_idx, i)] = current_node_idx

            # Convert RGBA -> Plotly color string
            node_colors.append(rgba_to_plotly_string(color_dict[cluster_label]))

            current_node_idx += 1

    # 2) Build Sankey links from transition_list
    for t_idx, T in enumerate(transition_list):
        rows, cols = T.shape
        for i_row in range(rows):
            for j_col in range(cols):
                val = T[i_row, j_col]
                if val > threshold:
                    source_node = node_idx_map[(t_idx, i_row)]
                    target_node = node_idx_map[(t_idx + 1, j_col)]
                    link_sources.append(source_node)
                    link_targets.append(target_node)
                    link_values.append(val * linethick_factor)

    # 3) Construct the Sankey figure
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values
        )
    ))

    # 4) Configure the layout
    fig.update_layout(
        title_text=title if title else "Differentiation Map",
        font_size=24,
        height=plot_height,
        width=plot_width
    )

    # 5) Optional: save the figure
    if save_as_svg and save_name is not None:
        fig.write_image(f"{save_name}.svg")
    elif save_as_svg and save_name is None:
        fig.write_image("diffmap.svg")  # default

    # 6) Display the figure
    fig.show()


def plot_labeled_differentiation_sankey_sorted(
    population_list: List[List[int]],
    transition_list: List[np.ndarray],
    label_list: List[List[int]],
    color_dict: Dict[int, Tuple[float, float, float, float]],
    cell_type_labels: Optional[List[Optional[List[str]]]] = None,
    clustering_type: str = "ml",
    reference_index: Optional[int] = None,
    dotsize_factor: float = 1.0,
    linethick_factor: float = 10.0,
    plot_height: int = 600,
    plot_width: int = 800,
    save_name: Optional[str] = None,
    title: Optional[str] = None,
    save_as_svg: bool = False,
    threshold: float = 0.0
) -> None:
    """
    Create a Sankey diagram similar to `plot_labeled_differentiation_sankey`, but with
    **manually assigned** (x,y) node positions and sorted cluster labels within each slice.

    This is useful if you want to ensure the nodes appear in a particular order vertically
    (e.g., alphabetical), or to reduce visual clutter by spacing them out. Each transition
    is thresholded, removing links below 'threshold'.

    Args:
        population_list (List[List[int]]):
            List of length N, each element is the population counts per cluster in that slice
            (currently not directly used in the Sankey, but could be in a future extension).
        transition_list (List[np.ndarray]):
            A list of length N-1, each element shape (r_t, r_{t+1}), specifying transitions
            between consecutive slices t and t+1.
        label_list (List[List[int]]):
            Cluster labels for each slice. Each sub-list is of length r_t. 
        color_dict (Dict[int, Tuple[float, float, float, float]]):
            Maps each cluster label to an RGBA color. 
        cell_type_labels (List[Optional[List[str]]] | None, optional):
            A list of length N. Each sub-list is either None or a list of 
            string labels for each cluster in the slice.
        clustering_type (str, optional):
            For reference. Defaults to 'ml'.
        reference_index (int | None, optional):
            Not used here; part of a pipeline context. Defaults to None.
        dotsize_factor (float, optional):
            Unused in this function, but left for consistency. Defaults to 1.0.
        linethick_factor (float, optional):
            Scales T[i, j] for link thickness. Default is 10.0.
        plot_height (int, optional):
            Height of the Sankey diagram in pixels. Default is 600.
        plot_width (int, optional):
            Width of the Sankey diagram in pixels. Default is 800.
        save_name (str, optional):
            If provided, saves the figure to this filename. Defaults to None.
        title (str, optional):
            Title for the Sankey diagram. Defaults to None.
        save_as_svg (bool, optional):
            If True, saves as an SVG file instead of a raster format. Defaults to False.
        threshold (float, optional):
            Minimum transition value to include. Links below this are omitted. Default is 0.0.

    Returns:
        None. Displays the Sankey diagram with manually spaced nodes and optionally saves it.

    Notes:
        - Sorting is achieved by the custom `alphabetic_key` which strips non-alphabetic chars.
        - The (x, y) positions of each node are manually computed to maintain separation and order.
        - If you want automatic layout, see `plot_labeled_differentiation_sankey` instead.
    """
    # Preliminary setups
    import numpy as np
    import plotly.graph_objects as go

    N_slices = len(population_list)

    # Gather data for Sankey
    node_labels = []
    link_sources = []
    link_targets = []
    link_values = []
    node_colors = []
    node_x = []
    node_y = []

    # Map from (slice_idx, cluster_original_idx) -> global node index
    node_idx_map = {}
    current_node_idx = 0

    # 1) Sort and place nodes
    sorted_label_list = []
    sorted_indices_list = []

    for slice_idx, labels_this_slice in enumerate(label_list):
        # Build an indexing approach. 
        # If cell_type_labels is available, sort by the textual label; otherwise by numeric label.
        def label_to_sort_key(i: int) -> str:
            if cell_type_labels and cell_type_labels[slice_idx] is not None:
                return alphabetic_key(cell_type_labels[slice_idx][i] or str(labels_this_slice[i]))
            else:
                return alphabetic_key(str(labels_this_slice[i]))

        # Sort indices in ascending alphabetical order
        sorted_indices = sorted(range(len(labels_this_slice)), key=label_to_sort_key)
        sorted_indices_list.append(sorted_indices)

        # Create a sorted textual label list, fallback to numeric if needed
        sorted_text_labels = []
        for i in sorted_indices:
            if cell_type_labels and cell_type_labels[slice_idx] is not None:
                lbl_str = cell_type_labels[slice_idx][i] or str(labels_this_slice[i])
            else:
                lbl_str = str(labels_this_slice[i])
            sorted_text_labels.append(lbl_str)
        sorted_label_list.append(sorted_text_labels)

        # 2) Assign vertical spacing (y-coordinates)
        num_nodes_slice = len(labels_this_slice)
        padding = 0.1 / num_nodes_slice if num_nodes_slice > 0 else 0.0
        y_positions = np.linspace(1 - padding, padding, num_nodes_slice) if num_nodes_slice > 1 else [0.5]

        # Build the nodes with manual x, y
        for rank_in_slice, i_original in enumerate(sorted_indices):
            node_label_str = sorted_text_labels[rank_in_slice]
            cluster_label_val = labels_this_slice[i_original]

            node_labels.append(node_label_str)
            node_idx_map[(slice_idx, i_original)] = current_node_idx

            node_colors.append(rgba_to_plotly_string(color_dict[cluster_label_val]))

            # Manually assign x from 0..1 across slices
            if N_slices > 1:
                node_x.append(slice_idx / (N_slices - 1))
            else:
                node_x.append(0.5)  # single slice corner case

            # y-coordinates from array above
            node_y.append(y_positions[rank_in_slice])
            current_node_idx += 1

    # 3) Build link definitions with sorted ordering
    for t_idx, T in enumerate(transition_list):
        r1, r2 = T.shape
        for i_row in range(r1):
            for j_col in range(r2):
                val = T[i_row, j_col]
                if val > threshold:
                    # Map original i_row -> rank_in_slice, then node_idx
                    sorted_source_idx = sorted_indices_list[t_idx].index(i_row)
                    sorted_target_idx = sorted_indices_list[t_idx + 1].index(j_col)

                    source_node = node_idx_map[(t_idx, i_row)]
                    target_node = node_idx_map[(t_idx + 1, j_col)]
                    link_sources.append(source_node)
                    link_targets.append(target_node)
                    link_values.append(val * linethick_factor)

    # 4) Create Sankey with manual x,y
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values
        )
    ))

    # 5) Layout
    fig.update_layout(
        title_text=title if title else "Differentiation Map (Sorted)",
        font_size=24,
        height=plot_height,
        width=plot_width
    )

    # 6) Save or show
    if save_as_svg and save_name is not None:
        fig.write_image(f"{save_name}.svg")
    elif save_as_svg and save_name is None:
        fig.write_image("diffmap.svg")

    fig.show()


def diffmap_from_QT_sankey(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
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
    cmap: str = "tab",
    order: bool = False
) -> None:
    """
    Create and display a Sankey diagram representing cluster transitions across
    multiple slices (defined by Qs and Ts). Depending on the `clustering_type`, 
    it uses either max-likelihood clustering or reference clustering.

    It then calls one of two plotting functions:
      - `plot_labeled_differentiation_sankey_sorted` (if `order=True`),
      - `plot_labeled_differentiation_sankey` otherwise.
    Both functions produce Sankey diagrams in Plotly, optionally sorted 
    alphabetically by provided cell type labels.

    Args:
        Qs (List[np.ndarray]):
            A list of length N, where each Qs[t] is shaped (n_t, r_t).
            Each represents probabilities or distributions for slice t.
        Ts (List[np.ndarray]):
            A list of length (N-1), where each Ts[t] is shaped (r_t, r_{t+1}),
            representing transitions between consecutive slices t and t+1.
        cell_type_labels (List[Optional[List[str]]] | None, optional):
            A list of length N, where each element is either None or a list of 
            strings labeling the clusters in that slice. Defaults to None.
        clustering_type (str, optional):
            One of:
              - "ml": (default) uses `max_likelihood_clustering(Qs)`.
              - "reference": uses `reference_clustering(Qs, Ts, reference_index)`.
        reference_index (int | None, optional):
            Required if clustering_type == "reference". Specifies the reference slice.
            Defaults to None.
        title (str, optional):
            Title for the resulting diagram. Defaults to None.
        save_name (str, optional):
            If provided, saves the resulting diagram under this file name/path. 
            Defaults to None.
        dsf (float, optional):
            A dotsize scaling factor (passed to the Sankey function, 
            e.g. might scale node size). Defaults to 1.0.
        plot_height (int, optional):
            The Plotly figure height in pixels. Defaults to 600.
        plot_width (int, optional):
            The Plotly figure width in pixels. Defaults to 1000.
        save_as_svg (bool, optional):
            If True, saves the figure as an SVG. Otherwise, no figure is saved 
            (unless the Sankey plotting function defaults otherwise). Defaults to True.
        threshold (float, optional):
            Minimum transition value to include (flows below this are omitted). 
            Defaults to 0.0.
        cmap (str, optional):
            Colormap name (passed down to `get_diffmap_inputs` for cluster colors).
            Defaults to "tab".
        order (bool, optional):
            If True, calls the "sorted" Sankey function, which sorts cluster labels
            alphabetically in each slice. Otherwise, uses the simpler Sankey. 
            Defaults to False.

    Returns:
        None. Plots and optionally saves a Plotly Sankey diagram.

    Raises:
        ValueError:
            If `clustering_type` is invalid or if `reference_index` is required 
            but not provided.
    """
    # 1) Build the clustering list from Qs (and Ts if reference-based)
    if clustering_type == "ml":
        clustering_list = max_likelihood_clustering(Qs)
    elif clustering_type == "reference":
        if reference_index is None:
            raise ValueError("A reference index is required for reference clustering.")
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError(f"Invalid clustering type: {clustering_type}")

    # 2) Generate population_list, labels_list, color_dict
    population_list, labels_list, color_dict = get_diffmap_inputs(
        clustering_list,
        clustering_type,
        cmap=cmap
    )

    # 3) Determine which Sankey plotting function to call
    if order:
        plot_labeled_differentiation_sankey_sorted(
            population_list,
            Ts,
            labels_list,
            color_dict,
            cell_type_labels=cell_type_labels,
            clustering_type=clustering_type,
            dotsize_factor=dsf,
            plot_height=plot_height,
            plot_width=plot_width,
            title=title,
            save_name=save_name,
            save_as_svg=save_as_svg,
            threshold=threshold
        )
    else:
        plot_labeled_differentiation_sankey(
            population_list,
            Ts,
            labels_list,
            color_dict,
            cell_type_labels=cell_type_labels,
            clustering_type=clustering_type,
            dotsize_factor=dsf,
            linethick_factor=10,
            plot_height=plot_height,
            plot_width=plot_width,
            title=title,
            save_name=save_name,
            save_as_svg=save_as_svg,
            threshold=threshold
        )


################################################################################################
#   older plotting code for making "pairwise" Sankey diagram
################################################################################################

def make_sankey(
    gt_clustering: Union[List[int], np.ndarray, pd.Series],
    pred_clustering: Union[List[int], np.ndarray, pd.Series],
    gt_labels: List[str],
    save_format: str = 'jpg',
    save_name: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Generate and display a Sankey diagram showing the transitions from
    ground-truth clusters to predicted clusters.

    The function also saves the diagram to disk if a file name is provided.

    Args:
        gt_clustering (List[int] | np.ndarray | pd.Series):
            Ground-truth cluster assignments.
        pred_clustering (List[int] | np.ndarray | pd.Series):
            Predicted cluster assignments.
        gt_labels (List[str]):
            Labels or names corresponding to each ground-truth cluster index (in sorted order).
        save_format (str, optional):
            Format in which to save the figure. Supported values are 'jpg', 'pdf', 'svg', or 'png'.
            Default is 'jpg'.
        save_name (str, optional):
            Name (or path) for the saved figure. If None, defaults to 'sankey_diagram'.
        title (str, optional):
            Title to display on the Sankey diagram. If None, defaults to
            "Cluster Transition Sankey Diagram".

    Returns:
        None. Displays an interactive Sankey diagram in a Plotly figure window
        and optionally saves it to disk in the specified format.

    Notes:
        - Adjust the `threshold` variable in the code if you wish to remove transitions
          below a certain size.
        - This function requires Plotly and Matplotlib to be installed.
    """

    # Combine ground-truth (GT) and predicted clusters in a DataFrame
    df = pd.DataFrame({'GT clusters': gt_clustering, 'Predicted clusters': pred_clustering})
    transition_matrix = pd.crosstab(df['GT clusters'], df['Predicted clusters'])

    # Sort cluster indices for consistent ordering
    gt_clusters = sorted(df['GT clusters'].unique())
    pred_clusters = sorted(df['Predicted clusters'].unique())

    # Node labels: ground-truth labels followed by predicted cluster labels
    labels = gt_labels + [f'Predicted Cluster {i}' for i in pred_clusters]

    # Number of clusters
    num_gt_clusters = len(gt_clusters)
    num_pred_clusters = len(pred_clusters)

    # Helper function to create colors from a Matplotlib colormap
    def generate_colors(num_colors: int, colormap_name: str) -> List[str]:
        cmap = plt.get_cmap(colormap_name)
        colors_array = cmap(np.linspace(0, 1, num_colors))
        # Convert RGBA floats [0,1] to hex string
        return [
            '#{:02x}{:02x}{:02x}'.format(
                int(r * 255), int(g * 255), int(b * 255)
            ) for r, g, b, _ in colors_array
        ]

    # Generate distinct color sets
    gt_colors = generate_colors(num_gt_clusters, 'Blues')
    pred_colors = generate_colors(num_pred_clusters, 'Oranges')
    node_colors = gt_colors + pred_colors

    # Build link connections
    threshold = 0  # Increase to hide small transitions
    source_indices = []
    target_indices = []
    values = []

    for gt_idx, gt_val in enumerate(gt_clusters):
        for pred_idx, pred_val in enumerate(pred_clusters):
            if gt_val in transition_matrix.index and pred_val in transition_matrix.columns:
                count = transition_matrix.at[gt_val, pred_val]
                if count > threshold:
                    # Source is GT cluster index
                    source_indices.append(gt_idx)
                    # Target is predicted cluster index, offset by num_gt_clusters
                    target_indices.append(pred_idx + num_gt_clusters)
                    values.append(count)

    # Construct the Sankey figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values
        )
    )])

    # Determine the title
    title_text = title if title else "Cluster Transition Sankey Diagram"

    # Update layout
    fig.update_layout(
        title_text=title_text,
        font_size=10,
        width=1000,  # Adjust as needed
        height=800   # Adjust as needed
    )

    # Display the diagram in the notebook or interactive environment
    fig.show()

    # Handle saving
    save_basename = save_name if save_name else 'sankey_diagram'
    
    # Choose file extension for the output
    if save_format == 'jpg':
        fig.write_image(save_basename + '.jpg')
    elif save_format == 'pdf':
        fig.write_image(save_basename + '.pdf')
    elif save_format == 'svg':
        fig.write_image(save_basename + '.svg')
    else:
        # Default to PNG if not specified or recognized
        fig.write_image(save_basename + '.png')