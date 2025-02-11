import numpy as np
import torch
import pandas as pd

from scipy.stats import entropy

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Optional, Tuple, Union


################################################################################################
#   optional low-rank factorization, for cost matrices small enough to instantiate
################################################################################################


def factor_mats_tens(
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    device: torch.device,
    z: Optional[int] = None,
    c: float = 1.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor]]:
    """
    Scales and (optionally) factorizes three matrices C, A, and B using PyTorch's SVD routines.

    Args:
        C (torch.Tensor): Matrix (e.g., distance matrix) to factorize or pass through.
        A (torch.Tensor): Matrix to factorize or pass through.
        B (torch.Tensor): Matrix to factorize or pass through.
        device (torch.device): Target device to place the resulting tensors on (e.g., 'cpu' or 'cuda').
        z (int, optional): Rank for the truncated SVD. If None, no SVD is applied and the
            matrices are simply scaled and paired with an identity matrix.
        c (float, optional): Scaling constant applied to each matrix. Default is 1.0.

    Returns:
        (C_factors, A_factors, B_factors) (Tuple[Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        Each element of the tuple is itself a two-tuple containing:
            1. The left factor (or scaled full matrix if z is None).
            2. The right factor (or identity if z is None).

        Specifically:
            - If z is None:
                C_factors = (C / c, I)          # same for A, B
            - Otherwise, after SVD-based factorization to rank z:
                C_factors = (U Σ_z / c, V_z / c)

        where U, Σ_z, and V_z come from the truncated SVD of the matrix.

    Example:
        >>> C = torch.randn(100, 100)
        >>> A = torch.randn(100, 100)
        >>> B = torch.randn(100, 100)
        >>> device = torch.device("cpu")
        >>> # No factorization
        >>> C_factors, A_factors, B_factors = factor_mats_tens(C, A, B, device, z=None, c=2.0)
        >>> # With rank-10 factorization
        >>> C_factors, A_factors, B_factors = factor_mats_tens(C, A, B, device, z=10, c=2.0)
    """

    # Compute scaling norms
    norm1 = c
    norm2 = A.max() * c
    norm3 = B.max() * c

    if z is None:
        # No low-rank factorization: just scale and pair with identity
        C_factors = (C.to(device) / norm1,
                     torch.eye(C.shape[1], dtype=torch.double, device=device))
        A_factors = (A.to(device) / norm2,
                     torch.eye(A.shape[1], dtype=torch.double, device=device))
        B_factors = (B.to(device) / norm3,
                     torch.eye(B.shape[1], dtype=torch.double, device=device))
    else:
        # SVD-based factorization, truncated to rank z

        # Factor C
        u_c, s_c, v_c = torch.linalg.svd(C.to(device), full_matrices=False)
        V_C, U_C = torch.mm(u_c[:, :z], torch.diag(s_c[:z])), v_c[:, :z]

        # Factor A
        u_a, s_a, v_a = torch.svd(A.to(device))
        V1_A, V1_B = torch.mm(u_a[:, :z], torch.diag(s_a[:z])), v_a[:, :z]

        # Factor B
        u_b, s_b, v_b = torch.svd(B.to(device))
        V2_A, V2_B = torch.mm(u_b[:, :z], torch.diag(s_b[:z])), v_b[:, :z]

        # Scale each factor
        C_factors = (V_C / norm1, U_C / norm1)
        A_factors = (V1_A / norm2, V2_A / norm2)
        B_factors = (V1_B / norm3, V2_B / norm3)

        print("C done")

    return C_factors, A_factors, B_factors




################################################################################################
#   Sankey diagram
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


################################################################################################
#   entropy
################################################################################################

def compute_transition_entropy(
    matrix: Union[np.ndarray, list],
    mode: str = "row",
    epsilon: float = 1e-12
) -> float:
    """
    Compute the entropy of a transition matrix, either row-based,
    column-based, or flattened.

    Args:
        matrix (np.ndarray | list):
            Transition matrix or any 2D numeric data.
        mode (str, optional):
            Specifies how to compute entropy:
              - 'row':    Each row is normalized (row-stochastic), and
                          row-wise entropies are summed.
              - 'column': Each column is normalized (column-stochastic), and
                          column-wise entropies are summed.
              - 'flat':   No row/column normalization. Compute entropy on the
                          entire matrix flattened into 1D.
            Default is 'row'.
        epsilon (float, optional):
            A small constant added to avoid log(0). Default is 1e-12.

    Returns:
        float:
            The sum (or total) of entropies according to the chosen mode.
    """
    # Convert input to NumPy array
    matrix = np.array(matrix, dtype=float)

    # Edge case: if the matrix is empty or 1D
    if matrix.ndim < 2:
        # For safety, treat as "flatten"
        matrix = matrix.ravel()
        # Add epsilon to avoid log(0)
        matrix_nonzero = np.where(matrix == 0, epsilon, matrix)
        # Compute standard 1D entropy
        matrix_nonzero /= matrix_nonzero.sum()  # normalize
        return -np.sum(matrix_nonzero * np.log(matrix_nonzero))

    if mode == "flat":
        # Flatten out the entire matrix
        flat_matrix = matrix.ravel()
        flat_matrix_nonzero = np.where(flat_matrix == 0, epsilon, flat_matrix)
        # Normalize
        flat_matrix_nonzero /= flat_matrix_nonzero.sum()
        # Entropy over the entire distribution
        return -np.sum(flat_matrix_nonzero * np.log(flat_matrix_nonzero))

    elif mode == "row":
        # Row-based entropy
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid divide-by-zero
        row_sums[row_sums == 0] = epsilon
        row_normed = matrix / row_sums

        # Add epsilon to each cell to avoid log(0)
        row_nonzero = np.where(row_normed == 0, epsilon, row_normed)

        # Sum of entropies across rows
        entropy_sum = 0.0
        for i in range(row_nonzero.shape[0]):
            entropy_sum += -np.sum(row_nonzero[i] * np.log(row_nonzero[i]))
        return entropy_sum

    elif mode == "column":
        # Column-based entropy
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = epsilon
        col_normed = matrix / col_sums

        col_nonzero = np.where(col_normed == 0, epsilon, col_normed)

        # Sum of entropies across columns
        entropy_sum = 0.0
        for j in range(col_nonzero.shape[1]):
            entropy_sum += -np.sum(col_nonzero[:, j] * np.log(col_nonzero[:, j]))
        return entropy_sum

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'row', 'column', or 'flat'.")
    

def compare_transition_entropies(
    Ts_ann: List[np.ndarray],
    Ts_pred: List[np.ndarray],
    mode: str = "row"
) -> None:
    """
    Compare entropy between two lists of transition matrices (annotated vs. predicted),
    using one of three modes: 'row', 'column', or 'flat'.

    For each pair (T_ann, T_pred) in (Ts_ann, Ts_pred), compute the chosen
    transition entropy and print which one is larger.

    Args:
        Ts_ann (List[np.ndarray]):
            List of "annotated" transition matrices (often ground-truth).
        Ts_pred (List[np.ndarray]):
            List of "predicted" transition matrices (often model output).
        mode (str, optional):
            - 'row': row-based entropy
            - 'column': column-based entropy
            - 'flat': flatten the matrix and compute standard 1D entropy
            Default is 'row'.

    Returns:
        None. Prints a line for each comparison.
    """
    if len(Ts_ann) != len(Ts_pred):
        raise ValueError("Length mismatch: Ts_ann and Ts_pred must have the same length.")

    for i, (T_ann, T_pred) in enumerate(zip(Ts_ann, Ts_pred)):
        # Convert annotated to NumPy if it's a PyTorch tensor
        if hasattr(T_ann, "cpu") and callable(T_ann.cpu):
            T_ann = T_ann.cpu().numpy()

        # Convert predicted to NumPy if needed
        if hasattr(T_pred, "cpu") and callable(T_pred.cpu):
            T_pred = T_pred.cpu().numpy()

        ent_ann = compute_transition_entropy(T_ann, mode=mode)
        ent_pred = compute_transition_entropy(T_pred, mode=mode)

        if ent_pred > ent_ann:
            print(
                f"Pred transitions {i} -> {i+1} are MORE entropic in {mode} mode: "
                f"{ent_pred:.3f} > {ent_ann:.3f}"
            )
        else:
            print(
                f"Pred transitions {i} -> {i+1} are LESS entropic in {mode} mode: "
                f"{ent_pred:.3f} < {ent_ann:.3f}"
            )


################################################################################################
#   ARI, AMI
################################################################################################

def compute_ARI_and_AMI(
    gt_types_list: List[List[int]],
    pred_types_list: List[List[int]],
    x_percent: float = 5.0
) -> None:
    """
    Compute Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
    on pairs of ground-truth and predicted cluster assignments. Also compute
    "filtered" ARI and AMI by removing any ground-truth clusters that occupy
    fewer than x_percent of the total data points.

    For each pair (gt_types, pred_types) in the provided lists:

    1. Compute the raw ARI and AMI.
    2. Filter out all data points whose ground-truth cluster is smaller than
       x_percent of the data.
    3. Recompute ARI and AMI on this filtered subset.
    4. Print both the raw and filtered values.

    Args:
        gt_types_list (List[List[int]]):
            A list of lists. Each inner list contains the ground-truth cluster
            labels for one "slice" or partition of data.
        pred_types_list (List[List[int]]):
            A list of lists containing predicted cluster labels, corresponding
            one-to-one with gt_types_list.
        x_percent (float, optional):
            The minimum cluster size threshold (as a percent) below which a
            cluster is ignored. Default is 5.0.

    Returns:
        None. This function prints ARI and AMI values for each slice.
    
    Example:
        >>> # Suppose we have two slices of data, each with ground-truth and predicted clusters
        >>> gt_slices = [[0, 0, 1, 1, 2], [0, 1, 1, 1, 2]]
        >>> pred_slices = [[0, 0, 2, 1, 1], [0, 1, 1, 2, 2]]
        >>> compute_ARI_and_AMI(gt_slices, pred_slices, x_percent=10)
    """
    print(
        f"ARI and AMI of predictions (filtered excludes ground-truth clusters "
        f"smaller than {x_percent}% of the data)\n"
    )

    # Ensure the two lists match in length
    if len(gt_types_list) != len(pred_types_list):
        raise ValueError("gt_types_list and pred_types_list must have the same length.")

    # Iterate over each pair of GT/pred type lists
    for i, (gt_types, pred_types) in enumerate(zip(gt_types_list, pred_types_list)):
        # Convert to NumPy arrays for easier filtering
        gt_labels = np.array(gt_types)
        pred_labels = np.array(pred_types)

        # Compute raw ARI, AMI
        raw_ari = ari(gt_labels, pred_labels)
        raw_ami = ami(gt_labels, pred_labels)

        total_points = len(gt_labels)
        if total_points == 0:
            print(f"No data in slice {i}; skipping.\n")
            continue

        # Count occurrences of each ground-truth cluster
        unique_labels, counts = np.unique(gt_labels, return_counts=True)
        percentages = counts / total_points * 100

        # Identify clusters that meet the x_percent threshold
        clusters_to_keep = unique_labels[percentages >= x_percent]

        # Filter out all points not in the "kept" clusters
        mask = np.isin(gt_labels, clusters_to_keep)
        gt_labels_filtered = gt_labels[mask]
        pred_labels_filtered = pred_labels[mask]

        # Compute filtered ARI, AMI
        filtered_ari = ari(gt_labels_filtered, pred_labels_filtered)
        filtered_ami = ami(gt_labels_filtered, pred_labels_filtered)

        # Print out results
        print(
            f"Slice {i}:"
            f"\n  ARI = {raw_ari:.3f} (filtered: {filtered_ari:.3f})"
            f"\n  AMI = {raw_ami:.3f} (filtered: {filtered_ami:.3f})\n"
        )