import numpy as np
import torch
from scipy.optimize import linprog
import pandas as pd

import networkx as nx
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# in nb: !pip install -U plotly -q   # for making Sankey diagram
# in nb: !pip install -U kaleido -q   # for saving Sankey diagram

################################################################################################
#   misc
################################################################################################

def factor_mats(C, A, B, device, z=None, c=100, nidx_1=None, nidx_2=None):    
    norm1 = c
    norm2 = A.max()*c
    norm3 = B.max()*c
    
    if z is None:
        # No low-rank factorization applied to the distance matrix
        A = torch.from_numpy(A).to(device)
        B = torch.from_numpy(B).to(device)
        C_factors = (C/ (norm1), torch.eye(C.shape[1]).type(torch.DoubleTensor).to(device))
        A_factors = (A/ (norm2), torch.eye(A.shape[1]).type(torch.DoubleTensor).to(device))
        B_factors = (B/ (norm3), torch.eye(B.shape[1]).type(torch.DoubleTensor).to(device))

    else:
        # Distance matrix factored using SVD
        u, s, v = torch.svd(torch.from_numpy(C).to(device))
        print('C done')
        V_C,U_C = torch.mm(u[:,:z], torch.diag(s[:z])), v[:,:z].mT
        u, s, v = torch.svd(torch.from_numpy(A).to(device))
        print('A done')
        V1_A,V1_B = torch.mm(u[:,:z], torch.diag(s[:z])), v[:,:z].mT
        u, s, v = torch.svd(torch.from_numpy(B).to(device))
        print('B done')
        V2_A,V2_B = torch.mm(u[:,:z], torch.diag(s[:z])), v[:,:z].mT
        C_factors, A_factors, B_factors = ((V_C.type(torch.DoubleTensor).to(device)/norm1, U_C.type(torch.DoubleTensor).to(device)/norm1), \
                                       (V1_A.type(torch.DoubleTensor).to(device)/norm2, V1_B.type(torch.DoubleTensor).to(device)/norm2), \
                                       (V2_A.type(torch.DoubleTensor).to(device)/norm3, V2_B.type(torch.DoubleTensor).to(device)/norm3))
    
    return C_factors, A_factors, B_factors

################################################################################################
#   magic cost matrix
################################################################################################

def make_graph_from_coords(S, n_neighbors=4, draw_graph=False):
    G = kneighbors_graph(X=S,
                         n_neighbors=n_neighbors,
                         mode='connectivity',
                         metric='minkowski',
                         p=2,
                         metric_params=None,
                         include_self=False,
                         n_jobs=None)
    G = nx.from_scipy_sparse_array(G)

    if draw_graph:
        nx.draw(G,
                pos=list(S),
                with_labels=False,
                node_size=5)
    else: pass

    return G


def make_Lap_ei(S, X, n_neighbors=4, draw_graph=False):
    G = make_graph_from_coords(S=S, n_neighbors=n_neighbors,draw_graph=draw_graph)
    Adj = nx.adjacency_matrix(G).todense()

    A = distance.cdist(X,X)
    wAdj = A * Adj
    wAdj_row_sums = np.sum(wAdj, axis=1)
    D = np.diag(wAdj_row_sums)
    Lap = D = wAdj

    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    Lap_star = D_inv_sqrt @ Lap @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(Lap_star) 
    return eigenvectors, eigenvalues


def HDM(spot_index, eigenvectors, eigenvalues, truncation=None, time=10.):
    """
    Returns embedding of spot into R^{n-1}, where
    n is the number of eigenvalues, as well as the dimension of the eigenvectors

    Args:
        spot_index: int, index of spot in synthetic tissue
        eigenvectors: np.ndarray, eigenvectors of normalized sheaf Laplacian
        eigenvalues: np.ndarray, eigenvalues of normalized sheaf Laplacian
        truncation: int or None, optional truncation of eigenvalues/vectors
        time: float, "diffusion time"

    Returns:
        HDM_vec: np.ndarray, of length n-1 (or truncation-1)
    """
    n = eigenvectors.shape[1]
    if truncation is None:
        ell_range = np.arange(1, n)
    else:
        ell_range = np.arange(1, truncation)

    # Compute the eigenvalues raised to the power of 'time'
    evals = eigenvalues[ell_range] ** time  # Shape: (n-1,) or (truncation-1,)

    # Get the eigenvector entries for the given spot_index
    entries = eigenvectors[spot_index, ell_range]  # Shape: (n-1,) or (truncation-1,)

    # Element-wise multiplication
    HDM_vec = evals * entries  # Shape: (n-1,) or (truncation-1,)

    return HDM_vec


def HDM_representation(S, eigenvectors, eigenvalues, truncation=None, time=10.):
    """
    Returns HDM embedding stacked across all slice spots

    Args:
        S: np.ndarray, stack of n slice spatial coordinates
        eigenvectors: np.ndarray, eigenvectors of normalized sheaf Laplacian
        eigenvalues: np.ndarray, eigenvalues of normalized sheaf Laplacian
        truncation: int or None, optional truncation of eigenvalues/vectors
        time: float, "diffusion time"

    Returns:
        HDM_stack: np.ndarray, stack of n HDM representations, each
                   of which is a row-vector of length n - 1 (or truncation - 1)
    """
    n = eigenvectors.shape[0]
    if truncation is None:
        ell_range = np.arange(1, eigenvectors.shape[1])
    else:
        ell_range = np.arange(1, truncation)

    # Compute the eigenvalues raised to the power of 'time'
    evals = eigenvalues[ell_range] ** time  # Shape: (n-1,) or (truncation-1,)

    # Get the eigenvector entries for all spots and selected eigenvalues
    entries = eigenvectors[:, ell_range]  # Shape: (n, n-1) or (n, truncation-1)

    # Multiply each column (eigenvector) by the corresponding eigenvalue
    HDM_stack = entries * evals  # Broadcasting over columns

    return HDM_stack  # Shape: (n, n-1) or (n, truncation-1)

def HDM_from_XS(S, X, n_neighbors=4, truncation=100, time=10.):
    eigenvectors, eigenvalues = make_Lap_ei(S, X, n_neighbors)
    HDM_stack = HDM_representation(S, eigenvectors, eigenvalues, truncation, time)
    return HDM_stack


################################################################################################
#   for Sankey diagrams
################################################################################################

def make_sankey(gt_clustering, pred_clustering, gt_labels, save_format='jpg', save_name=None, title=None):
    df1 = pd.DataFrame({'GT clusters': gt_clustering, 'Predicted clusters': pred_clustering})
    transition_matrix = pd.crosstab(df1['GT clusters'], df1['Predicted clusters'])


    # Get unique cluster labels
    gt_clusters = sorted(df1['GT clusters'].unique())
    pred_clusters = sorted(df1['Predicted clusters'].unique())

    # Define node labels
    labels = gt_labels + [f'Predicted Cluster {i}' for i in pred_clusters]


    # Number of clusters
    num_gt_clusters = len(gt_clusters)
    num_pred_clusters = len(pred_clusters)

    # Function to generate colors
    def generate_colors(num_colors, colormap_name):
        cmap = plt.get_cmap(colormap_name)
        colors = cmap(np.linspace(0, 1, num_colors))
        return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors]

    # Generate colors for clusters
    gt_colors = generate_colors(num_gt_clusters, 'Blues')       # Colormap for GT clusters
    pred_colors = generate_colors(num_pred_clusters, 'Oranges') # Colormap for Predicted clusters

    # Combine colors
    node_colors = gt_colors + pred_colors

    # Initialize lists for sources, targets, and values
    threshold = 0   # NOTE: changing threshold will disappear small pop cell types in GT
    source_indices = []
    target_indices = []
    values = []

    for gt_idx, gt_cluster in enumerate(gt_clusters):
        for pred_idx, pred_cluster in enumerate(pred_clusters):
            if gt_cluster in transition_matrix.index and pred_cluster in transition_matrix.columns:
                count = transition_matrix.at[gt_cluster, pred_cluster]
                if count > threshold:
                    source_indices.append(gt_idx)
                    target_indices.append(pred_idx + num_gt_clusters)
                    values.append(count)


    # Create the Sankey diagram
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
    if title is not None:
        title_text=title
    else:
        title_text="Cluster Transition Sankey Diagram"

    # Update layout
    fig.update_layout(
        title_text=title_text,
        font_size=10,
        width=1000,    # Adjust width as needed
        height=800     # Adjust height as needed
    )

    # Display the diagram
    fig.show()

    if save_name is not None:
        save=save_name
    else:
        save='sankey_diagram'
    
    if save_format=='jpg':
        # Export as JPEG
        fig.write_image(save+'.jpg')
    elif save_format=='pdf':
        # Export as PDF
        fig.write_image(save+'.pdf')
    elif save_format=='svg':
        # Export as SVG
        fig.write_image(save+'.svg')
    else:
        # Export as PNG (default if no extension is specified)
        fig.write_image(save+'.png')