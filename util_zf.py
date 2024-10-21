import numpy as np
import torch
from scipy.optimize import linprog
import pandas as pd

import networkx as nx
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance
from scipy.stats import entropy
from scipy import sparse
from scipy.sparse import linalg as splinalg

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


import matplotlib.pyplot as plt
import plotly.graph_objects as go

# in nb: !pip install -U plotly -q   # for making Sankey diagram
# in nb: !pip install -U kaleido -q   # for saving Sankey diagram

################################################################################################
#   misc
################################################################################################

def factor_mats(C, A, B, device, z=None, c=1, nidx_1=None, nidx_2=None):    
    norm1 = c
    norm2 = A.max()*c
    norm3 = B.max()*c
    
    if z is None:
        # No low-rank factorization applied to the distance matrix
        A = torch.from_numpy(A).to(device)
        B = torch.from_numpy(B).to(device)
        C_factors = (torch.from_numpy(C).to(device)/ (norm1), torch.eye(C.shape[1]).type(torch.DoubleTensor).to(device))
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

def factor_mats_for_sc(C, A, B, device, z=None, c=100, nidx_1=None, nidx_2=None):    
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
#   HDM representations
################################################################################################

def make_graph_from_coords(S, n_neighbors=4, draw_graph=False):
    # Use 'distance' mode to get distances as weights
    G_sparse = kneighbors_graph(X=S,
                         n_neighbors=n_neighbors,
                         mode='distance',
                         metric='minkowski',
                         p=2,
                         include_self=False,
                         n_jobs=-1)  # Use all available cores for speed
    G_nx = nx.from_scipy_sparse_array(G_sparse)

    if draw_graph:        
        nx.draw(G_nx,
                pos=S,
                with_labels=False,
                node_size=5)
    # Return the sparse weighted adjacency matrix directly
    return G_nx

def make_Lap_ei(S, X, n_neighbors=4, draw_graph=False, num_eigvals=100, MELD=False, MELD_eps=1e-2):
    G = make_graph_from_coords(S=S, n_neighbors=n_neighbors, draw_graph=draw_graph)
    
    Adj = nx.to_numpy_array(G) # convert to array
    A = distance.cdist(X, X) # make feature-based distance matrix

    # Compute the weighted degree matrix D
    wAdj = A * Adj # hadamard feature distances with spatial adjancency matrix
    wAdj_row_sums = np.array(wAdj.sum(axis=1))
    D = np.diag(wAdj_row_sums)

    # Compute the unnormalized Laplacian
    Lap = D - wAdj

    D_inv_sqrt = np.diag(1.0 / np.sqrt(wAdj_row_sums + 1e-10))  # Add epsilon to prevent division by zero

    Lap_star = D_inv_sqrt @ Lap @ D_inv_sqrt

    # Ensure the matrix is symmetric
    Lap_star = (Lap_star + Lap_star.T) / 2

    n = Lap_star.shape[0]
    k = min(num_eigvals, n - 2)  # Adjust k based on the size of the matrix
    eigenvalues, eigenvectors = np.linalg.eigh(Lap_star)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lambda_max = eigenvalues[-1]

    if MELD:
        eigenvalues = np.exp(-eigenvalues / (lambda_max * MELD_eps))
    else:
        eigenvalues = eigenvalues / lambda_max
    return eigenvectors, eigenvalues

def HDM(spot_index, eigenvectors, eigenvalues, truncation=None, time=10.):
    n = eigenvectors.shape[1]
    if truncation is None:
        ell_range = np.arange(1, n)
    else:
        ell_range = np.arange(1, min(truncation, n))

    # Compute the eigenvalues raised to the power of 'time'
    evals = np.power(eigenvalues[ell_range], time)  # Use np.power for stability

    # Get the eigenvector entries for the given spot_index
    entries = eigenvectors[spot_index, ell_range]

    # Element-wise multiplication
    HDM_vec = evals * entries

    return HDM_vec

def HDM_representation(S, eigenvectors, eigenvalues, truncation=None, time=10.):
    if truncation is None:
        ell_range = np.arange(1, eigenvectors.shape[1], dtype=int)  # Ensuring dtype=int
    else:
        ell_range = np.arange(1, min(truncation, eigenvectors.shape[1]), dtype=int)  # Ensuring dtype=int

    # Compute the eigenvalues raised to the power of 'time'
    evals = np.power(eigenvalues[ell_range], time)

    # Handle potential negative or zero eigenvalues due to numerical errors
    evals = np.where(evals > 0, evals, 1e-10)

    # Get the eigenvector entries for all spots
    entries = eigenvectors[:, ell_range]

    # Multiply each column (eigenvector) by the corresponding eigenvalue
    HDM_stack = entries * evals

    return HDM_stack

def HDM_from_XS(S, X, n_neighbors=4, truncation=100, time=10., MELD=False, MELD_eps=1e-2):
    eigenvectors, eigenvalues = make_Lap_ei(S, X, n_neighbors, num_eigvals=truncation, MELD=MELD, MELD_eps=MELD_eps)
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



################################################################################################
#   computing entropy of latent coupling matrices
################################################################################################

def compute_column_entropy(gamma):
    # Ensure gamma is a NumPy array
    gamma = np.array(gamma)
    g = np.sum(gamma, axis=0)
    gamma = gamma @ np.diag(1 / g)
    # Avoid log(0) by adding a small epsilon where gamma is zero
    epsilon = 1e-12
    gamma_nonzero = gamma + (gamma == 0) * epsilon
    col_entropy_avg = 0
    for i in range(gamma.shape[1]):
        col_entropy_avg += -np.sum(gamma[:,i] * np.log(gamma_nonzero[:,i]))
    return col_entropy_avg

def compute_row_entropy(gamma):
    # Ensure gamma is a NumPy array
    gamma = np.array(gamma)
    # Sum along the rows
    row_sums = np.sum(gamma, axis=1)
    # Normalize to make rows sum to 1 (row-stochastic)
    gamma = np.diag(1 / row_sums) @ gamma
    # Avoid log(0) by adding a small epsilon where gamma is zero
    epsilon = 1e-12
    gamma_nonzero = gamma + (gamma == 0) * epsilon
    row_entropy_avg = 0
    # Compute the entropy for each row and sum
    for i in range(gamma.shape[0]):
        row_entropy_avg += -np.sum(gamma[i, :] * np.log(gamma_nonzero[i, :]))
    return row_entropy_avg

def compare_T_entropies(Ts_ann, Ts_pred):
    for i, T_pair in enumerate(zip(Ts_ann, Ts_pred)):
        T_ann, T_pred = T_pair
        ent_ann = entropy(T_ann.cpu().numpy().flatten())
        ent_pred = entropy(T_pred.flatten())
        if ent_pred > ent_ann:
            print(f'Pred transitions {i} -> {i+1} are **MORE** entropic: {ent_pred:.3f} > {ent_ann:.3f}')
        else:
            print(f'Pred transitions {i} to {i+1} are **LESS** entropic: {ent_pred:.3f} < {ent_ann:.3f}') 

def compare_T_col_entropies(Ts_ann, Ts_pred):
    for i, T_pair in enumerate(zip(Ts_ann, Ts_pred)):
        T_ann, T_pred = T_pair
        ent_ann = compute_column_entropy(T_ann.cpu().numpy())
        ent_pred = compute_column_entropy(T_pred)
        if ent_pred > ent_ann:
            print(f'Pred transitions {i} -> {i+1} have **MORE** column entropy: {ent_pred:.3f} > {ent_ann:.3f}')
        else:
            print(f'Pred transitions {i} -> {i+1} have **LESS** column entropy: {ent_pred:.3f} < {ent_ann:.3f}')

def compare_T_row_entropies(Ts_ann, Ts_pred):
    for i, T_pair in enumerate(zip(Ts_ann, Ts_pred)):
        T_ann, T_pred = T_pair
        ent_ann = compute_row_entropy(T_ann.cpu().numpy())
        ent_pred = compute_row_entropy(T_pred)
        if ent_pred > ent_ann:
            print(f'Pred transitions {i} -> {i+1} have **MORE** row entropy: {ent_pred:.3f} > {ent_ann:.3f}')
        else:
            print(f'Pred transitions {i} -> {i+1} have **LESS** row entropy: {ent_pred:.3f} < {ent_ann:.3f}')

################################################################################################
#   computing ARIs, AMIs
################################################################################################

def compute_ARI_and_AMI(gt_types_list, pred_types_list, x_percent=5):
    print(f"ARI and AMI of predictions (filtered excludes ground truth clusters smaller than {x_percent}% of the data)\n")
    for i, (gt_types, pred_types) in enumerate(zip(gt_types_list, pred_types_list)):
        raw_ari = ari(gt_types, pred_types)
        raw_ami = ami(gt_types, pred_types)

        gt_labels = np.array(gt_types)
        pred_labels = np.array(pred_types)
        total_points = len(gt_labels)

        # Compute counts of ground truth clusters
        unique_labels, counts = np.unique(gt_labels, return_counts=True)
        percentages = counts / total_points * 100

        # Identify clusters to keep (clusters with size >= x%)
        clusters_to_keep = unique_labels[percentages >= x_percent]

        # Create a mask to keep only data points in clusters_to_keep
        mask = np.isin(gt_labels, clusters_to_keep)

        # Apply mask to both gt_labels and pred_labels
        gt_labels_filtered = gt_labels[mask]
        pred_labels_filtered = pred_labels[mask]

        # Compute ARI and AMI on the filtered labels
        x_ari = ari(gt_labels_filtered, pred_labels_filtered)
        x_ami = ami(gt_labels_filtered, pred_labels_filtered)

        print(f'ARI for {i}th slice is {raw_ari:.3f} (filtered: {x_ari:.3f}) \t')
        print(f'AMI for {i}th slice is {raw_ami:.3f} (filtered: {x_ami:.3f})')
        print('\n')


################################################################################################
#   computing silhouette scores
################################################################################################

def silhouette(gt_types_list, pred_types_list, Xs, Ss):
    for i, pair in enumerate(zip(gt_types_list, pred_types_list)):
        gt_types, pred_types = pair
        expr_score_gt = silhouette_score(Xs[i], gt_types)
        expr_score_pred = silhouette_score(Xs[i], pred_types)
        if expr_score_pred > expr_score_gt:
            print(f'\tPred clusters {i} have **HIGHER** expression silhouette score: {expr_score_pred:.3f} > {expr_score_gt:.3f}')
        else:
            print(f'\tPred clusters {i} have **LOWER** expression silhouette score: {expr_score_pred:.3f} < {expr_score_gt:.3f}')
    
    print('\n')

    for i, pair in enumerate(zip(gt_types_list, pred_types_list)):
        gt_types, pred_types = pair
        spa_score_gt = silhouette_score(Ss[i], gt_types)
        spa_score_pred = silhouette_score(Xs[i], pred_types)
        if spa_score_pred > spa_score_gt:
            print(f'\tPred clusters {i} have **HIGHER** spatial silhouette score: {spa_score_pred:.3f} > {spa_score_gt:.3f}')
        else:
            print(f'\tPred clusters {i} have **LOWER** spatial silhouette score: {spa_score_pred:.3f} < {spa_score_gt:.3f}')

def cosine_silhouette_score(X, labels):
    """
    Compute the silhouette score for each sample in X using cosine similarity.
    
    Parameters:
    - X: Data matrix (n_samples, n_features)
    - labels: Cluster labels for each point in X
    
    Returns:
    - silhouette_avg: The average silhouette score for all samples.
    """
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(X)
    
    # Convert cosine similarity to a cosine "distance" (1 - similarity)
    cosine_dist = 1 - cosine_sim
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0  # Silhouette score is not defined in these cases
    
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Points in the same cluster as point i
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude the point itself
        
        if np.sum(same_cluster) > 0:
            # a(i) = mean distance to other points in the same cluster
            a_i = np.mean(cosine_dist[i, same_cluster])
        else:
            a_i = 0.0
        
        # b(i) = smallest mean distance to points in the nearest different cluster
        b_i = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = labels == label
                mean_dist_to_cluster = np.mean(cosine_dist[i, other_cluster])
                b_i = min(b_i, mean_dist_to_cluster)
        
        # Compute the silhouette score for point i
        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    # Average silhouette score over all samples
    silhouette_avg = np.mean(silhouette_scores)
    
    return silhouette_avg

def cos_silhouette(gt_types_list, pred_types_list, Xs, Ss):
    for i, pair in enumerate(zip(gt_types_list, pred_types_list)):
        gt_types, pred_types = pair
        expr_score_gt = cosine_silhouette_score(Xs[i], gt_types)
        expr_score_pred = cosine_silhouette_score(Xs[i], pred_types)
        if expr_score_pred > expr_score_gt:
            print(f'\tPred clusters {i} have **HIGHER** expression cosine-silhouette score: {expr_score_pred:.3f} > {expr_score_gt:.3f}')
        else:
            print(f'\tPred clusters {i} have **LOWER** expression cosine-silhouette score: {expr_score_pred:.3f} < {expr_score_gt:.3f}')
    
    print('\n')

    for i, pair in enumerate(zip(gt_types_list, pred_types_list)):
        gt_types, pred_types = pair
        spa_score_gt = silhouette_score(Ss[i], gt_types)
        spa_score_pred = silhouette_score(Xs[i], pred_types)
        if spa_score_pred > spa_score_gt:
            print(f'\tPred clusters {i} have **HIGHER** spatial silhouette score: {spa_score_pred:.3f} > {spa_score_gt:.3f}')
        else:
            print(f'\tPred clusters {i} have **LOWER** spatial silhouette score: {spa_score_pred:.3f} < {spa_score_gt:.3f}')

################################################################################################
# computing / plotting collision profiles
################################################################################################

def top_k_collision_ratio(T, k):
    # Flatten the matrix while keeping track of the original indices
    flattened_indices = [(i, j) for i in range(T.shape[0]) for j in range(T.shape[1])]
    flattened_values = T.flatten()

    # Get top k entries based on values in the matrix
    top_k_indices = np.argsort(flattened_values)[-k:][::-1]  # Top k indices (sorted in descending order)
    
    # Extract the columns of the top k entries
    top_k_columns = [flattened_indices[idx][1] for idx in top_k_indices]

    # Count collisions (entries sharing the same column)
    collision_count = len(top_k_columns) - len(set(top_k_columns))  # Subtract unique columns from total

    # Return the ratio (collision_count / k)
    return collision_count / k

def plot_collision_profile(T):
    # Get the total number of entries in T
    num_entries = T.size()
    
    # Initialize a list to store the ratio for each k
    ratios = []

    # Compute the ratio for each k from 1 to the total number of entries
    for k in range(1, int(num_entries) + 1):
        ratio = top_k_collision_ratio(T, k)
        ratios.append(ratio)

    # Plot the ratio as a function of k
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_entries + 1), ratios, marker='o', linestyle='-', color='b')
    plt.xlabel('k (Top-k Entries)')
    plt.ylabel('Collision Ratio')
    plt.title('Collision Profile of Matrix T')
    plt.grid(True)
    plt.show()

"""
def plot_collision_profiles(T1, T2, title, tolerance=0.01, consecutive_agreement=10, plot_ticks_step=5):
    # Get the total number of entries in T1 and T2 (assuming same shape for both)
    num_entries = T1.size

    # Initialize lists to store the ratios for each matrix
    ratios_T1 = []
    ratios_T2 = []

    # Compute the ratio for each k from 1 to the total number of entries
    for k in range(1, num_entries + 1):
        ratio_T1 = top_k_collision_ratio(T1, k)
        ratio_T2 = top_k_collision_ratio(T2, k)
        ratios_T1.append(ratio_T1)
        ratios_T2.append(ratio_T2)

    # Find the point where the profiles start to agree completely
    truncate_index = num_entries  # Default to the full range if no early stopping is found
    agreement_count = 0

    for k in range(1, num_entries):
        if abs(ratios_T1[k] - ratios_T2[k]) < tolerance:
            agreement_count += 1
            if agreement_count >= consecutive_agreement:
                truncate_index = k + 1  # We stop after finding `consecutive_agreement` steps of agreement
                break
        else:
            agreement_count = 0  # Reset if they disagree again

    # Truncate the profiles at the point of agreement
    k_range = range(1, truncate_index + 1)
    truncated_ratios_T1 = ratios_T1[:truncate_index]
    truncated_ratios_T2 = ratios_T2[:truncate_index]

    # Plot both profiles on the same graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, truncated_ratios_T1, marker='o', linestyle='-', color='b', label='Matrix T1')
    plt.plot(k_range, truncated_ratios_T2, marker='x', linestyle='--', color='r', label='Matrix T2')
    plt.xlabel('k (Top-k Entries)')
    plt.ylabel('Collision Ratio')
    plt.title(title)  # Use the title provided as an argument
    plt.grid(False)

    plt.xticks(np.arange(1, truncate_index + 1, step=plot_ticks_step)) # Adjust the step size for x-axis ticks

    plt.legend()
    plt.show()
"""

def plot_collision_profiles(list_of_Ts, title, tolerance=0.01, consecutive_agreement=10, plot_ticks_step=5, label_list=None):
    """
    Plots collision profiles for a list of matrices, comparing their top-k entry collision ratios.

    Parameters:
    - list_of_Ts: A list of matrices to compare.
    - title: Title for the plot.
    - tolerance: Tolerance for comparing the ratios of the matrices.
    - consecutive_agreement: The number of consecutive agreements needed to stop plotting early.
    - plot_ticks_step: Step size for x-axis ticks.
    """
    num_matrices = len(list_of_Ts)
    
    if num_matrices < 2:
        raise ValueError("At least two matrices are required for comparison.")
    
    # Get the total number of entries in the first matrix (assuming all matrices are the same size)
    num_entries = list_of_Ts[0].size

    # Initialize lists to store the ratios for each matrix
    all_ratios = []

    # Compute the ratio for each matrix for each k from 1 to the total number of entries
    for T in list_of_Ts:
        ratios = []
        for k in range(1, num_entries + 1):
            ratio = top_k_collision_ratio(T, k)
            ratios.append(ratio)
        all_ratios.append(ratios)

    # Find the point where all profiles start to agree completely
    truncate_index = num_entries  # Default to the full range if no early stopping is found
    agreement_count = 0

    for k in range(1, num_entries):
        # Check agreement across all matrices
        differences = [abs(all_ratios[i][k] - all_ratios[j][k]) for i in range(num_matrices) for j in range(i + 1, num_matrices)]
        
        if all(diff < tolerance for diff in differences):
            agreement_count += 1
            if agreement_count >= consecutive_agreement:
                truncate_index = k + 1  # Stop after finding `consecutive_agreement` steps of agreement
                break
        else:
            agreement_count = 0  # Reset if they disagree again

    # Truncate the profiles at the point of agreement
    k_range = range(1, truncate_index + 1)
    truncated_ratios = [ratios[:truncate_index] for ratios in all_ratios]

    # Plot the profiles for each matrix on the same graph
    plt.figure(figsize=(8, 6))

    if label_list is None:
        label_list = [f'Matrix T{i+1}' for i, tr in enumerate(truncated_ratios)]
    else:
        pass
    for i, truncated_ratio in enumerate(truncated_ratios):
        plt.plot(k_range, truncated_ratio, marker='o', linestyle='-', label=label_list[i])

    plt.xlabel('k (Top-k Entries)')
    plt.ylabel('Collision Ratio')
    plt.title(title)  # Use the title provided as an argument
    plt.grid(False)

    plt.xticks(np.arange(1, truncate_index + 1, step=plot_ticks_step)) # Adjust the step size for x-axis ticks

    plt.legend()
    plt.show()