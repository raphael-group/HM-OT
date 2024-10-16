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
        ent_pred = entropy(T_pred.flatten())
        if ent_pred > ent_ann:
            print(f'Pred transitions {i} -> {i+1} have **MORE** column entropy: {ent_pred:.3f} > {ent_ann:.3f}')
        else:
            print(f'Pred transitions {i} -> {i+1} have **LESS** column entropy: {ent_pred:.3f} < {ent_ann:.3f}')

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
        