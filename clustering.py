import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_short_vectors(Q,R,T):
    """
    Input
        Q : np.ndarray, of shape (n, r)
        R : np.ndarray, of shape (m, r)
        T : np.ndarray, of shape (r, r)
        emphasis : str, 'slice1', 'slice2', or 'both', default='both'
    Output
        W : np.ndarray, of shape (n, r)
        H : np.ndarray, of shape (m, r)
    """
    gQ = np.sum(Q, axis=0)
    # gR = np.sum(R, axis=0)
    oQ = np.sum(Q, axis=1)
    oR = np.sum(R, axis=1)

    # W is prob(mapped to ct ell' given spot i)
    W = np.diag(1/oQ) @ Q @ np.diag(1/gQ) @ T
    # H is prob(mapped to ct ell' given spot j)
    H = np.diag(1/oR) @ R

    return W, H

def extract_short_vectors_basic(Q,R,T):
    """
    Input
        Q : np.ndarray, of shape (n, r)
        R : np.ndarray, of shape (m, r)
        T : np.ndarray, of shape (r, r)
        emphasis : str, 'slice1', 'slice2', or 'both', default='both'
    Output
        W : np.ndarray, of shape (n, r)
        H : np.ndarray, of shape (m, r)
    """
    gQ = np.sum(Q, axis=0)
    gR = np.sum(R, axis=0)

    # W is prob(mapped to ct ell' given spot i)
    W = Q # @ np.diag(1/gQ) @ T
    # H is prob(mapped to ct ell' given spot j)
    H = R
    return W, H

def max_likelihood_clustering(W,H):
    """
    Input
        W : np.ndarray, of shape (n, r)
        H : np.ndarray, of shape (m, r)
    Output
        labels_W : np.ndarray, of shape (n,)
        labels_H : np.ndarray, of shape (m,)
    """
    # n = W.shape[0]
    # m = H.shape[0]
    # WH = np.vstack((W,H))
    #WH = WH[:,None]/np.sum(WH, axis=1)
    # labels = np.argmax(WH, axis=1)
    labels_W = np.argmax(W, axis=1)
    labels_H = np.argmax(H, axis=1)
    return labels_W, labels_H

def ancestral_clustering(Q,R,T, full_P = True):
    
    gQ = np.sum(Q, axis=0)
    gR = np.sum(R, axis=0)
    Q = Q @ (np.diag(1/gQ) @ T)
    g = gR
    
    # Fixed labels over slice 1
    labels_W = np.argmax(Q, axis=1)
    labels_H = [None]*R.shape[0]
    
    if full_P:
        # Pushback of mass to slice 1 determines final labels
        P = Q @ np.diag(1/g) @ R.T
        i_maxs = np.argmax(P, axis=0)
        # Use clusters on slice 1 to determine co-clustered labels on slice 2
        labels_H = labels_W[i_maxs]
    else:
        # If the full matrix can't be stored, we can still clumsily compute this using a loop
        for idx in range(R.shape[0]):
            if idx % 10000 == 0:
                print(f'Progress: {idx}/{R.shape[0]}')
            P_j = Q @ (np.diag(1/g) @ R.T[:,idx])
            i_max = np.argmax(P_j)
            labels_H[idx] = labels_W[i_max]
    
    return labels_W, labels_H

def k_means_clustering(W, H, k):
    """
    Input
        W : np.ndarray, of shape (n, r)
        H : np.ndarray, of shape (m, r)
        k : int, number of clusters
    Output
        labels_W : np.ndarray, of shape (n,)
        labels_H : np.ndarray, of shape (m,)
    """
    # n_components = k
    # rank_r = W.shape[1]

    km_k = k # np.min((n_components, rank_r)) # use minimum of input k and rank r
    kmeans = KMeans(n_clusters=km_k, n_init=10)

    W_length = len(W)
    # H_length = len(H)

    WH_stack = np.vstack((W, H))

    WH_km = kmeans.fit(WH_stack)
    WH_clusters = WH_km.labels_

    labels_W = WH_clusters[:W_length]
    labels_H = WH_clusters[W_length:]

    return labels_W, labels_H


def plot_cluster_pair(S1, S2, labels_W, labels_H, dotsize=15, \
                      spacing=20, flipy=True, color_scheme='tab', \
                      title=None, save_name=None, flip=False, cell_type_labels=None):
    """
    Input
        S1 : np.ndarray, spatial coordinates for the first slice of shape (n, 2)
        S2 : np.ndarray, spatial coordinates for the second slice of shape (m, 2)
        labels_W : np.ndarray, cluster labels for the first slice
        labels_H : np.ndarray, cluster labels for the second slice
        dotsize : int, size of the dots, default=15
        spacing : int, spacing around the dots, default=20
        flipy : bool, whether to flip the y-axis, default=True
        color_scheme : str, color scheme for the clusters, default='tab', other options: 'rainbow'
        title : str, title for the plot, default=None
        save_name : str, file name to save the plot, default=None
        flip : bool, whether to flip the spatial coordinates, default=False # NOTE: is this flipy?
        cell_type_labels : list, list of length two. 
                            each element of list is itself a list of cell type labels, or None

    Output
        Plots the two slices side by side with spots colored according to their labels from k-means clustering
    """
    set_labels_W = set(labels_W)
    set_labels_H = set(labels_H)
    labels_union = set_labels_W.union(set_labels_H)
    labels_intersection = set_labels_W.intersection(set_labels_H)
    print(f'Number of clusters in slice 1: {len(set_labels_W)}')
    print(f'Number of clusters in slice 2: {len(set_labels_H)}')
    print(f'Number of clusters in common: {len(labels_intersection)}')
    print(f'Number of clusters total: {len(labels_union)}')
    
    num_clusters = len(labels_union)
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='white')

    S1 = S1.cpu().numpy()
    S2 = S2.cpu().numpy()

    S1 = S1 - np.mean(S1, axis=0)
    S2 = S2 - np.mean(S2, axis=0)
    
    slices = [S1, S2]
    labels = [labels_W, labels_H]
    unique_values = np.unique(np.concatenate(labels))
        
    if color_scheme == 'tab':
        cmap_tab20 = plt.get_cmap('tab20')
        cmap_tab20b = plt.get_cmap('tab20b')

        colors_tab20 = [cmap_tab20(i) for i in range(cmap_tab20.N)]
        colors_tab20b = [cmap_tab20b(i) for i in range(cmap_tab20b.N)]

        combined_colors = colors_tab20 + colors_tab20b
        colors = [combined_colors[i % len(combined_colors)] for i in range(len(unique_values))]
    else: 
        colors = [mcolors.hsv_to_rgb((i / num_clusters, 1, 1)) for i in range(num_clusters)]
    
    # Make color_dict for either color scheme
    color_dict = dict(zip(unique_values, colors))
   
    xmin = np.min([np.min(S1[:, 0]), np.min(S2[:, 0])])
    xmax = np.max([np.max(S1[:, 0]), np.max(S2[:, 0])])
    ymin = np.min([np.min(S1[:, 1]), np.min(S2[:, 1])])
    ymax = np.max([np.max(S1[:, 1]), np.max(S2[:, 1])])
    
    for i, (S, value_vec) in enumerate(zip(slices, labels)):
        ax = axes[i]
        ax.set_facecolor('black')

        spatial = S if not flip else S @ np.array([[-1, 0], [0, 1]])
        df = pd.DataFrame({'x': spatial[:, 0], 'y': spatial[:, 1], 'value': value_vec})

        sns.scatterplot(
            x='x', y='y', hue='value', palette=color_dict, data=df, ax=ax, s=dotsize, legend=True
        )

        ax.set_xlim(xmin - spacing, xmax + spacing)
        ax.set_ylim(ymin - spacing, ymax + spacing)
        if flipy:
            ax.invert_yaxis()
        else:
            pass
        ax.axis('off')
        ax.set_title(f'Slice {i+1}\n', color='black')
        ax.set_aspect('equal', adjustable='box')

        if cell_type_labels[i] is not None and i == 0:
            cell_type_labels_2 = cell_type_labels[i]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=cell_type_labels_2, title='')
        elif cell_type_labels[i] is not None and i == 1:
            cell_type_labels_3 = cell_type_labels[i]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=cell_type_labels_3, title='')
            
    plt.tight_layout()
    #plt.legend()
    
    if title:
        plt.suptitle(title, fontsize=16, color='black')

    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor='black')

    plt.show()

def plot_cluster_triple(S1, S2, S3, labels_W, labels_H, labels_I, color_scheme='tab', title=None, save_name=None, flip=False, dotsize=15):
    """
    Input
        S1 : np.ndarray, spatial coordinates for the first slice of shape (n, 2)
        S2 : np.ndarray, spatial coordinates for the second slice of shape (m, 2)
        S3 : np.ndarray, spatial coordinates for the third slice of shape (o, 2)
        labels_W : np.ndarray, cluster labels for the first slice
        labels_H : np.ndarray, cluster labels for the second slice
        labels_I : np.ndarray, cluster labels for the third slice
        color_scheme : str, color scheme for the clusters, default='tab', other options: 'rainbow'
        title : str, title for the plot, default=None
        save_name : str, file name to save the plot, default=None
        flip : bool, whether to flip the spatial coordinates, default=False

    Output
        Plots the three slices side by side with spots colored according to their labels from k-means clustering
    """
    set_labels_W = set(labels_W)
    set_labels_H = set(labels_H)
    set_labels_I = set(labels_I)
    labels_union = set_labels_W.union(set_labels_H).union(set_labels_I)
    labels_intersection = set_labels_W.intersection(set_labels_H).intersection(set_labels_I)
    print(f'Number of clusters in slice 1: {len(set_labels_W)}')
    print(f'Number of clusters in slice 2: {len(set_labels_H)}')
    print(f'Number of clusters in slice 3: {len(set_labels_I)}')
    print(f'Number of clusters in common: {len(labels_intersection)}')
    print(f'Number of clusters total: {len(labels_union)}')
    num_clusters = len(labels_union)
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5)
    fig, axes = plt.subplots(1, 3, figsize=(30, 20), facecolor='white')

    slices = [S1, S2, S3]
    labels = [labels_W, labels_H, labels_I]
    unique_values = np.unique(np.concatenate(labels))

    if color_scheme == 'tab':
        cmap_tab20 = plt.get_cmap('tab20')
        cmap_tab20b = plt.get_cmap('tab20b')

        colors_tab20 = [cmap_tab20(i) for i in range(cmap_tab20.N)]
        colors_tab20b = [cmap_tab20b(i) for i in range(cmap_tab20b.N)]

        combined_colors = colors_tab20 + colors_tab20b
        colors = [combined_colors[i % len(combined_colors)] for i in range(len(unique_values))]
    else:
        colors = [mcolors.hsv_to_rgb((i / num_clusters, 1, 1)) for i in range(num_clusters)]

    # Make color_dict for either color scheme
    color_dict = dict(zip(unique_values, colors))

    # Determine the combined limits of the axes
    all_spatial = np.vstack(slices)
    x_min, x_max = np.min(all_spatial[:, 0]), np.max(all_spatial[:, 0])
    y_min, y_max = np.min(all_spatial[:, 1]), np.max(all_spatial[:, 1])
    
    for i, (S, value_vec) in enumerate(zip(slices, labels)):
        ax = axes[i]
        ax.set_facecolor('black')

        spatial = S if not flip else S @ np.array([[-1, 0], [0, 1]])
        df = pd.DataFrame({'x': spatial[:, 0], 'y': spatial[:, 1], 'value': value_vec})

        sns.scatterplot(
            x='x', y='y', hue='value', palette=color_dict, data=df, ax=ax, s=dotsize, legend=False
        )

        ax.set_xlim(xmin - spacing, xmax + spacing)
        ax.set_ylim(ymin - spacing, ymax + spacing)
        if flipy:
            ax.invert_yaxis()
        else:
            pass
        ax.axis('off')
        ax.set_title(f'Slice {i+1}\n', color='black')
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if title:
        plt.suptitle(title, fontsize=16, color='black')

    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor='black')
    plt.show()

def plot_cluster_list(spatial_list, cluster_list, color_scheme='tab', title=None, save_name=None, flip=False):
    """
    Input
        spatial_list : list of np.ndarray, spatial coordinates for the slices
        cluster_list : list of np.ndarray, labels for the spots across the slices
        color_scheme : str, color scheme for the clusters, default='tab', other options: 'rainbow'
        title : str, title for the plot, default=None
        save_name : str, file name to save the plot, default=None
        flip : bool, whether to flip the spatial coordinates, default=False

    Output
        Plots the three slices side by side with spots colored according to their labels from k-means clustering
    """
    N_slices = len(spatial_list)

    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5)

    fig, axes = plt.subplots(1, N_slices, figsize=(20 * N_slices, 20), facecolor='white')

    slices = [S - np.mean(S, axis=0) for S in spatial_list]

    unique_values = np.unique(np.concatenate(cluster_list))

    if color_scheme == 'tab':
        cmap_tab20 = plt.get_cmap('tab20')
        cmap_tab20b = plt.get_cmap('tab20b')
        cmap_tab20c = plt.get_cmap('tab20c')

        colors_tab20 = [cmap_tab20(i) for i in range(cmap_tab20.N)]
        colors_tab20b = [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
        colors_tab20c = [cmap_tab20c(i) for i in range(cmap_tab20c.N)]

        combined_colors = colors_tab20 + colors_tab20b + colors_tab20c
        colors = [combined_colors[i % len(combined_colors)] for i in range(len(unique_values))]
    else:
        colors = [mcolors.hsv_to_rgb((i / len(unique_values), 1, 1)) for i in range(len(unique_values))]

    # Make color_dict for either color scheme
    color_dict = dict(zip(unique_values, colors))

    # Determine the combined limits of the axes
    all_spatial = np.vstack(slices)
    x_min, x_max = np.min(all_spatial[:, 0]), np.max(all_spatial[:, 0])
    y_min, y_max = np.min(all_spatial[:, 1]), np.max(all_spatial[:, 1])

    for i, (S, value_vec) in enumerate(zip(slices, cluster_list)):
        ax = axes[i]
        ax.set_facecolor('black')

        spatial = S if not flip else S @ np.array([[-1, 0], [0, 1]])
        df = pd.DataFrame({'x': spatial[:, 0], 'y': spatial[:, 1], 'value': value_vec})

        sns.scatterplot(
            x='x', y='y', hue='value', palette=color_dict, data=df, ax=ax, s=100, legend=False
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if flip:
            ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f'Slice {i+1}\n', color='black')
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    plt.subplots_adjust(wspace=.4)  # Adjust the horizontal spacing between subplots

    if title:
        plt.suptitle(title, fontsize=16, color='black')

    if save_name:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor='black')
    plt.show()

def get_color_scheme(cluster_list, color_scheme='tab'):
    unique_values = np.unique(np.concatenate(cluster_list))

    if color_scheme == 'tab':
        cmap_tab20 = plt.get_cmap('tab20')
        cmap_tab20b = plt.get_cmap('tab20b')
        cmap_tab20c = plt.get_cmap('tab20c')

        colors_tab20 = [cmap_tab20(i) for i in range(cmap_tab20.N)]
        colors_tab20b = [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
        colors_tab20c = [cmap_tab20c(i) for i in range(cmap_tab20c.N)]

        combined_colors = colors_tab20 + colors_tab20b + colors_tab20c
        colors = [combined_colors[i % len(combined_colors)] for i in range(len(unique_values))]
    else: 
        colors = [mcolors.hsv_to_rgb((i / len(unique_values), 1, 1)) for i in range(len(unique_values))]
    
    # Make color_dict for either color scheme
    color_dict = dict(zip(unique_values, colors))
    return color_dict

'''
def plot_multi_differentiation(population_list, 
                               transition_list,
                               label_list,
                               color_dict, 
                               dotsize_factor=400, 
                               linethick_factor=10):
    sns.set(style="white")  # Set the Seaborn style
    dsf = dotsize_factor
    ltf = linethick_factor

    N_slices = len(population_list)

    # form x_positions, y_positions, and colors for each slice
    x_positions = []
    y_positions = []
    for i, population in enumerate(population_list):
        y_positions.append(np.arange(len(population)))
        x_positions.append(np.ones(len(population)) * (i))


    plt.figure(figsize=(5 * (N_slices - 1), 20))  # Adjust the figure size

    for pair_ind, T in enumerate(transition_list):
        plt.scatter(x_positions[pair_ind], 
                    y_positions[pair_ind],
                    c=[color_dict[label] for label in label_list[pair_ind]],
                    s=population_list[pair_ind],
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)
        print(pair_ind + 1)
        print(x_positions[pair_ind+1])
        plt.scatter(x_positions[pair_ind+1], 
                    y_positions[pair_ind+1],
                    c=[color_dict[label] for label in label_list[pair_ind+1]],
                    s=population_list[pair_ind+1],
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)
        r1 = T.shape[0]
        r2 = T.shape[1]
        for i in range(r1):
            for j in range(r2):
                if T[i, j] > 0:  # Plot line only if T[i, j] is greater than 0
                    plt.plot([x_positions[pair_ind][i], x_positions[pair_ind+1][j]], 
                            [y_positions[pair_ind][i], y_positions[pair_ind+1][j]], 
                            'k-', lw=T[i, j] * ltf, zorder=0)


    # Add titles and labels
    plt.title('Differentiation Map')
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')

    # Remove the top and right spines
    sns.despine()

    plt.show()
'''
def plot_multi_differentiation(population_list, 
                               transition_list,
                               label_list,
                               color_dict, 
                               dotsize_factor=400, 
                               linethick_factor=10):
    sns.set(style="white")  # Set the Seaborn style
    dsf = dotsize_factor
    ltf = linethick_factor

    N_slices = len(population_list)

    # form x_positions, y_positions, and colors for each slice
    x_positions = []
    y_positions = []
    for i, population in enumerate(population_list):
        y_positions.append(np.arange(len(population)))
        x_positions.append(np.ones(len(population)) * (i))


    plt.figure(figsize=(5 * (N_slices - 1), 20))  # Adjust the figure size

    for pair_ind, T in enumerate(transition_list):
        plt.scatter(x_positions[pair_ind], 
                    y_positions[pair_ind],
                    c=[color_dict[label] for label in label_list[pair_ind]],
                    s=dsf*population_list[pair_ind],
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)
        
        plt.scatter(x_positions[pair_ind+1], 
                    y_positions[pair_ind+1],
                    c=[color_dict[label] for label in label_list[pair_ind+1]],
                    s=dsf*population_list[pair_ind+1],
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)
        r1 = T.shape[0]
        r2 = T.shape[1]
        for i in range(r1):
            for j in range(r2):
                if T[i, j] > 0:  # Plot line only if T[i, j] is greater than 0
                    plt.plot([x_positions[pair_ind][i], x_positions[pair_ind+1][j]], 
                            [y_positions[pair_ind][i], y_positions[pair_ind+1][j]], 
                            'k-', lw=T[i, j] * ltf, zorder=0)


    # Add titles and labels
    plt.title('Differentiation Map')
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')

    # Remove the top and right spines
    sns.despine()

    plt.show()

def plot_labeled_differentiation(population_list,
                                 transition_list,
                                 label_list,
                                 color_dict, 
                                 node_labels=None,  # New parameter for node labels
                                 dotsize_factor=400, 
                                 linethick_factor=10):
    sns.set(style="white")  # Set the Seaborn style
    dsf = dotsize_factor
    ltf = linethick_factor

    N_slices = len(population_list)

    # form x_positions, y_positions, and colors for each slice
    x_positions = []
    y_positions = []
    for i, population in enumerate(population_list):
        y_positions.append(np.arange(len(population)))
        x_positions.append(np.ones(len(population)) * (i))

    plt.figure(figsize=(5 * (N_slices - 1), 10))  # Adjust the figure size

    for pair_ind, T in enumerate(transition_list):
        scatter1 = plt.scatter(x_positions[pair_ind], 
                    y_positions[pair_ind],
                    c=[color_dict[label] for label in label_list[pair_ind]],
                    s=dsf*np.array(population_list[pair_ind]),
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)

        scatter2 = plt.scatter(x_positions[pair_ind+1], 
                    y_positions[pair_ind+1],
                    c=[color_dict[label] for label in label_list[pair_ind+1]],
                    s=dsf*np.array(population_list[pair_ind+1]),
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)

        r1 = T.shape[0]
        r2 = T.shape[1]
        for i in range(r1):
            for j in range(r2):
                if T[i, j] > 0:  # Plot line only if T[i, j] is greater than 0
                    plt.plot([x_positions[pair_ind][i], x_positions[pair_ind+1][j]], 
                            [y_positions[pair_ind][i], y_positions[pair_ind+1][j]], 
                            'k-', lw=T[i, j] * ltf, zorder=0)

    # Add node labels
    if node_labels is not None:
        for i in range(N_slices):
            for j in range(len(population_list[i])):
                plt.text(x_positions[i][j], y_positions[i][j], node_labels[i][j],
                        fontsize=12, ha='right', va='bottom')

    # Add titles and labels
    plt.title('Differentiation Map')
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')

    # Remove the top and right spines
    sns.despine()

    plt.show()