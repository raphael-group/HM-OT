import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###
# clustering
###

def max_likelihood_clustering(W,H):
    """
    Input
        W : np.ndarray, of shape (n, r)
        H : np.ndarray, of shape (m, r)
    Output
        labels_W : np.ndarray, of shape (n,)
        labels_H : np.ndarray, of shape (m,)
    """
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

###
# plotting
###

def plot_cluster_list(spatial_list, 
                      cluster_list, 
                      cell_type_labels=None,
                      color_scheme='tab', 
                      title=None, 
                      save_name=None, 
                      flip=False):
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

    if cell_type_labels is None:
        cell_type_labels = [None]*N_slices

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
            x='x', y='y', hue='value', palette=color_dict, data=df, ax=ax, s=100, legend=True
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if flip:
            ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f'Slice {i+1}\n', color='black')
        ax.set_aspect('equal', adjustable='box')

        if cell_type_labels[i] is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=cell_type_labels[i], title='')

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
            if node_labels[i] is not None:
                for j in range(len(population_list[i])):
                    plt.text(
                        x_positions[i][j], y_positions[i][j], node_labels[i][j],
                        fontsize=12, ha='right', va='bottom'
                    )

    # Add titles and labels
    plt.title('Differentiation Map')
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')

    # Remove the top and right spines
    sns.despine()

    plt.show()

def get_diffmap_inputs(clustering_list):
    # make population_list
    populations0 = [len(np.where(clustering_list[0] == label)[0]) for label in set(clustering_list[0])]
    populations1 = [len(np.where(clustering_list[1] == label)[0]) for label in set(clustering_list[1])]
    populations2 = [len(np.where(clustering_list[2] == label)[0]) for label in set(clustering_list[2])]

    population_list = [populations0, populations1, populations2]

    # make label_list
    label_list = []

    cs = 0

    for i in range(len(clustering_list)):
        label_list.append(list(set(clustering_list[i] + cs)))
        cs += len(set(clustering_list[i]))

    # make color_dict
    color_dict = get_color_scheme(label_list, color_scheme='tab')

    return population_list, label_list, color_dict

def diffmap_from_QT(Qs, Ts, node_labels=None, clustering_type='ml'):
    '''
    Args:
        Qs : list of (N) np.ndarrays, of shape (n_t, r_t), for each slice
        Ts : list of (N-1) np.ndarray, of shape (r_t, r_{t+1}), for each transition
        clustering_type : str, 'ml' or 'kmeans', default='ml'
    '''
    # make clustering_list
    clustering_list = []
    for i in range(len(Qs)):
        if clustering_type == 'ml':
            Q_i_clusters, _ = max_likelihood_clustering(Qs[i], Qs[i])
            clustering_list += [Q_i_clusters]
        elif clustering_type == 'ancestral':
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i], Ts[i], full_P=True)
            clustering_list += [Q_i_clusters]
        else:
            raise ValueError('Invalid clustering type')
    
    # return clustering_list
    # get diffmap inputs
    population_list, label_list, color_dict = get_diffmap_inputs(clustering_list)

    # make transition_list
    transition_list = Ts

    plot_labeled_differentiation(population_list,
                                 transition_list,
                                 label_list,
                                 color_dict, 
                                 node_labels,  # New parameter for node labels
                                 dotsize_factor=1, 
                                 linethick_factor=10)
    
    return None

def plot_clusters_from_QT(Ss, Qs, Ts, node_labels=None, clustering_type='ml', title=None, save_name=None):
    '''
    Args:
        Ss : list of (N) np.ndarrays, of shape (n_t, 2), for each slice, spatial coords
        Qs : list of (N) np.ndarrays, of shape (n_t, r_t), for each slice
        Ts : list of (N-1) np.ndarray, of shape (r_t, r_{t+1}), for each transition
        clustering_type : str, 'ml' or 'kmeans', default='ml'
    '''
    # make clustering_list
    clustering_list = []
    for i in range(len(Qs)):
        if clustering_type == 'ml':
            Q_i_clusters, _ = max_likelihood_clustering(Qs[i], Qs[i])
            clustering_list += [Q_i_clusters]
        elif clustering_type == 'ancestral':
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i], Ts[i], full_P=True)
            clustering_list += [Q_i_clusters]
        else:
            raise ValueError('Invalid clustering type')

    plot_cluster_list(spatial_list=Ss,
                      cluster_list=clustering_list,
                      cell_type_labels=node_labels,
                      color_scheme='tab', 
                      title=title, 
                      save_name=save_name, 
                      flip=False)
    
    return None