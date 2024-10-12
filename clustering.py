import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###
#   clustering
###


CLUSTERING_DICT = {
    0:'ml', # max_likelihood_clustering(..., mode='standard') ** default
    1:'ml-emission', # max_likelihood_clustering(..., mode='emission')
    2:'ml-soft', # max_likelihood_clustering(..., mode='soft')
    3:'ancestral', # ancestral_clustering(..., descendant=False) ** default
    4:'descendent', # ancestral_clustering(..., descendant=True)
    5:'kmeans-ancestral', # k_means_descendant(..., descendant=False) ** default
    6:'kmeans-descendant' # k_means_descendant(..., descendant=True)
}


def max_likelihood_clustering(Q, R, mode='standard'):
    '''
    Args
        Q : np.ndarray, of shape (n, r_1), slice 1
        R : np.ndarray, of shape (m, r_2), slice 2
        mode : str, 'standard', 'emission', 'soft', default='standard'
    Out
        labels_Q : np.ndarray, of shape (n,) with r_1 labels in {0, ..., r_1-1} 
        labels_R : np.ndarray, of shape (m,) with r_2 labels in {0, ..., r_2-1}

    Description
        Assigns each spot in slice 1 to the cluster (column index) with the highest probability
        Assigns each spot in slice 2 to the cluster (column index) with the highest probability
        Assumes distinct sets of labels for each slice

        in 'standard' mode, Q and R are used as is, where each is assumed to be a joint distribution
        between spots and cell types of a given slice. Entries are  joint probabilities.

        in 'emission' mode, Q and R are normalized by the inner marginals to be column-stochastic.
        Entries are now conditional probabilities of spots given cell types, 
        which we call emission probabilities, as in HMMs.

        in 'soft' mode, Q and R are normalized by the outer marginals to be row-stochastic.
        Entries are now conditional probabilities of cell types given spots,
        constituting soft assignments / clusterings / transition matrices
    '''
    # form inner marginals
    gQ = np.sum(Q, axis=0)
    gR = np.sum(R, axis=0)

    # form outer marginals
    a = np.sum(Q, axis=1)
    b = np.sum(R, axis=1)

    if mode == 'standard':
        Q_matrix = Q
        R_matrix = R
    elif mode == 'emission':
        Q_matrix = Q @ np.diag(1/gQ)
        R_matrix = R @ np.diag(1/gR)
    elif mode == 'soft':
        Q_matrix = np.diag(1/a) @ Q
        R_matrix = np.diag(1/b) @ R
    else:
        raise ValueError('Invalid mode')
    labels_Q = np.argmax(Q_matrix, axis=1)
    labels_R = np.argmax(R_matrix, axis=1)
    return labels_Q, labels_R


def ancestral_clustering(Q, R, T, full_P=True, descendant=False):
    '''
    Args
        Q : np.ndarray, of shape (n, r_1), slice 1
        R : np.ndarray, of shape (m, r_2), slice 2
        T : np.ndarray, of shape (r_1, r_2), cell type coupling between slice 1 and slice 2
        full_P : bool, whether to compute using full matrix P, default=True
                set to False if the full matrix P can't be stored
    Out
        labels_Q : np.ndarray, of shape (n,) with r_1 labels in {0, ..., r_1-1}
        labels_R : np.ndarray, of shape (m,) with r_1 labels in {0, ..., r_1-1}

    Description
        Assigns each spot in slice 1 to the cluster (column index) with the highest probability
        Uses transport plan to determine map "phi : (slice 2) -> (slice 1)" 
                        from columns (j's in slice 2) to rows (i's in slice 1)
        The slice 1 assignments and map phi determine the slice 2 assignments

    *** Setting argument descendant=True switches the roles of slice 1 and slice 2 ***
    '''
    if descendant==True:
        T = T.T
        Q, R = R, Q
    else:
        pass
    # form inner marginals
    gQ = np.sum(Q, axis=0)
    gR = np.sum(R, axis=0)
    # represent slice 1 using slice 2 cell types
    Q_tilde = Q @ (np.diag(1/gQ) @ T)
    
    # labels in slice 1 are determined as in max_likelihood_clustering
    labels_Q = np.argmax(Q, axis=1)
    # labels in slice 2 initialized as None
    labels_R = [None]*R.shape[0]
    
    if full_P:
        # complete Q_tilde to full transport plan P
        P = Q_tilde @ np.diag(1/gR) @ R.T
        # i_maxs is the map phi : (slice 2) -> (slice 1)
        i_maxs = np.argmax(P, axis=0) # i_maxs[j] is the index of the max value in column j
        # labels on slice 1 determine labels on slice 2 through i_maxs
        labels_R = labels_Q[i_maxs] # labels_R[j] is the label of the co-clustered spot in slice 1
    else:
        # If the full matrix P can't be stored, we can still slowly compute using a loop
        for j in range(R.shape[0]):
            if j % 10000 == 0:
                print(f'Progress: {j}/{R.shape[0]}')
            P_j = Q_tilde @ (np.diag(1/gR) @ R.T[:,j])
            # i_max = phi(j)
            i_max = np.argmax(P_j)
            labels_R[j] = labels_Q[i_max]
    
    if descendant==True:
        labels_Q, labels_R = labels_R, labels_Q
    else:
        pass
    return labels_Q, labels_R


def k_means_descendant(Q, R, T, k, descendant=False):
    '''
    Input
        Q : np.ndarray, of shape (n, r_1), slice 1
        R : np.ndarray, of shape (m, r_2), slice 2
        T : np.ndarray, of shape (r_1, r_2), cell type coupling between slice 1 and slice 2
        k : int, number of clusters to use for k-means
    Output
        labels_Q : np.ndarray, of shape (n,), with k labels in {0, ..., k-1}
        labels_R : np.ndarray, of shape (m,), with k labels in {0, ..., k-1}

    Description
        Q_tilde = Q @ (np.diag(1/gQ) @ T) represents slice 1 using slice 2 cell types,
        and has shape (n, r_2).
        The two representations Q_tilde, R are stacked with shape (n+m, r_2).
        k-means is applied to the stack, and the cluster labels are assigned to the two slices.
    
    *** Setting argument descendant=False switches the roles of slice 1 and slice 2 ***
    '''
    if descendant==False:
        T = T.T
        Q, R = R, Q
    else:
        pass
    Q_length = len(Q)
    # compute inner marginals
    gQ = np.sum(Q, axis=0)
    gR = np.sum(R, axis=0)
    # represent slice 1 using slice 2 cell types
    Q_tilde = Q @ (np.diag(1/gQ) @ T)

    # stack the two representations, which use the same set of cell types
    QR_stack = np.vstack((Q_tilde, R))

    # initialize k-means, fit to the stack
    kmeans = KMeans(n_clusters=k, n_init=10)
    QR_km = kmeans.fit(QR_stack)
    QR_clusters = QR_km.labels_

    # assign cluster labels to the two slices from k-means
    labels_Q = QR_clusters[:Q_length]
    labels_R = QR_clusters[Q_length:]

    if descendant==False:
        labels_Q, labels_R = labels_R, labels_Q
    else:
        pass
    return labels_Q, labels_R

###
#   plotting
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
                    print(len(x_positions), len(x_positions[pair_ind]), len(x_positions[pair_ind+1]))
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
    population_list = []
    for clustering in clustering_list:
        populations = [len(np.where(clustering == label)[0]) for label in set(clustering)]
        population_list.append(populations)

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
        elif clustering_type == 'ancestral' and i < len(Qs) - 1:
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i+1], Ts[i], full_P=True)
            clustering_list += [Q_i_clusters]
        elif clustering_type == 'ancestral' and i == len(Qs) - 1:
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i-1], Ts[i-1].T, full_P=True)
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
        elif clustering_type == 'ancestral' and i < len(Qs) - 1:
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i+1], Ts[i], full_P=True)
            clustering_list += [Q_i_clusters]
        elif clustering_type == 'ancestral' and i == len(Qs) - 1:
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i-1], Ts[i-1].T, full_P=True)
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

def testing_ancestral(Qs, Ts, node_labels=None, clustering_type='ml', title=None, save_name=None):
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
        elif clustering_type == 'ancestral' and i < len(Qs) - 1:
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i+1], Ts[i], full_P=True)
            clustering_list += [Q_i_clusters]
        elif clustering_type == 'ancestral' and i == len(Qs) - 1:
            Q_i_clusters, _ = ancestral_clustering(Qs[i], Qs[i-1], Ts[i-1].T, full_P=True)
            clustering_list += [Q_i_clusters]
        else:
            raise ValueError('Invalid clustering type')
    
    print(clustering_list)
    return None
    plot_cluster_list(spatial_list=Ss,
                      cluster_list=clustering_list,
                      cell_type_labels=node_labels,
                      color_scheme='tab', 
                      title=title, 
                      save_name=save_name, 
                      flip=False)
    
    return None

# TODO:
# whenever we're plotting the original zf clusters, it would be nice to use their original color scheme. 