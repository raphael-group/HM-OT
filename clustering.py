import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


################################################################################################
#   clustering functions
################################################################################################

CLUSTERING_DICT = {
    0:'ml', # max_likelihood_clustering(..., mode='standard') ** default
    1:'reference', # reference_clustering(..., reference_index=...)
    2:'ml-emission', # max_likelihood_clustering(..., mode='emission')
    3:'ml-soft', # max_likelihood_clustering(..., mode='soft')
    # 4:'kmeans-reference', # k_means_reference(..., reference_index=..., k=...) 
}


def max_likelihood_clustering(Qs, mode='standard'):
    '''
    Args
        Qs : list of length N. 
                each element of Qs is a 
                np.ndarray, of shape (n_t, r_t), for slice t in {0, ..., N-1}
        mode : str, 'standard', 'emission', 'soft', default='standard'
    Out
        clustering_list : list of length N.
                each element of clustering_list is a
                np.ndarray, of shape (n_t,), with r_t labels in {0, ..., r_t-1}
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
    N = len(Qs)
    if mode == 'standard':
        Ms = Qs
    elif mode == 'emission':
        Ms = [None]*N
        for t in range(N):
            g_t = np.sum(Qs[t], axis=0)
            Ms[t] = Qs[t] @ np.diag(1/g_t)
    elif mode == 'soft':
        Ms = [None]*N
        for t in range(N):
            a_t = np.sum(Qs[t], axis=1)
            Ms[t] = np.diag(1/a_t) @ Qs[t]
    else:
        raise ValueError('Invalid mode')
    
    clustering_list = [np.argmax(M, axis=1) for M in Ms]
    return clustering_list


def reference_clustering(Qs, Ts, reference_index, full_P=True):
    '''
    Args
        Qs : list of length N.
                each element of Qs is a 
                np.ndarray, of shape (n_t, r_t), for slice t in {0, ..., N-1}
        Ts : list of length N-1.
                each element of Ts is a
                np.ndarray, of shape (r_t, r_{t+1}), for transition t in {0, ..., N-2}
        reference_index : int s in {0, ..., N-1}, index of the slice to use as reference
                denote as s, with r_s cell types
        full_P : bool, whether to compute using full matrix P, default=True
                set to False if the full matrix P can't be stored
    Out
        clustering_list : list of length N.
                each element of clustering_list is a
                np.ndarray, of shape (n_t,), with r_s labels in {0, ..., r_s-1}
                there are now the same number of labels for each slice

    Description
        Assigns each spot in reference slice to the cluster (column index) with the highest probability
        This is identical to ml clusteirng on the reference slice. 

        Let s := reference_index,
        labels_s = max_likelihood_clustering(Q_s)

        P^(s, s+1) := Q_s @ diag(1/g_s) @ T^(s, s+1) @ diag(1/g_{s+1}) @ Q_{s+1}^T
        P^(s, s+2) := Q_s @ diag(1/g_s) @ T^(s, s+1) @ diag(1/g_{s+1}) @ T^(s+1, s+2) @ diag(1/g_{s+2}) @ Q_{s+2}^T
        ...
        P^(s, N-1) := Q_s @ diag(1/g_s) @ T^(s, s+1) @ diag(1/g_{s+1}) @ ... @ diag(1/g_{N-2}) @ T^{N-2, N-1} @ diag(1/g_{N-1}) @ Q_{N-1}^T

        and analogously,
        P^(s-1, s) := Q_{s-1} @ diag(1/g_{s-1}) @ T^{s-1, s} @ diag(1/g_s) @ Q_s^T
        P^(s-2, s) := Q_{s-2} @ diag(1/g_{s-2}) @ T^{s-2, s-1} @ diag(1/g_{s-1}) @ T^{s-1, s} @ diag(1/g_s) @ Q_s^T
        ...
        P^(0, s) := Q_0 @ diag(1/g_0) @ T^{0, 1} @ diag(1/g_1) @ ... @ diag(1/g_{s-1}) @ T^{s-1, s} @ diag(1/g_s) @ Q_s^T

        for slices t > s, 
            imaxs_t = argmax(P^(s, t), axis=0)
            labels_t = labels_s[imaxs_t]

        for slices t < s,
            imaxs_t = argmax(P^(t, s), axis=1)
            labels_t = labels_s[imaxs_t]
    '''
    N = len(Qs)
    s = reference_index
    Qs_past = Qs[: s] 
    Q_s = Qs[s]
    g_s = np.sum(Q_s, axis=0)
    Qs_future = Qs[s+1:]

    labels_s = np.argmax(Q_s, axis=1)
    
    Ts_past = Ts[:s-1]
    T_sm1 = Ts[s-1]
    if s == N-1:
        T_s = None
    else:
        T_s = Ts[s]
    if s == N-1:
        Ts_future = None
    else:
        Ts_future = Ts[s+1:]

    # make list of suffix factors
    if reference_index == N-1:
        suffixes = []
    else:
        suffixes = [ np.diag(1 / g_s) @ T_s ] # initialize earliest timepoint suffix
        for T in Ts_future:
            g = np.sum(T, axis=1)
            new_suffix_end = np.diag(1/g) @ T
            suffixes.append(suffixes[-1] @ new_suffix_end)

    # make list of prefix factors
    if reference_index == 0:
        prefixes = []
    else:
        prefixes = [ T_sm1 @ np.diag(1 / g_s) ] # initialize latest timepoint prefix
        for T in Ts_past[::-1]: # iterate backwards in time
            g = np.sum(T, axis=0)
            new_prefix_start = T @ np.diag(1/g)
            prefixes.insert(0, new_prefix_start @ prefixes[0]) # insert at beginning of prefixes, as we move backward, so that prefixes is oriented forwards in time
        
    # in either case of full_P, we construct the cluster lists separately for future and past timepoints
    clustering_list_future = []
    clustering_list_past =[]

    if full_P and reference_index < N-1 and reference_index > 0:
        # make full transport plans between reference s and future timepoints t
        for Q_t, suffix in zip(Qs_future, suffixes):
            g_t = np.sum(Q_t, axis=0)
            P_st = Q_s @ suffix @ np.diag(1 / g_t) @ Q_t.T
            i_maxs_t = np.argmax(P_st, axis=0) # map t -> s, axis=0 because this arrow is backwards in time
            labels_t = labels_s[i_maxs_t]
            clustering_list_future.append(labels_t)
        
        # make full transport plans between past timepoints t and reference s
        for Q_t, prefix in zip(Qs_past, prefixes):
            g_t = np.sum(Q_t, axis=0)
            P_ts = Q_t @ np.diag(1 / g_t) @ prefix @ Q_s.T
            i_maxs_t = np.argmax(P_ts, axis=1) # map t -> s, axis=1 because this arrow is forwards in time
            labels_t = labels_s[i_maxs_t]
            clustering_list_past.append(labels_t)

    elif full_P and reference_index == 0:
        # make full transport plans between reference s and future timepoints t
        for Q_t, suffix in zip(Qs_future, suffixes):
            g_t = np.sum(Q_t, axis=0)
            P_st = Q_s @ suffix @ np.diag(1 / g_t) @ Q_t.T
            i_maxs_t = np.argmax(P_st, axis=0)
            labels_t = labels_s[i_maxs_t]
            clustering_list_future.append(labels_t)

    elif full_P and reference_index == N-1:
        # make full transport plans between past timepoints t and reference s
        for Q_t, prefix in zip(Qs_past, prefixes):
            g_t = np.sum(Q_t, axis=0)
            P_ts = Q_t @ np.diag(1 / g_t) @ prefix @ Q_s.T
            i_maxs_t = np.argmax(P_ts, axis=1)
            labels_t = labels_s[i_maxs_t]
            clustering_list_past.append(labels_t)  

    elif not full_P and reference_index < N-1 and reference_index > 0:       
        # If the full matrices P_st, P_ts can't be stored, we can still slowly compute labels using a loop

        # make labels for future timepoints t from those at reference s
        for Q_t, suffix in zip(Qs_future, suffixes):
            g_t = np.sum(Q_t, axis=0)
            labels_t = np.zeros(Q_t.shape[0], dtype=int)
            for j in range(Q_t.shape[0]):
                if j % 10000 == 0:
                    print(f'Progress: {j}/{Q_t.shape[0]}')
                #P_j = Q_tilde @ (np.diag(1/gR) @ R.T[:,j])
                P_st_j = Q_s @ suffix @ np.diag(1 / g_t) @ Q_t.T[:,j]
                # i_maxs_t = phi(j) : t -> s
                i_maxs_t = np.argmax(P_st_j)
                labels_t[j] = labels_s[i_maxs_t]
        
            clustering_list_future.append(labels_t)
        
        # make labels for past timepoints t from those at reference s
        for Q_t, prefix in zip(Qs_past, prefixes):
            g_t = np.sum(Q_t, axis=0)
            labels_t = np.zeros(Q_t.shape[0], dtype=int)
            for j in range(Q_t.shape[0]):
                if j % 10000 == 0:
                    print(f'Progress: {j}/{Q_t.shape[0]}')
                #P_j = Q_tilde @ (np.diag(1/gR) @ R.T[:,j])
                P_ts_j = Q_t @ np.diag(1 / g_t) @ prefix @ Q_s.T[:,j]
                # i_maxs_t = phi(j) : t -> s
                i_maxs_t = np.argmax(P_ts_j)
                labels_t[j] = labels_s[i_maxs_t]
        
            clustering_list_past.append(labels_t)
    
    elif not full_P and reference_index == 0:
        # make labels for future timepoints t from those at reference s
        for Q_t, suffix in zip(Qs_future, suffixes):
            g_t = np.sum(Q_t, axis=0)
            labels_t = np.zeros(Q_t.shape[0], dtype=int)
            for j in range(Q_t.shape[0]):
                if j % 10000 == 0:
                    print(f'Progress: {j}/{Q_t.shape[0]}')
                #P_j = Q_tilde @ (np.diag(1/gR) @ R.T[:,j])
                P_st_j = Q_s @ suffix @ np.diag(1 / g_t) @ Q_t.T[:,j]
                # i_maxs_t = phi(j) : t -> s
                i_maxs_t = np.argmax(P_st_j)
                labels_t[j] = labels_s[i_maxs_t]
        
            clustering_list_future.append(labels_t)

    elif not full_P and reference_index == N-1:
        # make labels for past timepoints t from those at reference s
        for Q_t, prefix in zip(Qs_past, prefixes):
            g_t = np.sum(Q_t, axis=0)
            labels_t = np.zeros(Q_t.shape[0], dtype=int)
            for j in range(Q_t.shape[0]):
                if j % 10000 == 0:
                    print(f'Progress: {j}/{Q_t.shape[0]}')
                #P_j = Q_tilde @ (np.diag(1/gR) @ R.T[:,j])
                P_ts_j = Q_t @ np.diag(1 / g_t) @ prefix @ Q_s.T[:,j]
                # i_maxs_t = phi(j) : t -> s
                i_maxs_t = np.argmax(P_ts_j)
                labels_t[j] = labels_s[i_maxs_t]
        
            clustering_list_past.append(labels_t)
        
    clustering_list = clustering_list_past + [labels_s] + clustering_list_future
        
    return clustering_list

"""
def k_means_reference(Q, R, T, k, ...):
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
"""

################################################################################################
#   plotting helper functions
################################################################################################

def get_color_dict(labels_list, 
                   cmap='tab'):
    '''
    Input
        clustering_list : list of np.ndarray, labels for the spots across the slices
        cmap : str, color map for the clusters, default='tab', other options: 'rainbow'
    Output
        color_dict : dict, dictionary with cluster labels as keys and colors as values
    '''
    unique_values = np.unique(np.concatenate(labels_list))

    if cmap== 'tab':
        cmap_tab20 = plt.get_cmap('tab20')
        cmap_tab20b = plt.get_cmap('tab20b')
        cmap_tab20c = plt.get_cmap('tab20c')

        colors_tab20 = [cmap_tab20(i) for i in range(cmap_tab20.N)]
        colors_tab20b = [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
        colors_tab20c = [cmap_tab20c(i) for i in range(cmap_tab20c.N)]

        combined_colors = colors_tab20 + colors_tab20b + colors_tab20c

        print(f'total number of tab colors that can be displayed: {len(combined_colors)}')
        print(f'total number of unique values: {len(unique_values)}')

        colors = [combined_colors[i % len(combined_colors)] for i in range(len(unique_values))]
    else: 
        colors = [mcolors.hsv_to_rgb((i / len(unique_values), 1, 1)) for i in range(len(unique_values))]
    
    # Make color_dict for either color map
    color_dict = dict(zip(unique_values, colors))
    return color_dict

def get_diffmap_inputs(clustering_list, 
                       clustering_type, 
                       cmap='tab'):
    '''
    Input
        clustering_list : list of np.ndarray, labels for the spots across the slices
        clustering_type : str, 'ml' or 'reference'
        cmap : str, color map for the clusters, default='tab', 
                NOTE: anything other than 'tab' results in 'rainbow', currently
    Output
        population_list : list of lists, number of spots in each cluster for each slice
        label_list : list of lists, unique cluster labels for each slice
        color_dict : dict, dictionary with cluster labels as keys and colors as
    '''
    # make population_list
    population_list = []
    for clustering in clustering_list:
        populations = [len(np.where(clustering == label)[0]) for label in set(clustering)]
        population_list.append(populations)

    # make label_list
    labels_list = []
    cs = 0 # count shift or cumulative sum, to make unique labels across slices in ml case.

    if clustering_type == 'ml':
        for i in range(len(clustering_list)):
            labels_list.append(list(set(clustering_list[i] + cs)))
            cs += len(set(clustering_list[i]))
    elif clustering_type == 'reference':
        labels_list = [ list(set(clustering)) for clustering in clustering_list]
    else:
        pass

    # make color_dict
    color_dict = get_color_dict(labels_list, cmap=cmap)

    return population_list, labels_list, color_dict

def get_reference_transition_matrices(Qs, Ts, reference_index):
    clustering_list = reference_clustering(Qs, Ts, reference_index)


################################################################################################
#   plotting: core functions
################################################################################################


def plot_clustering_list(spatial_list, 
                      clustering_list,
                      clustering_type='ml',
                      cell_type_labels=None,
                      cmap='tab', 
                      title=None, 
                      save_name=None, 
                      dotsize=1,
                      flip=False):
    '''
    Input
        spatial_list : list of np.ndarray, spatial coordinates for the slices
        clustering_list : list of np.ndarray, labels for the spots across the slices
        clustering_type : str, 'ml' or 'reference', default='ml'
        cmap : str, color map for the clusters, default='tab', other options: 'rainbow'
        title : str, title for the plot, default=None
        save_name : str, file name to save the plot, default=None
        flip : bool, whether to flip the spatial coordinates, default=False

    Output 
    '''

    N_slices = len(spatial_list)

    if cell_type_labels is None:
        cell_type_labels = [None]*N_slices

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5)

    fig, axes = plt.subplots(1, N_slices, figsize=(20 * N_slices, 20), facecolor='white')

    # Center the spatial coordinates
    slices = [S - np.mean(S, axis=0) for S in spatial_list]

    # Make color_dict
    _, _, color_dict = get_diffmap_inputs(clustering_list, clustering_type, cmap)

    # Determine the combined limits of the axes
    all_spatial = np.vstack(slices)
    x_min, x_max = np.min(all_spatial[:, 0]), np.max(all_spatial[:, 0])
    y_min, y_max = np.min(all_spatial[:, 1]), np.max(all_spatial[:, 1])

    if clustering_type == 'ml':
        cs = 0 # plays same role as in get_diffmap_inputs

    for i, (S, value_vec) in enumerate(zip(slices, clustering_list)):
        ax = axes[i]
        ax.set_facecolor('black')

        if clustering_type == 'ml':
            value_vec_prime = value_vec + cs
        else:
            value_vec_prime = value_vec

        spatial = S if not flip else S @ np.array([[-1, 0], [0, 1]])
        df = pd.DataFrame({'x': spatial[:, 0], 'y': spatial[:, 1], 'value': value_vec_prime})

        if clustering_type == 'ml':
            cs += len(set(value_vec))

        sns.scatterplot(
            x='x', y='y', hue='value', palette=color_dict, data=df, ax=ax, s=dotsize, legend=True
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
        plt.suptitle(title, fontsize=36, color='black')

    if save_name is not None:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor='black')
    plt.show()

    return None


def plot_labeled_differentiation(population_list,
                                 transition_list,
                                 label_list,
                                 color_dict, 
                                 cell_type_labels=None,  # New parameter for node labels
                                 clustering_type='ml',
                                 reference_index=None,
                                 dotsize_factor=1, 
                                 linethick_factor=10,
                                 save_name=None,
                                 title=None,
                                 stretch=1):
    '''
    Args
        population_list : list of lists, number of spots in each cluster for each slice
        transition_list : list of np.ndarrays, cell type coupling matrices between consecutive slices
        label_list : list of lists, unique cluster labels (ints) for each slice
        color_dict : dict, dictionary with cluster labels as keys and colors as values
        cell_type_labels : list of lists of str, default=None
        clustering_type : str, 'ml' or 'reference', default='ml'
        reference_index : int, index of the slice to use as reference, default=None
                        NOTE: reference_index is required if clustering_type='reference'
        dotsize_factor : int, factor to scale the size of the dots, default=1
        linethick_factor : int, factor to scale the thickness of the lines, default=10
        save_name : str, file name to save the plot, default=None

    Output

    Description
    '''
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

    plt.figure(figsize=(stretch*5 * (N_slices - 1), 10))  # Adjust the figure size
    
    for pair_ind, T in enumerate(transition_list):
        plt.scatter(stretch*x_positions[pair_ind], 
                    y_positions[pair_ind],
                    c=[color_dict[label] for label in label_list[pair_ind]],
                    s=dsf*np.array(population_list[pair_ind]),
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)

        plt.scatter(stretch*x_positions[pair_ind+1], 
                    y_positions[pair_ind+1],
                    c=[color_dict[label] for label in label_list[pair_ind+1]],
                    s=dsf*np.array(population_list[pair_ind+1]),
                    edgecolor='b',
                    linewidth=1,
                    zorder=1)

        r1 = T.shape[0]
        r2 = T.shape[1]

        if clustering_type == 'ml':
            for i in range(r1):
                for j in range(r2):
                    if T[i, j] > 0:  # Plot line only if T[i, j] is greater than 0
                        # print(len(x_positions), len(x_positions[pair_ind]), len(x_positions[pair_ind+1]))
                        plt.plot([stretch*x_positions[pair_ind][i], stretch*x_positions[pair_ind+1][j]], 
                                [y_positions[pair_ind][i], y_positions[pair_ind+1][j]], 
                                'k-', lw=T[i, j] * ltf, zorder=0)
        else:
            pass

    # Add node labels
    if cell_type_labels is not None:
        for i in range(N_slices):
            if cell_type_labels[i] is not None:
                for j in range(len(population_list[i])):
                    plt.text(
                        x_positions[i][j], y_positions[i][j], cell_type_labels[i][j],
                        fontsize=12, ha='right', va='bottom'
                    )

    # Add titles and labels
    if title:
        plt.suptitle(title, fontsize=36, color='black')
    else:
        plt.title('Differentiation Map')
    plt.yticks([])
    plt.xticks([])
    plt.axis('off')

    # Remove the top and right spines
    sns.despine()
    if save_name is not None:
        plt.savefig(save_name, dpi=300, transparent=True, bbox_inches="tight", facecolor='black')
    plt.show()

    return None


################################################################################################
#   plotting: more directly from from output Qs, Ts
################################################################################################

def diffmap_from_QT(Qs, 
                    Ts, 
                    cell_type_labels=None, 
                    clustering_type='ml', 
                    reference_index=None,
                    title=None,
                    save_name=None, 
                    dsf=1,
                    stretch=1):
    '''
    Args:
        Qs : list of (N) np.ndarrays, of shape (n_t, r_t), for each slice
        Ts : list of (N-1) np.ndarray, of shape (r_t, r_{t+1}), for each transition
        clustering_type : str, 'ml' or 'reference', default='ml'
        cell_type_labels : list of (N) lists of str, default=None
        reference_index : int, index of the slice to use as reference, default=None
                        NOTE: reference_index is required if clustering_type='reference'
    '''
    # make clustering_list
    if clustering_type == 'ml':
        clustering_list = max_likelihood_clustering(Qs)
    
    elif clustering_type == 'reference':
        if reference_index is None:
            raise ValueError('Reference index required for reference clustering')
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError('Invalid clustering type')

    # get diffmap inputs
    population_list, labels_list, color_dict = get_diffmap_inputs(clustering_list, clustering_type)

    # make transition_list
    transition_list = Ts

    plot_labeled_differentiation(population_list,
                                 transition_list,
                                 labels_list,
                                 color_dict, 
                                 cell_type_labels,
                                 clustering_type,
                                 dotsize_factor=dsf, 
                                 linethick_factor=10,
                                 title=title,
                                 save_name=save_name,
                                 stretch=stretch)
    
    return None

def plot_clusters_from_QT(Ss, 
                          Qs, 
                          Ts, 
                          cell_type_labels=None, 
                          clustering_type='ml',
                          reference_index=None,
                          title=None, 
                          save_name=None, 
                          dotsize=1,
                             flip=False):
    '''
    Args:
        Ss : list of (N) np.ndarrays, of shape (n_t, 2), for each slice, spatial coords
        Qs : list of (N) np.ndarrays, of shape (n_t, r_t), for each slice
        Ts : list of (N-1) np.ndarray, of shape (r_t, r_{t+1}), for each transition
        cell_type_labels : list of (N) lists of str, default=None
        clustering_type : str, 'ml' or 'reference', default='ml'
        reference_index : int, index of the slice to use as reference, default=None
                        NOTE: reference_index is required if clustering_type='reference'
    '''
    # make clustering_list
    if clustering_type == 'ml':
        clustering_list = max_likelihood_clustering(Qs)
    
    elif clustering_type == 'reference':
        if reference_index is None:
            raise ValueError('Reference index required for reference clustering')
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError('Invalid clustering type')

    plot_clustering_list(spatial_list=Ss,
                      clustering_list=clustering_list,
                      cell_type_labels=cell_type_labels,
                      clustering_type=clustering_type,
                      cmap='tab', 
                      title=title, 
                      save_name=save_name, 
                      dotsize=dotsize,
                      flip=flip)
    
    return None

################################################################################################
#   plotting: both, directly from from output Qs, Ts
################################################################################################

def both_from_QT(Ss, 
                 Qs, 
                 Ts, 
                 cell_type_labels=None, 
                 clustering_type='ml', 
                 reference_index=None,
                 save_name=None,
                 title=None):
    '''
    Args:
        Ss : list of (N) np.ndarrays, of shape (n_t, 2), for each slice, spatial coords
        Qs : list of (N) np.ndarrays, of shape (n_t, r_t), for each slice
        Ts : list of (N-1) np.ndarray, of shape (r_t, r_{t+1}), for each transition
        cell_type_labels : list of (N) lists of str, default=None
        clustering_type : str, 'ml' or 'reference', default='ml'
        reference_index : int, index of the slice to use as reference, default=None
                        NOTE: reference_index is required if clustering_type='reference'
        save_name : str, file name to save the plot, default=None
        title : str, title for the plot, default=None
    '''
    diffmap_from_QT(Qs=Qs, 
                    Ts=Ts, 
                    clustering_type=clustering_type, 
                    cell_type_labels=cell_type_labels, 
                    reference_index=reference_index,
                    title=title,
                    save_name=save_name)

    plot_clusters_from_QT(Ss=Ss, 
                          Qs=Qs, 
                          Ts=Ts, 
                          cell_type_labels=cell_type_labels, 
                          clustering_type=clustering_type,
                          reference_index=reference_index,
                          title=title, 
                          save_name=save_name)
    return None



# TODO:
# whenever we're plotting the original zf clusters, it would be nice to use their original color scheme. 

################################################################################################
#   sankey plotting
################################################################################################

def rgba_to_plotly_string(rgba):
    ''' Convert a list of [r, g, b, a] values to an rgba string for plotly '''
    r, g, b, a = rgba
    return f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})'

def plot_labeled_differentiation_sankey(population_list,
                                        transition_list,
                                        label_list,
                                        color_dict, 
                                        cell_type_labels=None,  # New parameter for node labels
                                        clustering_type='ml',
                                        reference_index=None,
                                        dotsize_factor=1, 
                                        linethick_factor=10,
                                        plot_height=600,  # New parameter for height adjustment
                                        save_name=None,
                                        title=None):
    '''
    Args
        population_list : list of lists, number of spots in each cluster for each slice
        transition_list : list of np.ndarrays, cell type coupling matrices between consecutive slices
        label_list : list of lists, unique cluster labels (ints) for each slice
        color_dict : dict, dictionary with cluster labels as keys and colors as values
        cell_type_labels : list of lists of str, default=None or None
        clustering_type : str, 'ml' or 'reference', default='ml'
        reference_index : int, index of the slice to use as reference, default=None
        dotsize_factor : int, factor to scale the size of the dots, default=1
        linethick_factor : int, factor to scale the thickness of the lines, default=10
        plot_height : int, height of the plot in pixels, default=600
        save_name : str, file name to save the plot, default=None

    Output

    Description
    '''

    N_slices = len(population_list)
    
    # Prepare node and link data for Sankey plot
    node_labels = []
    link_sources = []
    link_targets = []
    link_values = []
    node_colors = []
    
    node_idx_map = {}  # To keep track of node indices for different slices
    current_node_idx = 0
    
    # Build nodes and transitions (links)
    for slice_idx, population in enumerate(population_list):
        for i, label in enumerate(label_list[slice_idx]):
            # Add node label; handle None case
            if cell_type_labels and cell_type_labels[slice_idx] is not None:
                node_label = cell_type_labels[slice_idx][i] if cell_type_labels[slice_idx][i] is not None else str(label)
            else:
                node_label = str(label)  # Default to the cluster label if no cell type labels are given
            
            node_labels.append(node_label)
            node_idx_map[(slice_idx, i)] = current_node_idx  # Map to node index
            
            # Convert color to a plotly-friendly format
            node_colors.append(rgba_to_plotly_string(color_dict[label]))
            current_node_idx += 1
    
    for pair_ind, T in enumerate(transition_list):
        r1 = T.shape[0]
        r2 = T.shape[1]

        for i in range(r1):
            for j in range(r2):
                if T[i, j] > 0:  # Only add non-zero transitions
                    source_node = node_idx_map[(pair_ind, i)]
                    target_node = node_idx_map[(pair_ind + 1, j)]
                    link_sources.append(source_node)
                    link_targets.append(target_node)
                    link_values.append(T[i, j] * linethick_factor)

    # Create the Sankey plot
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,  # Correctly formatted colors
        ),
        link=dict(
            source=link_sources,  # Indices of source nodes
            target=link_targets,  # Indices of target nodes
            value=link_values     # Flow values for transitions
        )
    ))

    # Add title and adjust height
    fig.update_layout(
        title_text=title if title else 'Differentiation Map',
        font_size=24,
        height=plot_height  # Set plot height dynamically
    )

    # Save plot if needed
    if save_name is not None:
        fig.write_image(save_name)
    
    fig.show()

    return None
def diffmap_from_QT_sankey(Qs, 
                    Ts, 
                    cell_type_labels=None, 
                    clustering_type='ml', 
                    reference_index=None,
                    title=None,
                    save_name=None, 
                    dsf=1,
                    plot_height=600):
    '''
    Args:
        Qs : list of (N) np.ndarrays, of shape (n_t, r_t), for each slice
        Ts : list of (N-1) np.ndarray, of shape (r_t, r_{t+1}), for each transition
        clustering_type : str, 'ml' or 'reference', default='ml'
        cell_type_labels : list of (N) lists of str, default=None
        reference_index : int, index of the slice to use as reference, default=None
                        NOTE: reference_index is required if clustering_type='reference'
    '''
    # make clustering_list
    if clustering_type == 'ml':
        clustering_list = max_likelihood_clustering(Qs)
    
    elif clustering_type == 'reference':
        if reference_index is None:
            raise ValueError('Reference index required for reference clustering')
        clustering_list = reference_clustering(Qs, Ts, reference_index)
    else:
        raise ValueError('Invalid clustering type')

    # get diffmap inputs
    population_list, labels_list, color_dict = get_diffmap_inputs(clustering_list, clustering_type)

    # make transition_list
    transition_list = Ts

    # Call the Sankey plot function instead of the previous one
    plot_labeled_differentiation_sankey(population_list,
                                        transition_list,
                                        labels_list,
                                        color_dict, 
                                        cell_type_labels,
                                        clustering_type,
                                        dotsize_factor=dsf, 
                                        linethick_factor=10,
                                        plot_height=plot_height,
                                        title=title,
                                        save_name=save_name)
    
    return None

def both_from_QT_sankey(Ss, 
                 Qs, 
                 Ts, 
                 cell_type_labels=None, 
                 clustering_type='ml', 
                 reference_index=None,
                 save_name=None,
                 title=None):
    '''
    Args:
        Ss : list of (N) np.ndarrays, of shape (n_t, 2), for each slice, spatial coords
        Qs : list of (N) np.ndarrays, of shape (n_t, r_t), for each slice
        Ts : list of (N-1) np.ndarray, of shape (r_t, r_{t+1}), for each transition
        cell_type_labels : list of (N) lists of str, default=None
        clustering_type : str, 'ml' or 'reference', default='ml'
        reference_index : int, index of the slice to use as reference, default=None
                        NOTE: reference_index is required if clustering_type='reference'
        save_name : str, file name to save the plot, default=None
        title : str, title for the plot, default=None
    '''
    diffmap_from_QT_sankey(Qs=Qs, 
                    Ts=Ts, 
                    clustering_type=clustering_type, 
                    cell_type_labels=cell_type_labels, 
                    reference_index=reference_index,
                    title=title,
                    save_name=save_name)

    plot_clusters_from_QT(Ss=Ss, 
                          Qs=Qs, 
                          Ts=Ts, 
                          cell_type_labels=cell_type_labels, 
                          clustering_type=clustering_type,
                          reference_index=reference_index,
                          title=title, 
                          save_name=save_name)
    return None