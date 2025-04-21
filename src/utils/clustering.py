import numpy as np
from typing import List, Literal 


################################################################################################
#   clustering functions
################################################################################################


def max_likelihood_clustering(
    Qs: List[np.ndarray],
    mode: Literal["standard", "emission", "soft"] = "standard"
) -> List[np.ndarray]:
    """
    Assigns each row (spot) in each matrix Q ∈ Qs to the cluster (column index) 
    with the highest probability. The function supports three normalization modes:
    'standard', 'emission', and 'soft'.

    Args:
        Qs (List[np.ndarray]):
            A list of length N, where each element is a 2D NumPy array (shape (n_t, r_t)).
            - The t-th array Qs[t] represents joint or conditional probabilities 
              between n_t "spots" (rows) and r_t "clusters" (columns).
        mode (Literal["standard", "emission", "soft"], optional):
            The normalization mode. One of:
                - 'standard': Use each Q as-is (assumed to be joint probabilities).
                - 'emission': Normalize columns to sum to 1, i.e., 
                  each column becomes a conditional distribution of spots | cluster.
                - 'soft': Normalize rows to sum to 1, i.e., 
                  each row becomes a conditional distribution of cluster | spot.
            Defaults to 'standard'.

    Returns:
        List[np.ndarray]: 
            A list of length N, where each element is a 1D NumPy array (shape (n_t,)), 
            specifying the cluster assignments for each row. The cluster labels are 
            integers in {0, ..., r_t-1} for the t-th slice.

    Notes:
        - If 'emission' is chosen, each Q is multiplied on the right by diag(1 / column_sum).
          This ensures each column sums to 1.
        - If 'soft' is chosen, each Q is multiplied on the left by diag(1 / row_sum).
          This ensures each row sums to 1.
        - If 'standard' is chosen, no normalization is performed.

    Example:
        >>> import numpy as np
        >>> Qs = [np.random.rand(5, 3), np.random.rand(6, 4)]  # two slices
        >>> cluster_assignments = max_likelihood_clustering(Qs, mode='soft')
        >>> for i, ca in enumerate(cluster_assignments):
        ...     print(f"Slice {i} cluster labels:", ca)
    """
    N = len(Qs)

    if mode == "standard":
        # No normalization
        Ms = Qs
    elif mode == "emission":
        Ms = [None] * N
        for t in range(N):
            # Column sums
            col_sum = np.sum(Qs[t], axis=0)
            Ms[t] = Qs[t] @ np.diag(1.0 / col_sum)
    elif mode == "soft":
        Ms = [None] * N
        for t in range(N):
            # Row sums
            row_sum = np.sum(Qs[t], axis=1)
            Ms[t] = np.diag(1.0 / row_sum) @ Qs[t]
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'standard', 'emission', or 'soft'.")

    # Assign each row to the cluster with the highest probability
    clustering_list = [np.argmax(M, axis=1) for M in Ms]
    return clustering_list


def reference_clustering(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    reference_index: int,
    full_P: bool = True
) -> List[np.ndarray]:
    """
    Assign consistent cluster labels across multiple slices of data by using 
    a designated reference slice. For each non-reference slice, each spot (row)
    is mapped to the cluster label of a reference slice's spot that maximizes a 
    transport plan, either computed fully (if memory permits) or incrementally.

    The function returns a list of label arrays, one for each slice. All slices
    share the same cluster label set as the reference slice.

    Args:
        Qs (List[np.ndarray]):
            A list of length N, where each element Qs[t] has shape (n_t, r_t).
            - Rows (n_t) often correspond to "spots" or data points in slice t.
            - Columns (r_t) correspond to the number of clusters in slice t.
            - Each entry can represent joint or raw probabilities between spots 
              and clusters in slice t.
        Ts (List[np.ndarray]):
            A list of length N-1, where each element Ts[t] has shape (r_t, r_{t+1}).
            - Ts[t] is a transition matrix linking the cluster space at slice t 
              to the cluster space at slice t+1 (or vice versa, depending on usage).
        reference_index (int):
            An integer s in {0, ..., N-1} specifying which slice to use as the reference.
            The reference slice has shape (n_s, r_s).
        full_P (bool, optional):
            If True, compute and store the full transport plans P^(s, t) or P^(t, s) 
            directly, then do argmax to assign labels. If False, compute each spot's 
            assignment incrementally without building the entire matrix in memory 
            (useful for large data). Defaults to True.

    Returns:
        List[np.ndarray]:
            A list of length N, where each element is a NumPy array (shape (n_t,)).
            - clustering_list[t] contains an integer label for each spot in slice t.
            - Labels are in {0, ..., r_s - 1}, matching the reference slice's cluster set.

    Notes:
        1. Let s := reference_index. The reference slice s has r_s clusters and we assign:
           labels_s = argmax(Q_s, axis=1).
        2. We compute or approximate transport plans that map each slice t (t ≠ s) 
           to the reference slice s. The argmax of the appropriate P^(s,t) or P^(t,s) 
           (depending on future vs. past relative to s) links each spot in slice t 
           to a spot in slice s, thus giving it the same label as that spot in slice s.
        3. The function handles boundary cases (s = 0, s = N-1) and different paths 
           to build the prefix/suffix products for transitions (Ts) when going backward
           or forward in time.
        4. If not full_P, the code loops over spots individually to avoid building 
           a large matrix in memory, which can be slow but necessary for large data.

    Example:
        >>> # Suppose we have 3 slices, each with Q of shape (num_spots, num_clusters)
        >>> Q0 = np.random.rand(100, 5)
        >>> Q1 = np.random.rand(120, 4)
        >>> Q2 = np.random.rand(90, 6)
        >>> Qs = [Q0, Q1, Q2]
        >>> T01 = np.random.rand(5, 4)  # transitions between slice 0 and 1
        >>> T12 = np.random.rand(4, 6)  # transitions between slice 1 and 2
        >>> Ts = [T01, T12]
        >>> # Use slice 1 as the reference
        >>> clustering = reference_clustering(Qs, Ts, reference_index=1, full_P=False)
        >>> for idx, labels in enumerate(clustering):
        ...     print(f"Slice {idx} labels shape:", labels.shape)
    """
    # Number of slices
    N = len(Qs)
    s = reference_index

    # Split the Qs list around the reference
    Qs_past = Qs[:s]
    Q_s = Qs[s]              # Q at reference slice
    Qs_future = Qs[s+1:]

    # Compute row-sum for reference slice
    g_s = np.sum(Q_s, axis=0)

    # Max-likelihood labels for reference slice (same approach as in max_likelihood_clustering)
    labels_s = np.argmax(Q_s, axis=1)

    # Split the Ts list around the reference
    Ts_past = Ts[:s-1] if s > 0 else []
    T_sm1 = Ts[s-1] if s > 0 else None
    T_s = Ts[s] if s < N-1 else None
    Ts_future = Ts[s+1:] if s < N-1 else []

    # Build suffixes: products of transitions from slice s forward
    if s == N-1:
        suffixes = []
    else:
        # First suffix factor from reference to next
        suffixes = [np.diag(1.0 / g_s) @ T_s]
        # Multiply across future transitions
        for T in Ts_future:
            g = np.sum(T, axis=1)
            new_suffix_end = np.diag(1.0 / g) @ T
            suffixes.append(suffixes[-1] @ new_suffix_end)

    # Build prefixes: products of transitions from slice s backward
    if s == 0:
        prefixes = []
    else:
        # First prefix factor from previous slice to reference
        prefixes = [T_sm1 @ np.diag(1.0 / g_s)]
        # Multiply across past transitions in reverse order
        for T in reversed(Ts_past):
            g = np.sum(T, axis=0)
            new_prefix_start = T @ np.diag(1.0 / g)
            prefixes.insert(0, new_prefix_start @ prefixes[0])

    # Prepare to accumulate cluster label assignments
    clustering_list_future = []
    clustering_list_past = []

    # ------------------- Full matrix approach (if full_P) -------------------
    if full_P:
        # Reference is neither the first nor the last slice
        if 0 < s < N-1:
            # Build full P for future slices
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                P_st = Q_s @ suffix @ np.diag(1.0 / g_t) @ Q_t.T
                i_maxs_t = np.argmax(P_st, axis=0)  # map col index to reference slice row
                labels_t = labels_s[i_maxs_t]
                clustering_list_future.append(labels_t)

            # Build full P for past slices
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                P_ts = Q_t @ np.diag(1.0 / g_t) @ prefix @ Q_s.T
                i_maxs_t = np.argmax(P_ts, axis=1)  # map row index to reference slice row
                labels_t = labels_s[i_maxs_t]
                clustering_list_past.append(labels_t)

        # Reference is the first slice
        elif s == 0:
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                P_st = Q_s @ suffix @ np.diag(1.0 / g_t) @ Q_t.T
                i_maxs_t = np.argmax(P_st, axis=0)
                labels_t = labels_s[i_maxs_t]
                clustering_list_future.append(labels_t)

        # Reference is the last slice
        else:  # s == N-1
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                P_ts = Q_t @ np.diag(1.0 / g_t) @ prefix @ Q_s.T
                i_maxs_t = np.argmax(P_ts, axis=1)
                labels_t = labels_s[i_maxs_t]
                clustering_list_past.append(labels_t)

    # ------------------- Incremental (spot-by-spot) approach (if not full_P) -------------------
    else:
        # Reference is neither the first nor the last slice
        if 0 < s < N-1:
            # Future
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                labels_t = np.zeros(Q_t.shape[0], dtype=int)
                for j in range(Q_t.shape[0]):
                    if j % 10000 == 0:
                        print(f"Progress (future): {j}/{Q_t.shape[0]}")
                    P_st_j = Q_s @ suffix @ np.diag(1.0 / g_t) @ Q_t.T[:, j]
                    labels_t[j] = labels_s[np.argmax(P_st_j)]
                clustering_list_future.append(labels_t)

            # Past
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                labels_t = np.zeros(Q_t.shape[0], dtype=int)
                for j in range(Q_t.shape[0]):
                    if j % 10000 == 0:
                        print(f"Progress (past): {j}/{Q_t.shape[0]}")
                    P_ts_j = Q_t @ np.diag(1.0 / g_t) @ prefix @ Q_s.T[:, j]
                    labels_t[j] = labels_s[np.argmax(P_ts_j)]
                clustering_list_past.append(labels_t)

        # Reference is the first slice
        elif s == 0:
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                labels_t = np.zeros(Q_t.shape[0], dtype=int)
                for j in range(Q_t.shape[0]):
                    if j % 10000 == 0:
                        print(f"Progress (future): {j}/{Q_t.shape[0]}")
                    P_st_j = Q_s @ suffix @ np.diag(1.0 / g_t) @ Q_t.T[:, j]
                    labels_t[j] = labels_s[np.argmax(P_st_j)]
                clustering_list_future.append(labels_t)

        # Reference is the last slice
        else:  # s == N-1
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                labels_t = np.zeros(Q_t.shape[0], dtype=int)
                for j in range(Q_t.shape[0]):
                    if j % 10000 == 0:
                        print(f"Progress (past): {j}/{Q_t.shape[0]}")
                    P_ts_j = Q_t @ np.diag(1.0 / g_t) @ prefix @ Q_s.T[:, j]
                    labels_t[j] = labels_s[np.argmax(P_ts_j)]
                clustering_list_past.append(labels_t)

    # Concatenate the final labels in the order: past slices, reference slice, future slices
    clustering_list = clustering_list_past + [labels_s] + clustering_list_future
    return clustering_list