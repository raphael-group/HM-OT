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

import numpy as np
from typing import List
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# TODO: optimize, code slower using reference index 0 vs -1 

def reference_clustering_prime(
    Qs: List[np.ndarray],
    Ts: List[np.ndarray],
    reference_index: int,
    full_P: bool = True,
    batch_size: int = 100,
    min_threshold: float = 1e-10,
    n_jobs: int = None
) -> List[np.ndarray]:
    """
    Assign consistent cluster labels across multiple slices of data by using 
    a designated reference slice. For each non-reference slice, each spot (row)
    is mapped to the cluster label of a reference slice's spot that maximizes a 
    transport plan, either computed fully (if memory permits) or incrementally.
    
    Args:
        Qs (List[np.ndarray]):
            A list of length N, where each element Qs[t] has shape (n_t, r_t).
        Ts (List[np.ndarray]):
            A list of length N-1, where each element Ts[t] has shape (r_t, r_{t+1}).
        reference_index (int):
            An integer s in {0, ..., N-1} specifying which slice to use as the reference.
        full_P (bool, optional):
            If True, compute and store the full transport plans directly.
            If False, use batched computation to save memory. Defaults to True.
        batch_size (int, optional):
            Batch size for incremental computation. Only used when full_P=False.
            Larger values use more memory but compute faster. Defaults to 100.
        min_threshold (float, optional):
            Minimum value for denominators to prevent division by very small numbers.
            Improves numerical stability. Defaults to 1e-10.
        n_jobs (int, optional):
            Number of parallel jobs for batch processing. If None, uses CPU count.
            Only used when full_P=False. Defaults to None.
            
    Returns:
        List[np.ndarray]: A list of length N, where each element is a NumPy array 
        containing integer labels for each spot in the corresponding slice.
    """
    # Number of slices
    N = len(Qs)
    s = reference_index
    
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    # Convert arrays to float64 for better precision
    Qs = [Q.astype(np.float64) if Q.dtype != np.float64 else Q for Q in Qs]
    Ts = [T.astype(np.float64) if T.dtype != np.float64 else T for T in Ts]

    # Split the Qs list around the reference
    Qs_past = Qs[:s]
    Q_s = Qs[s]              # Q at reference slice
    Qs_future = Qs[s+1:]

    # Compute row-sum for reference slice with stability safeguard
    g_s = np.sum(Q_s, axis=0)
    g_s_safe = np.maximum(g_s, min_threshold)  # Avoid division by very small numbers

    # Max-likelihood labels for reference slice
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
        suffixes = [np.diag(1.0 / g_s_safe) @ T_s]
        # Multiply across future transitions
        for T in Ts_future:
            g = np.sum(T, axis=1)
            g_safe = np.maximum(g, min_threshold)
            new_suffix_end = np.diag(1.0 / g_safe) @ T
            suffixes.append(suffixes[-1] @ new_suffix_end)

    # Build prefixes: products of transitions from slice s backward
    if s == 0:
        prefixes = []
    else:
        # First prefix factor from previous slice to reference
        prefixes = [T_sm1 @ np.diag(1.0 / g_s_safe)]
        # Multiply across past transitions in reverse order
        for T in reversed(Ts_past):
            g = np.sum(T, axis=0)
            g_safe = np.maximum(g, min_threshold)
            new_prefix_start = T @ np.diag(1.0 / g_safe)
            prefixes.insert(0, new_prefix_start @ prefixes[0])

    # Prepare to accumulate cluster label assignments
    clustering_list_future = []
    clustering_list_past = []

    # ------------------- Full matrix approach -------------------
    if full_P:
        # Reference is neither the first nor the last slice
        if 0 < s < N-1:
            # Build full P for future slices
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                # More stable computation order with parentheses
                P_st = (Q_s @ suffix) @ (np.diag(1.0 / g_t_safe) @ Q_t.T)
                i_maxs_t = np.argmax(P_st, axis=0)
                labels_t = labels_s[i_maxs_t]
                clustering_list_future.append(labels_t)

            # Build full P for past slices
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                # More stable computation order
                P_ts = (Q_t @ np.diag(1.0 / g_t_safe)) @ (prefix @ Q_s.T)
                i_maxs_t = np.argmax(P_ts, axis=1)
                labels_t = labels_s[i_maxs_t]
                clustering_list_past.append(labels_t)

        # Reference is the first slice
        elif s == 0:
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                P_st = (Q_s @ suffix) @ (np.diag(1.0 / g_t_safe) @ Q_t.T)
                i_maxs_t = np.argmax(P_st, axis=0)
                labels_t = labels_s[i_maxs_t]
                clustering_list_future.append(labels_t)

        # Reference is the last slice
        else:  # s == N-1
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                P_ts = (Q_t @ np.diag(1.0 / g_t_safe)) @ (prefix @ Q_s.T)
                i_maxs_t = np.argmax(P_ts, axis=1)
                labels_t = labels_s[i_maxs_t]
                clustering_list_past.append(labels_t)

    # ------------------- Batched approach -------------------
    else:
        # Batched processing function for parallelization
        def process_batch_future(batch_data, start_idx, Q_s, suffix, g_t_safe, labels_s):
            batch_size = batch_data.shape[0]
            Q_t_batch = batch_data
            inv_g_t_diag = np.diag(1.0 / g_t_safe)
            
            # Precompute part of the matrix multiplication
            Q_s_suffix = Q_s @ suffix
            
            # Process the batch
            P_st_batch = Q_s_suffix @ (inv_g_t_diag @ Q_t_batch.T)
            i_maxs_batch = np.argmax(P_st_batch, axis=0)
            return start_idx, labels_s[i_maxs_batch]
            
        def process_batch_past(batch_data, start_idx, Q_s, prefix, g_t_safe, labels_s):
            batch_size = batch_data.shape[0]
            Q_t_batch = batch_data
            inv_g_t_diag = np.diag(1.0 / g_t_safe)
            
            # Precompute the right part
            prefix_Q_s_T = prefix @ Q_s.T
            
            # Process the batch
            P_ts_batch = (Q_t_batch @ inv_g_t_diag) @ prefix_Q_s_T
            i_maxs_batch = np.argmax(P_ts_batch, axis=1)
            return start_idx, labels_s[i_maxs_batch]

        # Reference is neither the first nor the last slice
        if 0 < s < N-1:
            # Process future slices with batching
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                
                n_spots_t = Q_t.shape[0]
                labels_t = np.zeros(n_spots_t, dtype=int)
                
                # Use batched processing to save memory
                if n_jobs > 1 and n_spots_t > batch_size:
                    # Parallel version for large datasets
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        futures = []
                        for i in range(0, n_spots_t, batch_size):
                            end_idx = min(i + batch_size, n_spots_t)
                            batch = Q_t[i:end_idx]
                            future = executor.submit(
                                process_batch_future, 
                                batch, i, Q_s, suffix, g_t_safe, labels_s
                            )
                            futures.append(future)
                        
                        # Collect results
                        for future in futures:
                            start_idx, batch_labels = future.result()
                            end_idx = min(start_idx + len(batch_labels), n_spots_t)
                            labels_t[start_idx:end_idx] = batch_labels
                else:
                    # Sequential batched version for smaller datasets
                    Q_s_suffix = Q_s @ suffix  # Precompute
                    inv_g_t_diag = np.diag(1.0 / g_t_safe)
                    
                    for i in range(0, n_spots_t, batch_size):
                        if i % 10000 == 0 and i > 0:
                            print(f"Progress (future): {i}/{n_spots_t}")
                        
                        end_idx = min(i + batch_size, n_spots_t)
                        batch = Q_t[i:end_idx].T
                        
                        # More stable computation
                        P_st_batch = Q_s_suffix @ (inv_g_t_diag @ batch)
                        i_maxs_batch = np.argmax(P_st_batch, axis=0)
                        labels_t[i:end_idx] = labels_s[i_maxs_batch]
                
                clustering_list_future.append(labels_t)
            
            # Process past slices with batching
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                
                n_spots_t = Q_t.shape[0]
                labels_t = np.zeros(n_spots_t, dtype=int)
                
                # Use batched processing to save memory
                if n_jobs > 1 and n_spots_t > batch_size:
                    # Parallel version for large datasets
                    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                        futures = []
                        for i in range(0, n_spots_t, batch_size):
                            end_idx = min(i + batch_size, n_spots_t)
                            batch = Q_t[i:end_idx]
                            future = executor.submit(
                                process_batch_past, 
                                batch, i, Q_s, prefix, g_t_safe, labels_s
                            )
                            futures.append(future)
                        
                        # Collect results
                        for future in futures:
                            start_idx, batch_labels = future.result()
                            end_idx = min(start_idx + len(batch_labels), n_spots_t)
                            labels_t[start_idx:end_idx] = batch_labels
                else:
                    # Sequential batched version for smaller datasets
                    prefix_Q_s_T = prefix @ Q_s.T  # Precompute
                    inv_g_t_diag = np.diag(1.0 / g_t_safe)
                    
                    for i in range(0, n_spots_t, batch_size):
                        if i % 10000 == 0 and i > 0:
                            print(f"Progress (past): {i}/{n_spots_t}")
                        
                        end_idx = min(i + batch_size, n_spots_t)
                        batch = Q_t[i:end_idx]
                        
                        # More stable computation
                        P_ts_batch = (batch @ inv_g_t_diag) @ prefix_Q_s_T
                        i_maxs_batch = np.argmax(P_ts_batch, axis=1)
                        labels_t[i:end_idx] = labels_s[i_maxs_batch]
                
                clustering_list_past.append(labels_t)

        # Reference is the first slice - only process future slices
        elif s == 0:
            for Q_t, suffix in zip(Qs_future, suffixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                
                n_spots_t = Q_t.shape[0]
                labels_t = np.zeros(n_spots_t, dtype=int)
                
                # Precompute for better performance
                Q_s_suffix = Q_s @ suffix
                inv_g_t_diag = np.diag(1.0 / g_t_safe)
                
                for i in range(0, n_spots_t, batch_size):
                    if i % 10000 == 0 and i > 0:
                        print(f"Progress (future): {i}/{n_spots_t}")
                    
                    end_idx = min(i + batch_size, n_spots_t)
                    batch = Q_t[i:end_idx].T
                    
                    P_st_batch = Q_s_suffix @ (inv_g_t_diag @ batch)
                    i_maxs_batch = np.argmax(P_st_batch, axis=0)
                    labels_t[i:end_idx] = labels_s[i_maxs_batch]
                
                clustering_list_future.append(labels_t)

        # Reference is the last slice - only process past slices
        else:  # s == N-1
            for Q_t, prefix in zip(Qs_past, prefixes):
                g_t = np.sum(Q_t, axis=0)
                g_t_safe = np.maximum(g_t, min_threshold)
                
                n_spots_t = Q_t.shape[0]
                labels_t = np.zeros(n_spots_t, dtype=int)
                
                # Precompute for better performance
                prefix_Q_s_T = prefix @ Q_s.T
                inv_g_t_diag = np.diag(1.0 / g_t_safe)
                
                for i in range(0, n_spots_t, batch_size):
                    if i % 10000 == 0 and i > 0:
                        print(f"Progress (past): {i}/{n_spots_t}")
                    
                    end_idx = min(i + batch_size, n_spots_t)
                    batch = Q_t[i:end_idx]
                    
                    P_ts_batch = (batch @ inv_g_t_diag) @ prefix_Q_s_T
                    i_maxs_batch = np.argmax(P_ts_batch, axis=1)
                    labels_t[i:end_idx] = labels_s[i_maxs_batch]
                
                clustering_list_past.append(labels_t)

    # Concatenate the final labels in the order: past slices, reference slice, future slices
    clustering_list = clustering_list_past + [labels_s] + clustering_list_future
    return clustering_list