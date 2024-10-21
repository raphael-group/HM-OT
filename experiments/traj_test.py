import numpy as np

def normalize_to_row_stochastic(Ts):
    """
    Normalize a list of coupling matrices to be row-stochastic (i.e., each row sums to 1).

    Args:
        Ts (list of np.ndarray): A list of coupling matrices (2D numpy arrays).

    Returns:
        normalized_Ts (list of np.ndarray): A list of normalized row-stochastic coupling matrices.
    """
    normalized_Ts = []

    first_marginal = Ts[0].sum(axis=1, keepdims=False)
    
    for T in Ts:
        # Normalize each row of the matrix to sum to 1
        row_sums = T.sum(axis=1, keepdims=False)
        print(f'row_sums shape: {row_sums.shape}')
        print(f'T shape: {T.shape}')
        # Handle any rows that sum to zero by avoiding division by zero
        normalized_T = np.diag(1/row_sums) @ T
        normalized_Ts.append(normalized_T)
    
    return first_marginal, normalized_Ts


# for now, sample according to Ts, in future we can specify an initial cluster or even initial point
def sample_latent_trajectories(Ts_rs, g_init, sizes, num_trajectories=10):
    """
    Sample trajectories from the given latent coupling matrices Ts.

    Args:
        Ts_rs (list of np.ndarray): A list of coupling matrices T_t for t = 1, ..., N-1.
                                    These should be row-stochastic, but we ensure they are normalized.
        g_init (np.ndarray): The initial distribution over states.
        sizes (list of int): The sizes of the state spaces at each timepoint.
        num_trajectories (int): Number of trajectories to sample.

    Returns:
        trajectories (np.ndarray): An array of sampled trajectories of shape (num_trajectories, N).
    """
    N = len(sizes)  # Number of timepoints
    trajectories = np.zeros((num_trajectories, N), dtype=int)

    # Normalize initial distribution
    g_init = g_init / np.sum(g_init)

    # Sample initial state
    trajectories[:, 0] = np.random.choice(sizes[0], num_trajectories, p=g_init)

    # Sample subsequent states
    for t in range(1, N):
        for i in range(num_trajectories):
            prev_state = trajectories[i, t-1]
            # Normalize the row of Ts_rs corresponding to the previous state
            row = Ts_rs[t-1][prev_state]
            row = row / np.sum(row)  # Normalize to ensure it sums to 1
            # Sample next state based on the normalized row
            trajectories[i, t] = np.random.choice(sizes[t], p=row)
    
    return trajectories