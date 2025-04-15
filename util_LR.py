import random
import torch
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import clustering
from sklearn.metrics import adjusted_mutual_info_score
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def norm_factors(tup, c):
    """
    Scales a pair of low-rank factor tensors by a constant.

    Args:
        tup (tuple[torch.Tensor, torch.Tensor]): A tuple containing two tensors 
            (e.g., factors A1 and A2).
        c (float): A scaling constant.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the scaled factors.
    """
    return (tup[0] / c, tup[1] / c)


def _compute_Q(_adata, cell_type_key):
    """
    Creates a one-hot encoded matrix Q representing cell-type annotations and 
    extracts the list of unique cell-type labels.

    Args:
        _adata (AnnData): An AnnData object containing single-cell data 
            (e.g., gene expression).
        cell_type_key (str): The key in _adata.obs that stores cell-type annotations.

    Returns:
        tuple[np.ndarray, list]:
            - _Q (np.ndarray): A 2D one-hot encoded array of shape (n_cells, n_labels).
            - label (list): List of unique cell types corresponding to the columns of _Q.
    """

    if cell_type_key not in _adata.obs.columns:
        raise ValueError(f"Column '{cell_type_key}' not found in AnnData.obs.")

    # One-hot encode Q
    encoder = OneHotEncoder(sparse_output=False)
    ys_onehot = encoder.fit_transform(_adata.obs[cell_type_key].values.reshape(-1, 1))
    
    _Q = ys_onehot / np.sum(ys_onehot)
    label = list(encoder.categories_[0])
    
    return _Q, label


def get_replicates_from_AnnData(adata, \
                                replicates, timepoints, \
                                replicate_key, timepoint_key):
    """
    Splits an AnnData object into a list of sub-AnnData objects based on replicate 
    and timepoint metadata.

    This function iterates over pairs of (replicate, timepoint) and extracts the 
    corresponding subset of cells from `adata`.

    Args:
        adata (AnnData): The AnnData object containing single-cell data with 
            multiple replicates and timepoints.
        replicates (list[str]): A list of replicate IDs to filter on.
        timepoints (list[str]): A list of timepoints (or conditions) to filter on.
        replicate_key (str): Key in `adata.obs` specifying which replicate 
            each cell belongs to.
        timepoint_key (str): Key in `adata.obs` specifying which timepoint 
            each cell belongs to.

    Returns:
        list[AnnData]: A list of AnnData objects, each corresponding to the specified 
        replicate-timepoint pairs in `replicates` and `timepoints`.
    """
    adata_replicates = []

    if replicates is not None:
        # Subset to replicates of interest
        for idx, (rep, time) in enumerate(zip(replicates, timepoints)):
    
            mask_time = adata.obs[timepoint_key] == time
            adata_t = adata[mask_time]
            
            mask_rep = adata_t.obs[replicate_key] == rep
            adata_rep = adata_t[mask_rep]
            
            adata_replicates.append(adata_rep)
    else:
        # Simply aggregate across all replicates
        for time in timepoints:
    
            mask_time = adata.obs[timepoint_key] == time
            adata_t = adata[mask_time]
            
            adata_replicates.append(adata_t)
    
    return adata_replicates

    
def convert_adata(adata, 
                  timepoints, 
                  replicates = None, 
                  cell_type_key = 'cellstate', 
                  timepoint_key = 'timepoint', 
                  replicate_key = 'orig.ident', 
                  spatial_key = ['x_loc', 'y_loc'], 
                  feature_key = 'X_pca', 
                  dtype=torch.float32, 
                  device='cpu',
                  compute_Q = True,
                  spatial = True, 
                  dist_rank=100, 
                  dist_eps=0.02
                 ):
    
    """
    
    Converts an AnnData object and lists of replicates/timepoints into 
    inputs suitable for Hidden Markov Optimal Transport (HM-OT).

    This function:
      1. Splits the original AnnData into sub-AnnData objects for each 
         replicate-timepoint pair.
      2. Optionally computes a low-rank distance factorization for spatial data.
      3. Collects feature embeddings for each replicate-timepoint slice.
      4. Optionally computes a one-hot encoded cluster assignment matrix Q.

    Args:
        adata (AnnData): The AnnData object containing all cells across replicates 
            and timepoints.
        timepoints (list[str]): A list of timepoints (or conditions).
        replicates (list[str], optional): A list of replicate IDs.
        cell_type_key (str, optional): Column in `adata.obs` for cell annotations. 
            Defaults to 'cellstate'.
        timepoint_key (str, optional): Column in `adata.obs` for timepoint labels. 
            Defaults to 'timepoint'.
        replicate_key (str, optional): Column in `adata.obs` for replicate IDs. 
            Defaults to 'orig.ident'.
        spatial_key (tuple[str, str], optional): Column(s) in `adata.obs` storing 
            spatial coordinates. Defaults to ('x_loc','y_loc').
        feature_key (str, optional): Key in `adata.obsm` for feature embeddings 
            (e.g., PCA). Defaults to 'X_pca'.
        dtype (torch.dtype, optional): The torch data type to use. Defaults to torch.float32.
        device (str, optional): The device on which to place torch tensors.
        compute_Q (bool, optional): Whether to compute a one-hot matrix Q based on 
            cell_type_key. Defaults to True.
        spatial (bool, optional): Whether to compute a low-rank approximation of 
            spatial distances. Defaults to True.
        dist_rank (int, optional): The rank used in `low_rank_distance_factorization`. 
            Defaults to 100.
        dist_eps (float, optional): Epsilon parameter for the low-rank factorization 
            method. Defaults to 0.02.

    Returns:
        tuple:
            - C_factors_sequence (list[tuple[torch.Tensor, torch.Tensor]]): A list of 
              low-rank factors approximating pairwise distances between consecutive slices.
            - A_factors_sequence (list[tuple[torch.Tensor, torch.Tensor]]): A list of 
              low-rank factors for within-slice distances (spatial or otherwise).
            - Qs (list[torch.Tensor]): List of one-hot encoded cluster matrices (if compute_Q=True). 
            - labels (list[list]): List of lists of cell-type labels for each slice.
            - rank_list (list[tuple[int,int]]): For each pair of slices, a tuple describing 
              the dimensions (rank) of Q clusters from slice i to slice i+1.
            - spatial_sequence (list[np.ndarray]): List of arrays containing 
              spatial coordinates for each replicate-timepoint slice.
    """
    
    N = len(timepoints)
    
    A_factors_sequence = []
    
    features_sequence = []
    spatial_sequence = []
    
    Qs = [None]*N
    labels = [None]*N

    adata_replicates = get_replicates_from_AnnData(adata, \
                                replicates, timepoints, \
                                replicate_key, timepoint_key)
    
    for idx, adata_rep in enumerate(adata_replicates):
        
        if spatial:
            ### Compute low-rank approximation to the distance matrix and appropriately normalize
            spatial_coords = torch.tensor( adata_rep.obs[spatial_key].to_numpy(), dtype=dtype ).to(device)
            A_rep = low_rank_distance_factorization( spatial_coords, spatial_coords, dist_rank, dist_eps, device=device )
            c = estimate_max_norm( A_rep[0] , A_rep[1] )
            A_rep = norm_factors( A_rep, c**1/2 )
            A_factors_sequence.append( (A_rep[0].to(dtype).to(device), A_rep[1].to(dtype).to(device)) )
            spatial_sequence.append(spatial_coords.numpy())
        else:
            A_factors_sequence.append( None )
        
        feature_coords = torch.tensor(adata_rep.obsm[feature_key], dtype=dtype).to(device)
        features_sequence.append(feature_coords)
        
        if compute_Q:
            _Q, label = _compute_Q(adata_rep, cell_type_key=cell_type_key)
            Qs[idx] = torch.tensor(_Q, dtype=dtype, device=device)
            labels[idx] = label
    
    C_factors_sequence = []
    rank_list = []
    
    for i in range(0, N-1, 1):
        
        idx1, idx2 = i, i+1
        
        features1, features2 = features_sequence[idx1], features_sequence[idx2]
        print('Computing low-rank distance matrix!')
        C_rep = low_rank_distance_factorization( features1, features2, dist_rank, dist_eps , device = device )
        c = estimate_max_norm( C_rep[0] , C_rep[1] )
        C_rep = norm_factors( C_rep, c**1/2 )
        C_factors_sequence.append( (C_rep[0].to(dtype).to(device), C_rep[1].to(dtype).to(device)))
        
        rank_list.append( (len(labels[idx1]), len(labels[idx2]) ) )
    
    return C_factors_sequence, A_factors_sequence, Qs, labels, rank_list, spatial_sequence



def low_rank_distance_factorization(X1, X2, r, eps, device='cpu', dtype=torch.float64):

    """
    Approximates the pairwise distance matrix between X1 and X2 with two low-rank 
    factors using a sampling-based algorithm (inspired by Bauscke, Indyk, Woodruff).

    This function randomly samples rows/columns to produce an approximate 
    factorization of the distance matrix. The final return is (V, U.T) such that
    distance_matrix ~ V @ U.T.

    References:
        - Indyk, et al. (2019)
        - Frieze, et al. (2004)
        - Chen & Price (2017)

    Args:
        X1 (torch.Tensor): Coordinates (e.g., features, spatial) of shape (n, d).
        X2 (torch.Tensor): Coordinates of shape (m, d).
        r (int): Target rank.
        eps (float): Sampling parameter that influences the number of sampled rows/columns.
        device (str, optional): Device on which to perform computations. Defaults to 'cpu'.
        dtype (torch.dtype, optional): The tensor data type. Defaults to torch.float64.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - V (torch.Tensor): Approximate factor of shape (n, r).
            - U^T (torch.Tensor): Transpose of the approximate factor of shape (r, m).
    """
    
    n = X1.shape[0]
    m = X2.shape[0]
    '''
    Indyk '19
    '''
    # low-rank distance matrix factorization of Bauscke, Indyk, Woodruff
    
    t = int(r/eps) # this is poly(1/eps, r) in general -- this t might not achieve the correct bound tightly
    i_star = random.randint(1, n)
    j_star = random.randint(1, m)
    
    # Define probabilities of sampling
    p = (torch.cdist(X1, X2[j_star][None,:])**2 \
            + torch.cdist(X1[i_star,:][None,:], X2[j_star,:][None,:])**2 \
                    + (torch.sum(torch.cdist(X1[i_star][None,:], X2))/m) )[:,0]**2
    
    p_dist = (p / p.sum())
    
    # Use random choice to sample rows
    indices_p = torch.from_numpy(np.random.choice(n, size=(t), p=p_dist.cpu().numpy())).to(device)
    X1_t = X1[indices_p, :]
    '''
    Frieze '04
    '''
    P_t = torch.sqrt(p[indices_p]*t)
    S = torch.cdist(X1_t, X2)/P_t[:, None] # t x m
    
    # Define probabilities of sampling by row norms
    q = torch.norm(S, dim=0)**2 / torch.norm(S)**2 # m x 1
    q_dist = (q / q.sum())
    # Use random choice to sample rows
    indices_q = torch.from_numpy(np.random.choice(m, size=(t), p=q_dist.cpu().numpy())).to(device)
    S_t = S[:, indices_q] # t x t
    Q_t = torch.sqrt(q[indices_q]*t)
    W = S_t[:, :] / Q_t[None, :]
    # Find U
    U, Sig, Vh = torch.linalg.svd(W) # t x t for all
    F = U[:, :r] # t x r
    # U.T for the final return
    U_t = (S.T @ F) / torch.norm(W.T @ F) # m x r
    '''
    Chen & Price '17
    '''
    # Find V for the final return
    indices = torch.from_numpy(np.random.choice(m, size=(t))).to(device)
    X2_t = X2[indices, :] # t x dim
    D_t = torch.cdist(X1, X2_t) / np.sqrt(t) # n x t
    Q = U_t.T @ U_t # r x r
    U, Sig, Vh = torch.linalg.svd(Q)
    U = U / Sig # r x r
    U_tSub = U_t[indices, :].T # t x r
    B = U.T @ U_tSub / np.sqrt(t) # (r x r) (r x t)
    A = torch.linalg.inv(B @ B.T)
    Z = ((A @ B) @ D_t.T) # (r x r) (r x t) (t x n)
    V = Z.T @ U
    return V.double(), U_t.T.double()

def hadamard_square_lr(A1, A2, device='cpu'):
    
    """
    Computes a Hadamard (elementwise) square of a low-rank matrix approximation 
    via factor expansion.

    If A ~ A1 @ A2.T, then A * A ~ A1_tilde @ A2_tilde.T, where each has rank r^2.

    Args:
        A1 (torch.Tensor): Low-rank factor of shape (n, r).
        A2 (torch.Tensor): Low-rank factor of shape (n, r).
        device (str, optional): The device on which to perform tensor operations. 
            Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - A1_tilde (torch.Tensor): First factor of shape (n, r^2).
            - A2_tilde (torch.Tensor): Second factor of shape (n, r^2).
    """
    
    A1 = A1.to(device)
    A2 = A2.to(device)
    n, r = A1.shape
    A1_tilde = torch.einsum("ij,ik->ijk", A1, A1).reshape(n, r * r)
    A2_tilde = torch.einsum("ij,ik->ijk", A2, A2).reshape(n, r * r)
    
    return A1_tilde, A2_tilde


def hadamard_lr(A1, A2, B1, B2, device='cpu'):
    
    """
    Computes a Hadamard (elementwise) product of two low-rank matrix approximations 
    via factor expansion.

    If A ~ A1 @ A2.T and B ~ B1 @ B2.T, then A * B ~ M1_tilde @ M2_tilde.T, 
    each factor having rank r^2.

    Args:
        A1 (torch.Tensor): Low-rank factor of shape (n, r) for matrix A.
        A2 (torch.Tensor): Low-rank factor of shape (n, r) for matrix A.
        B1 (torch.Tensor): Low-rank factor of shape (n, r) for matrix B.
        B2 (torch.Tensor): Low-rank factor of shape (n, r) for matrix B.
        device (str, optional): The device on which to perform tensor operations. 
            Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - M1_tilde (torch.Tensor): First factor of the Hadamard product, shape (n, r^2).
            - M2_tilde (torch.Tensor): Second factor of the Hadamard product, shape (n, r^2).
    """
    A1 = A1.to(device)
    A2 = A2.to(device)
    B1 = B1.to(device)
    B2 = B2.to(device)
    n, r = A1.shape

    M1_tilde = torch.einsum("ij,ik->ijk", A1, B1).reshape(n, r * r)
    M2_tilde = torch.einsum("ij,ik->ijk", A2, B2).reshape(n, r * r)
    
    return M1_tilde, M2_tilde

def estimate_max_norm(A, B):
    
    """
    Estimates the maximum product of column/row L2 norms for the factorization A @ B.

    Specifically, computes:
      max_{k} (||A[:,k]||_2 * ||B[k,:]||_2).

    Args:
        A (torch.Tensor): A factor tensor of shape (n, r).
        B (torch.Tensor): Another factor tensor of shape (r, n) or (r, m).

    Returns:
        torch.Tensor: The esimate, the maximum value of ||A[:,k]|| * ||B[k,:]|| across all k.
    """
    
    assert A.shape[1] == B.shape[0], 'Inner matrix dimensions must equal.'
    col_norms = torch.linalg.norm(A, axis=0)
    row_norms = torch.linalg.norm(B, axis=1)
    colrow_prods = col_norms*row_norms
    C_max = torch.max(colrow_prods)
    
    return C_max

def normalize_mats(V_C, U_C, M1_A, M1_B, M2_A, M2_B, device='cpu'):

    """
    Normalizes multiple sets of factor matrices to ensure consistent scaling.

    Steps:
      1. Compute norm constants for each factor pair using `estimate_max_norm`.
      2. Scale factors to have comparable magnitudes.
      3. Apply an extra scaling factor to (V_C, U_C) to align it with (M1_, M2_).

    Args:
        V_C (torch.Tensor): Factor for cross-slice distance (shape (n, r)).
        U_C (torch.Tensor): Factor for cross-slice distance (shape (r, m)).
        M1_A (torch.Tensor): Factor for slice 1 distance (shape (n, r')).
        M1_B (torch.Tensor): Factor for slice 1 distance (shape (r', n)).
        M2_A (torch.Tensor): Factor for slice 2 distance (shape (m, r'')).
        M2_B (torch.Tensor): Factor for slice 2 distance (shape (r'', m)).
        device (str, optional): The device for tensor operations. Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The normalized versions of (V_C, U_C, M1_A, M1_B, M2_A, M2_B).
    """
    
    n = V_C.shape[0]

    norm_constant1 = estimate_max_norm(V_C, U_C)**1/2
    norm_constant2 = estimate_max_norm(M1_A, M1_B)**1/2
    norm_constant3 = estimate_max_norm(M2_A, M2_B)**1/2
    
    V_C, U_C = V_C/norm_constant1, U_C/norm_constant1
    M1_A, M1_B = M1_A/norm_constant2, M1_B/norm_constant2
    M2_A, M2_B = M2_A/norm_constant2, M2_B/norm_constant3
    
    c = (norm_constant1 / norm_constant2**2)**1/2
    V_C, U_C = V_C*c, U_C*c
    
    return V_C, U_C, M1_A, M1_B, M2_A, M2_B

def scale_matrix_rows(matrix):
    """
    Scales each row of a NumPy array by the maximum L2 norm found across rows 
    to constrain row values to a common scale.

    Args:
        matrix (np.ndarray): Input array of shape (n, d).

    Returns:
        np.ndarray: Scaled array of shape (n, d) with each row scaled so that 
        the maximum row norm is 1.
    """
    # Calculate the L2 norm for each row
    norms = np.linalg.norm(matrix, axis=1)
    # Find the maximum norm
    max_norm = np.max(norms)
    # Avoid division by zero
    if max_norm == 0:
        return matrix
    # Scale each row
    matrix_scaled = matrix / max_norm
    return matrix_scaled

def load_data(filehandle_embryo, r=100, r2=15, eps=0.03, device='cpu', \
              feature_handle1 = 'slice1_feature.npy', feature_handle2 = 'slice2_feature.npy',
             spatial_handle1 = 'slice1_coordinates.npy', spatial_handle2 = 'slice2_coordinates.npy',  hadamard = True,
             normalize=False):

    """
    Loads two slices of data (features and spatial coordinates) from `.npy` files
    and computes low-rank factorizations for use in OT-based pipelines.

    Args:
        filehandle_embryo (str): A prefix or directory path to where `.npy` files 
            are stored.
        r (int, optional): Rank for cross-slice distance factorization. Defaults to 100.
        r2 (int, optional): Rank for within-slice distance factorization. Defaults to 15.
        eps (float, optional): Sampling parameter for `low_rank_distance_factorization`. 
            Defaults to 0.03.
        device (str, optional): Compute device ('cpu' or 'cuda'). Defaults to 'cpu'.
        feature_handle1 (str, optional): Filename for slice 1's feature array. 
            Defaults to 'slice1_feature.npy'.
        feature_handle2 (str, optional): Filename for slice 2's feature array. 
            Defaults to 'slice2_feature.npy'.
        spatial_handle1 (str, optional): Filename for slice 1's spatial coordinates. 
            Defaults to 'slice1_coordinates.npy'.
        spatial_handle2 (str, optional): Filename for slice 2's spatial coordinates. 
            Defaults to 'slice2_coordinates.npy'.
        hadamard (bool, optional): If True, will compute the Hadamard product of
            expression and spatial distances. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (V_C, U_C, M1_A, M1_B, M2_A, M2_B), where each pair of matrices 
            approximates a distance matrix in low-rank form.
    """
    
    data_t1 = torch.from_numpy(np.load(filehandle_embryo + feature_handle1)).to(device)
    data_t2 = torch.from_numpy(np.load(filehandle_embryo + feature_handle2)).to(device)
    
    spatial_t1 =  torch.from_numpy(np.load(filehandle_embryo + spatial_handle1)).to(device)
    spatial_t2 =  torch.from_numpy(np.load(filehandle_embryo + spatial_handle2)).to(device)
    
    # Factorize C
    V_C, U_C = low_rank_distance_factorization(data_t1, data_t2, r, eps, device=device)
    
    n, m = V_C.shape[0], U_C.shape[1]
    
    # M1 matrix
    V_A, U_A = low_rank_distance_factorization(data_t1, data_t1, r2, eps, device=device)
    V_B, U_B = low_rank_distance_factorization(spatial_t1, spatial_t1, r2, eps, device=device)
    
    if hadamard:
        M1_A, M1_B = hadamard_lr(V_A, U_A.T, V_B, U_B.T, device=device)
    else:
        # Default to spatial dist
        M1_A, M1_B = V_B, U_B.T
    
    # M2 matrix
    V_A, U_A = low_rank_distance_factorization(data_t2, data_t2, r2, eps, device=device)
    V_B, U_B = low_rank_distance_factorization(spatial_t2, spatial_t2, r2, eps, device=device)
    
    if hadamard:
        M2_A, M2_B = hadamard_lr(V_A, U_A.T, V_B, U_B.T, device=device)
    else:
        # Default to spatial dist
        M2_A, M2_B = V_B, U_B.T
    
    del V_A, U_A, V_B, U_B
    
    if normalize:
        V_C, U_C, M1_A, M1_B, M2_A, M2_B = normalize_mats(V_C, U_C, M1_A, M1_B.T, M2_A, M2_B.T, device=device)
        M1_B, M2_B = M1_B.T, M2_B.T
    
    return V_C, U_C, M1_A, M1_B, M2_A, M2_B


def load_data_svd(filehandle_embryo, r=100, r2=100, eps=0.03, device='cpu', \
              feature_handle1 = 'slice1_feature.npy', feature_handle2 = 'slice2_feature.npy',
             spatial_handle1 = 'slice1_coordinates.npy', spatial_handle2 = 'slice2_coordinates.npy',  hadamard = True):
    
    data_t1 = np.load(filehandle_embryo + feature_handle1)
    data_t2 = np.load(filehandle_embryo + feature_handle2)
    
    n, m = data_t1.shape[0], data_t2.shape[0]
    data_t1 = scale_matrix_rows(data_t1)
    data_t2 = scale_matrix_rows(data_t2)
    
    data_t1 = torch.from_numpy(data_t1).to(device)
    data_t2 = torch.from_numpy(data_t2).to(device)
    
    spatial_t1 =  torch.from_numpy(np.load(filehandle_embryo + spatial_handle1)).to(device)
    spatial_t2 =  torch.from_numpy(np.load(filehandle_embryo + spatial_handle2)).to(device)
    
    # Factorize C
    C = torch.cdist(data_t1, data_t2).to(device)
    u, s, v = torch.svd(C)
    print('C done')
    V_C,U_C = torch.mm(u[:,:r], torch.diag(s[:r])), v[:,:r].mT
    del C, u, s, v

    A = torch.cdist(data_t1, data_t1).to(device)*torch.cdist(spatial_t1, spatial_t1).to(device)
    u, s, v = torch.svd(A)
    print('A done')
    M1_A,M1_B = torch.mm(u[:,:r2], torch.diag(s[:r2])), v[:,:r2].mT
    del A, u, s, v

    B = torch.cdist(data_t2, data_t2).to(device)*torch.cdist(spatial_t2, spatial_t2).to(device)
    u, s, v = torch.svd(B)
    print('B done')
    M2_A, M2_B = torch.mm(u[:,:r2], torch.diag(s[:r2])), v[:,:r2].mT
    del B, u, s, v
    
    V_C, U_C, M1_A, M1_B, M2_A, M2_B = normalize_mats(V_C, U_C, M1_A, M1_B, M2_A, M2_B)
    return V_C, U_C, M1_A, M1_B, M2_A, M2_B

def get_clusters(Qs, Rs, Ts, spatial_list, ancestral=True):
    cluster_label_list = []
    
    for i in range(len(Qs)):
        # Use labels of intermediate slice from previous pair as reference
        # First slice, intermediate labels
        if ancestral:
            ml_labels_W, ml_labels_H = clustering.ancestral_clustering(Qs[i].cpu().numpy(),Rs[i].cpu().numpy(),Ts[i].cpu().numpy(), full_P=False)
        else:
            ml_labels_W, ml_labels_H = clustering.max_likelihood_clustering(Qs[i].cpu().numpy(),Rs[i].cpu().numpy())
        '''
        clustering.plot_cluster_pair(spatial_list[i-1],
                                     spatial_list[i],
                                     ml_labels_W,
                                     ml_labels_H, dotsize=100)
        '''
        if i == 0:
            cluster_label_list.append(ml_labels_W)
            cluster_label_list.append(ml_labels_H)
        else:
            ml_labels_H = cluster_label_list[-1]
            # Intermediate slice, second slice labels
            if ancestral:
                ml_labels_W1, ml_labels_H1 = clustering.ancestral_clustering(Qs[i].cpu().numpy(),Rs[i].cpu().numpy(),Ts[i].cpu().numpy(), full_P=False)
            else:
                ml_labels_W1, ml_labels_H1 = clustering.max_likelihood_clustering(Qs[i].cpu().numpy(),Rs[i].cpu().numpy())
            
            # 1-1 correspondence between labels from intermediates
            label_dict = { }
            for j, label in enumerate(ml_labels_W1):
                label_dict[label] = ml_labels_H[j]
            
            for k, label in enumerate(ml_labels_H1):
                if label in label_dict:
                    ml_labels_H1[k] = label_dict[label]
                # Else, have differentiation
            print(f'AMI of intermediates (should be 1.0): {adjusted_mutual_info_score(ml_labels_H, ml_labels_W1)}')
            cluster_label_list.append(ml_labels_H1)
            
    return cluster_label_list