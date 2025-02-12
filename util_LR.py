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
    return (tup[0] / c, tup[1] / c)

def _compute_Q(_adata, cell_type_key):
    
    encoder = OneHotEncoder(sparse_output=False)
    ys_onehot = encoder.fit_transform(_adata.obs[cell_type_key].values.reshape(-1, 1))
    _Q = ys_onehot / np.sum(ys_onehot)
    label = list(encoder.categories_[0])
    
    return _Q, label

def convert_adata(adata, replicates, timepoints, \
                  cell_type_key = 'cellstate', timepoint_key = 'timepoint', \
                  replicate_key = 'orig.ident', spatial_key = ['x_loc', 'y_loc'], \
                  feature_key = 'X_pca', dtype=torch.float32, compute_Q = True,
                  spatial = True, dist_rank=100, dist_eps=0.02
                 ):
    
    N = len(replicates)
    
    A_factors_sequence = []
    
    features_sequence = []
    spatial_sequence = []
    
    adata_replicates = []
    Qs = [None]*N
    labels = [None]*N
    
    for idx, (rep, time) in enumerate(zip(replicates, timepoints)):
        
        adata_t = adata[adata.obs[timepoint_key] == time]
        adata_rep = adata_t[adata_t.obs[replicate_key] == rep]
        adata_replicates.append(adata_rep)
        
        if spatial:
            ### Compute low-rank approximation to the distance matrix and appropriately normalize
            spatial_coords = torch.tensor( adata_rep.obs[spatial_key].to_numpy(), dtype=dtype )
            A_rep = low_rank_distance_factorization( spatial_coords, spatial_coords, dist_rank, dist_eps )
            c = estimate_max_norm( A_rep[0] , A_rep[1] )
            A_rep = norm_factors( A_rep, c**1/2 )
            A_factors_sequence.append( (A_rep[0].to(dtype), A_rep[1].to(dtype)) )
            spatial_sequence.append(spatial_coords.numpy())
        else:
            A_factors_sequence.append( (None, None) )
        
        feature_coords = torch.tensor(adata_rep.obsm[feature_key], dtype=dtype)
        features_sequence.append(feature_coords)
        
        if compute_Q:
            _Q, label = _compute_Q(adata_rep, cell_type_key=cell_type_key)
            Qs[idx] = torch.tensor(_Q, dtype=dtype)
            labels[idx] = label
    
    C_factors_sequence = []
    rank_list = []
    
    for i in range(0, N-1, 1):
        
        idx1, idx2 = i, i+1
        
        features1, features2 = features_sequence[idx1], features_sequence[idx2]
        C_rep = low_rank_distance_factorization( features1, features2, dist_rank, dist_eps )
        c = estimate_max_norm( C_rep[0] , C_rep[1] )
        C_rep = norm_factors( C_rep, c**1/2 )
        C_factors_sequence.append( (C_rep[0].to(dtype), C_rep[1].to(dtype)))

        rank_list.append( (len(labels[idx1]), len(labels[idx2]) ) )
    
    return C_factors_sequence, A_factors_sequence, Qs, labels, rank_list, spatial_sequence



def low_rank_distance_factorization(X1, X2, r, eps, device='cpu', dtype=torch.float64):
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
    Input
        A1: torch.tensor, low-rank subcoupling of shape (n, r)
        A2: torch.tensor, low-rank subcoupling of shape (n, r)
                ( such that A \approx A1 @ A2.T )
    
    Output
        A1_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
        A2_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
               ( such that A * A \approx A1_tilde @ A2_tilde.T )
    """
    
    A1 = A1.to(device)
    A2 = A2.to(device)
    n, r = A1.shape
    A1_tilde = torch.einsum("ij,ik->ijk", A1, A1).reshape(n, r * r)
    A2_tilde = torch.einsum("ij,ik->ijk", A2, A2).reshape(n, r * r)
    
    return A1_tilde, A2_tilde


def hadamard_lr(A1, A2, B1, B2, device='cpu'):
    """
    Input
        A1: torch.tensor, low-rank subcoupling of shape (n, r)
        A2: torch.tensor, low-rank subcoupling of shape (n, r)
                ( such that A \approx A1 @ A2.T )
        
        B1: torch.tensor, low-rank subcoupling of shape (n, r)
        B2: torch.tensor, low-rank subcoupling of shape (n, r)
                ( such that B \approx B1 @ B2.T )
    
    Output
        M1_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
        M2_tilde: torch.tensor, low-rank subcoupling of shape (n, r**2)
               ( such that A * B \approx M1_tilde @ M2_tilde.T given low-rank factorizations for A & B)
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
    
    assert A.shape[1] == B.shape[0], 'Inner matrix dimensions must equal.'
    col_norms = torch.linalg.norm(A, axis=0)
    row_norms = torch.linalg.norm(B, axis=1)
    colrow_prods = col_norms*row_norms
    C_max = torch.max(colrow_prods)
    
    return C_max

def normalize_mats(V_C, U_C, M1_A, M1_B, M2_A, M2_B, device='cpu'):
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
             spatial_handle1 = 'slice1_coordinates.npy', spatial_handle2 = 'slice2_coordinates.npy',  hadamard = True):
    
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