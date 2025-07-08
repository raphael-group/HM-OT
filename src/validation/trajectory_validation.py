"""
src/validation/trajectory_validation.py

Validation utilities for cell trajectory coupling and clustering comparison.

Public functions:
- compute_pw_coupling_wot
- compute_global_couplings_gwot
- P_to_T
- compute_Ts_pairwise
- extract_KMeans_Qs
- analyze_noise_level
"""

import numpy as np
import scipy
scipy.matrix = np.matrix
import torch
from gwot import models
from scipy.spatial.distance import cdist
import ot
import matplotlib.pyplot as plt
import src.HiddenMarkovOT as HiddenMarkovOT
import src.utils.clustering as clustering
import src.utils.util_LR as util_LR
import src.plotting as plotting
from sklearn.cluster import KMeans
import matplotlib as mpl
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
from src.utils.waddington.metrics import frac_correct_latent_trajectory as latent_traj_acc


class DataSim:
    """Minimal sim-like object for multiple timepoints."""
    pass

def compute_pw_coupling_wot(X1, X2,
                            eps_df=0.025,
                            lamda_reg= 1.0,
                            dtype_np=np.float32):
    """
    Pairwise entropic OT coupling (WOT) between two snapshots.
    Returns P matrix of shape (N1, N2).
    """
    N1, N2 = len(X1), len(X2)
    
    p = np.full(N1, 1.0/N1)
    q = np.full(N2, 1.0/N2)

    # squared-Euclidean cost
    C = cdist(X1, X2, 'sqeuclidean') * lamda_reg
    
    # Sinkhorn
    P = ot.sinkhorn(a=p, b=q, M=C, reg=eps_df)
    return P.astype(dtype_np)

def compute_global_couplings_gwot(
    Xs, dt,
    eps_df   = 0.025,
    lamda_reg= 1.0,
    device='cpu',
    dtype=torch.float32,
    dtype_np=np.float32,
    D = 1.0
):
    """
    Compute entropic OT couplings across all timepoints in Xs (global WOT).
    Returns list of gamma matrices for each consecutive pair.
    
    Parameters
    ----------
    Xs : list of np.ndarray, each (Ni, d)
    dt : float or list of floats, spacing(s) between snapshots

    Returns
    -------
    couplings : list of np.ndarray
        List of length T-1 of coupling matrices (Ni x Ni+1).
    """
    torch.set_default_dtype(dtype)
    
    T = len(Xs)
    sim = DataSim()
    sim.T = T
    sim.N = np.array([X.shape[0] for X in Xs])
    sim.d = Xs[0].shape[1]
    sim.x = np.vstack(Xs).astype(dtype_np)
    sim.D = D

    # prepare dt array
    sim.dt = np.array([dt] * (T-1), dtype=dtype_np)
    sim.t_idx = np.concatenate([np.full(n, i, dtype=int) for i, n in enumerate(sim.N)])
    
    if np.isscalar(dt):
        dt_arr = [dt] * (T-1)
    else:
        dt_arr = list(dt)
    
    sim.dt = np.array(dt_arr, dtype=dtype_np)
    
    # instantiate OTModel for null-growth across all times
    eps_tensor = eps_df * torch.ones(sim.T, device=device)
    # instantiate GWOT model
    model = models.OTModel(
        sim,
        lamda_reg        = lamda_reg,
        eps_df           = eps_tensor,
        growth_constraint= "exact",
        pi_0             = "uniform",
        use_keops        = False,
        device           = device
    )
    # solve globally
    model.solve_lbfgs(
        steps          = 25,
        max_iter       = 100,
        lr             = 1.0,
        history_size   = 50,
        line_search_fn = "strong_wolfe",
        tol            = 1e-7,
        retry_max      = 10
    )
    
    # extract all couplings
    couplings: list[np.ndarray] = []
    for i in range(T-1):
        Ki    = model.get_K(i)
        gamma = model.get_coupling_reg(i , Ki)
        couplings.append(gamma.detach().cpu().numpy())
    return couplings


def P_to_T(P, label_t0, label_t1, row_norm=False):
    """
    Coarsen a fine-grained coupling P (Ni x Nj) into a cell-type transition coupling T.

    T[a,b] = P(i->j) summed over all i with label a at time0 and j with label b at time1,
    normalized so that transitions are scaled to be joint distribution (coupling).
    
    Returns
    -------
    T : np.ndarray, shape (K0, K1)
        Coarse-grained transition coupling.
    labels0 : np.ndarray
        Unique labels at t0 (row ordering).
    labels1 : np.ndarray
        Unique labels at t1 (column ordering).
    row_norm: bool (Default: False)
        Boolean for whether to row-normalize for T to be a transition matrix (if True),
        or whether to return it as a joint law (coupling) (if False).
    """
    labels0 = np.unique(label_t0)
    labels1 = np.unique(label_t1)
    T      = np.zeros((labels0.size, labels1.size), dtype=float)
    for i, a in enumerate(labels0):
        idx0 = np.where(label_t0 == a)[0]
        mass = P[idx0, :].sum()
        for j, b in enumerate(labels1):
            idx1    = np.where(label_t1 == b)[0]
            T[i, j] = P[np.ix_(idx0, idx1)].sum()
        if mass > 0 and row_norm:
            T[i, :] /= mass
    if not row_norm:
        # Scaled for T to be a coupling
        T = T / np.sum(T)
    return T, labels0, labels1


def compute_Ts_pairwise(
    Xs: list[np.ndarray],
    klabel_lst: list[np.ndarray],
    method: str = 'WOT',
    dt: float = 1.0,
    eps_df: float = 0.025,
    lamda_reg: float = 0.00215,
    D: float = 1.0,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    dtype_np: type = np.float32
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Compute coarse-grained cell-type transitions for each pair of snapshots.
    Supports 'WOT' (pairwise) and 'gWOT' (global) methods.
    
    Returns
    -------
    Ts : list of np.ndarray
        Each element is the coarse T matrix between time i and i+1.
    labels : list of tuples
        Each tuple (labels0, labels1) gives the label ordering for that T.
    """
    Ts: list[np.ndarray] = []
    labels: list[tuple[np.ndarray, np.ndarray]] = []
    if method == 'WOT':
        for i in range(len(Xs)-1):
            X1, X2      = Xs[i], Xs[i+1]
            lab0, lab1  = klabel_lst[i], klabel_lst[i+1]
            P           = compute_pw_coupling_wot(
                              X1, X2,
                              eps_df=eps_df,
                              lamda_reg=lamda_reg,
                                dtype_np=dtype_np)
            
            Tmat, l0, l1= P_to_T(P, lab0, lab1)
            Ts.append(Tmat)
            labels.append((l0, l1))
    elif method == 'gWOT':
        # compute all fine-grained couplings at once
        Ps = compute_global_couplings_gwot(
                 Xs, dt,
                 eps_df=eps_df,
                 lamda_reg=lamda_reg,
                 device=device,
                 dtype=dtype,
                dtype_np=dtype_np,
                D=D)
        for i, P in enumerate(Ps):
            lab0, lab1 = klabel_lst[i], klabel_lst[i+1]
            Tmat, l0, l1 = P_to_T(P, lab0, lab1)
            Ts.append(Tmat)
            labels.append((l0, l1))
    else:
        raise ValueError(f"Unknown method: {method}")
    return Ts, labels


def extract_KMeans_Qs(
        Ss: list[np.ndarray],
        K: int = 3,
        seed: int = 42
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run KMeans on each snapshot, return Q matrices and labels.
    """
    kmeans = [KMeans(n_clusters=K, random_state=seed) for _ in Ss]
    Qs_ann = []
    klabel_lst = []
    
    for Si, km in zip(Ss, kmeans):
        lab = km.fit_predict(Si)
        Qi = np.eye(K)[lab] / K
        klabel_lst.append(lab)
        Qs_ann.append(Qi)
    
    Ts = [None]*(len(Qs_ann)-1)
    plotting.plot_clusters_from_QT(Ss, Qs_ann, Ts, dotsize=500)
    
    return Qs_ann, klabel_lst

def analyze_noise_level(
    D,
    data_snap,
    snap_times,
    num_traj,
    device,
    dtype=torch.float32,
    dtype_np=np.float32,
    rank_list=None,
    iter=30,
    gamma=80.0,
    alpha=0.0,
    tau_in=10,
    seed=42,
    _plotting=True
):
    """
    Full pipeline: HM-OT, WOT, and gWOT evaluation for a given noise level.
    Returns (frac_correct_hmot, frac_correct_wot, frac_correct_gwot).
    """
    
    # Set new seed per iteration (avoids common initialization across iterations)
    torch.set_default_dtype(dtype)
    
    frac_correct_hmot = []
    frac_correct_wot = []
    frac_correct_gwot = []
    
    # 1) proportions
    last = data_snap[f'step_{snap_times[-1]}']
    p1 = np.mean(last[:,0] < 0)
    p2 = np.mean((last[:,0] > 0) & (last[:,1] > 0))
    props = torch.tensor([p1, p2, 1 - p1 - p2],
                         device=device, dtype=dtype)
    # make a full list of marginals
    props_list = [props]
    props_list.extend([None]*(len(snap_times)-1))
    
    # 2) build cost factors C12 (in double) + identity I (in double)
    C_factors_sequence = []
    for i in range(len(snap_times)-1):
        tp1 = data_snap[f'step_{snap_times[i]}']
        tp2 = data_snap[f'step_{snap_times[i+1]}']
        X1 = torch.from_numpy(tp1).to(device=device, dtype=dtype)
        X2 = torch.from_numpy(tp2).to(device=device, dtype=dtype)
        C12 = torch.cdist(X1, X2)**2
        # now cast the cost to match dtype (float64)
        C12 = C12.to(dtype)
        I   = torch.eye(C12.shape[1], device=device, dtype=dtype)
        C_factors_sequence.append((C12, I))
    
    Ss      = [data_snap[f'step_{t}'] for t in snap_times]
    Qs_ann, klabel_lst = extract_KMeans_Qs(Ss, K = 3, seed=42)
    
    plotting.plot_clusters_from_QT(Ss, Qs_ann, [None]*len(Ss), dotsize=500)
    plt.suptitle(f"D={D}: annotated clusters"); plt.show()
    
    Ss_all  = np.vstack(Ss)

    if _plotting:
        X1, X2, X3 = Ss[0], Ss[1], Ss[2]
        data   = [X1, X2, X3]
        colors = ['#1f77b4', '#ff7f0e', '#d62728'] 
        labels = ['Timepoint 1', 'Timepoint 2', 'Timepoint 3']
        letters = ['a', 'b', 'c']
        mpl.rcParams.update({
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.linewidth': 0.8,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.minor.visible': True,
            'xtick.minor.size': 1.5,
            'ytick.minor.visible': True,
            'ytick.minor.size': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharex=True, sharey=True, constrained_layout=True, dpi=300)
        for ax, X, color, label, letter in zip(axes, data, colors, labels, letters):
            ax.scatter(X[:, 0], X[:, 1], c=color, s=15, alpha=0.8, edgecolor='none')
            ax.set_title(f'({letter}) {label}', fontsize=10, pad=4)
            ax.set_aspect('equal', 'box')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        # Shared axis labels
        fig.supxlabel('Component 1', fontsize=12)
        fig.supylabel('Component 2', fontsize=12)
        # Save or show
        plt.show()
    
    n_per   = Ss[0].shape[0]
    time_labels = np.concatenate([[i]*n_per for i in range(len(snap_times))])
    df = {"x": Ss_all[:,0], "y": Ss_all[:,1], "timepoint": time_labels}
    
    '''
    HM-OT Unsupervised
    '''
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    
    hmot = HiddenMarkovOT.HM_OT(
        rank_list=rank_list,
        max_iter=iter, min_iter=iter,
        device=device,
        alpha=alpha,
        gamma=gamma,
        dtype=dtype,
        printCost=True,
        returnFull=False,
        initialization='Full',
        tau_in=tau_in,
        max_inner_iters_R=300,
        generator=gen,
        proportions=props_list
    )
    A_seq = [None] * len(snap_times)
    hmot.gamma_smoothing(C_factors_sequence, A_seq)
    
    Qs = [Q.cpu().detach().numpy() for Q in hmot.Q_gammas]
    Ts = [T.cpu().detach().numpy() for T in hmot.T_gammas]
    
    plotting.plot_clusters_from_QT(
        Ss, Qs, Ts, None,
        clustering_type='reference',
        reference_index=0,
        flip=False,
        dotsize=50, key_dotsize=1
    )
    plt.suptitle(f"D={D}: reference at t=0");  plt.show()
    
    plotting.plot_clusters_from_QT(
        Ss, Qs, Ts, None,
        clustering_type='reference',
        reference_index=len(Qs)-1,
        flip=False,
        dotsize=50, key_dotsize=1
    )
    plt.suptitle(f"D={D}: reference at final t"); plt.show()
    
    plotting.plot_diffmap_clusters_prime(
        X=Ss_all,
        time_labels=time_labels,
        Qs=Qs,
        Ts=Ts,
        df=df,
        cluster_key="cluster_pred"
    )
    plt.suptitle(f"D={D}: diffmap_pred"); plt.show()
    
    '''
    WOT Supervised on Clusters
    '''
    
    dt = snap_times[1] - snap_times[0]
    Ts_wot, labs_pair = compute_Ts_pairwise(Ss, klabel_lst, method='WOT', 
                                            dt=dt, dtype=dtype, 
                                            dtype_np=dtype_np, eps_df=0.0001)
    
    plotting.plot_diffmap_clusters_prime(
        X=Ss_all,
        time_labels=time_labels,
        Qs=Qs_ann,
        Ts=Ts_wot,
        df=df,
        cluster_key="cluster_wot"
    )
    plt.suptitle(f"D={D}: diffmap_wot"); plt.show()
    
    '''
    gWOT Supervised on Clusters (needs float64 to work)
    '''
    Ts_gwot, labs_glob = compute_Ts_pairwise(Ss, klabel_lst, method='gWOT', 
                                             dt=dt, dtype=torch.float64, 
                                             dtype_np=np.float64, eps_df=0.0001,
                                                D=dt*D)
    
    plotting.plot_diffmap_clusters_prime(
        X=Ss_all,
        time_labels=time_labels,
        Qs=Qs_ann,
        Ts=Ts_gwot,
        df=df,
        cluster_key="cluster_gwot"
    )
    plt.suptitle(f"D={D}: diffmap_gwot"); plt.show()
    
    '''
    Compute fractions
    '''
    clustering_wot = clustering.reference_clustering(
                                Qs_ann,
                                Ts_wot,
                                reference_index=len(Qs_ann)-1)
    frac_wot = latent_traj_acc(clustering_wot)
    clustering_hmot = clustering.reference_clustering(
                                Qs,
                                Ts,
                                reference_index=len(Qs)-1)
    print(f'AMI at final time between K-means and HM-OT clusters: {ami(clustering_hmot[-1], clustering_wot[-1])}')
    frac_hmot = latent_traj_acc(clustering_hmot)
    clustering_gwot = clustering.reference_clustering(
                                Qs_ann,
                                Ts_gwot,
                                reference_index=len(Qs_ann)-1)
    frac_gwot = latent_traj_acc(clustering_gwot)
    
    print(f'Frac Correct (HM-OT): {frac_hmot}, (WOT): {frac_wot}, (gWOT): {frac_gwot}')
    
    cost_hmot = util_LR.compute_total_cost(C_factors_sequence,
                       A_seq,
                       hmot.Q_gammas,
                       hmot.T_gammas,
                       device=device)
    cost_wot = util_LR.compute_total_cost(C_factors_sequence,
                       A_seq,
                       [torch.tensor(Q).to(device).to(dtype) for Q in Qs_ann],
                       [torch.tensor(T).to(device).to(dtype) for T in Ts_wot],
                       device=device)
    cost_gwot = util_LR.compute_total_cost(C_factors_sequence,
                       A_seq,
                       [torch.tensor(Q).to(device).to(dtype) for Q in Qs_ann],
                       [torch.tensor(T).to(device).to(dtype) for T in Ts_gwot],
                       device=device)
    
    print(f"OT Costs:\n  HM-OT: {cost_hmot:.4f}\n  WOT:   {cost_wot:.4f}\n  gWOT:  {cost_gwot:.4f}")
    
    return {
                'frac': {'hmot': frac_hmot, 'wot': frac_wot, 'gwot': frac_gwot},
                'cost': {'hmot': cost_hmot, 'wot': cost_wot, 'gwot': cost_gwot}
            }


__all__ = [
    "compute_pw_coupling_wot",
    "compute_global_couplings_gwot",
    "P_to_T",
    "compute_Ts_pairwise",
    "extract_KMeans_Qs",
    "analyze_noise_level"
]
    