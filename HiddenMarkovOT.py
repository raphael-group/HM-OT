import sys
import os

sys.path.append(os.path.abspath("FRLC"))

import torch
import matplotlib.pyplot as plt
import util_LR
from FRLC import FRLC_LR_opt
from FRLC_multimarginal import FRLC_LR_opt_multimarginal

class HM_OT:

    """
    A class to compute a full clustering (Q/Lambda) sequence and transition (T) sequence to 
    minimize a joint Wasserstein objective across multiple steps (or time points).
    
    This class implements a Forward-Backward-like procedure (alpha pass and beta pass) for 
    low-rank factored regularized linear cost (FRLC) optimal transport. It then performs 
    a gamma smoothing step to refine the clustering matrices (Q) and transitions (T).
    
    Attributes:
        N (int): 
            Number of transitions (or time steps) inferred from `rank_list` length.
        a (torch.Tensor, optional): 
            First marginal of the optimal transport problem (if needed).
        b (torch.Tensor, optional): 
            Second marginal of the optimal transport problem (if needed).
        tau_in (float): 
            Inner marginal parameter for FRLC optimization.
        tau_out (int): 
            Outer marginal parameter for FRLC optimization.
        gamma (float): 
            Entropic regularization parameter (step-size) for the FRLC coordinate mirror-descent problem.
        max_iter (int): 
            Maximum number of iterations for each FRLC solver call.
        min_iter (int): 
            Minimum number of iterations for each FRLC solver call.
        device (str): 
            PyTorch device used for tensor operations ('cpu' or 'cuda').
        dtype (torch.dtype): 
            Data type used for PyTorch tensors.
        printCost (bool): 
            Whether to print the cost at each iteration.
        returnFull (bool): 
            Whether to return the full transport plan or just factors.
        alpha (float): 
            Mixture weight parameter for FRLC optimization (e.g., a ratio between W and GW terms).
        initialization (str): 
            Method to initialize the low-rank factors for the solver. 
                              E.g., 'Full' or any custom initialization.
        proportions (list of type torch.Tensor, optional)
            A list of cluster proportions for each timepoint, e.g. if one wants to specify rare cell-types.
        
        Q_alphas (list): Stores the forward pass Q/R clusterings.
        T_alphas (list): Stores the forward pass transition matrices.
        Q_betas (list): Stores the backward pass Q/R clusterings.
        T_betas (list): Stores the backward pass transition matrices.
        Q_gammas (list): Stores the smoothed clusterings after alpha-beta passes.
        T_gammas (list): Stores the smoothed transition matrices after gamma smoothing.
        errs (dict): Tracks the error/cost values during smoothing, with keys:
                     'total_cost', 'W_cost', 'GW_cost'.
                
    Example:
        >>> # Suppose you have a list of rank pairs, cost factors, and distribution factors
        >>> rank_list = [(10, 8), (8, 8), (8, 6)]  # For each transition
        >>> C_factors_sequence = [C0, C1, C2]     # Cost factors for each step
        >>> A_factors_sequence = [A0, A1, A2, A3] # Factor sets for consecutive distributions
        >>> hm_ot = HM_OT(rank_list, a=a, b=b, tau_in=0.0001, tau_out=75, 
        ...               gamma=90, max_iter=200, device='cpu')
        >>> hm_ot.gamma_smoothing(C_factors_sequence, A_factors_sequence)
        >>> # Access smoothed Q and T
        >>> Qs = hm_ot.Q_gammas
        >>> Ts = hm_ot.T_gammas
    """
    
    # Alpha-pass variables
    Q_alphas = []
    T_alphas = []
    
    # Beta-pass variables
    Q_betas = []
    T_betas = []
    
    # Gamma-smoothed variables
    Q_gammas = []
    T_gammas = []
    
    errs = {'total_cost':[], 'W_cost':[], 'GW_cost': []}
    
    def __init__(self,
                 rank_list,
                 a=None,
                 b=None,
                 tau_in = 0.0001,
                 tau_out=75,
                 gamma=90,
                 max_iter=200,
                 min_iter=200,
                 device='cpu',
                 dtype=torch.float64,
                 printCost=True,
                 returnFull=True,
                 alpha=0.2,
                 initialization='Full',
                 active_Qs = None,
                 proportions = None):

        """
        Initializes the HM_OT class with the given parameters.
        
        Args:
            rank_list (list of tuples): 
                Each tuple (r1, r2) defines the low-rank factors 
                    dimensions for consecutive distributions at time t.
            a (torch.Tensor, optional): 
                First marginal distribution vector.
            b (torch.Tensor, optional): 
                Second marginal distribution vector.
            tau_in (float, optional): 
                Inner marginal regularization parameter for FRLC optimization. Defaults to 0.0001.
            tau_out (int, optional): 
                Outer marginal regularization parameter FRLC optimization. Defaults to 75.
            gamma (float, optional): 
                Entropic regularization parameter / step-size for the FRLC problem. Defaults to 90.
            max_iter (int, optional): 
                Maximum number of iterations for the FRLC solver. Defaults to 200.
            min_iter (int, optional): 
                Minimum number of iterations for the FRLC solver. Defaults to 200.
            device (str, optional): 
                Device for PyTorch tensors. Defaults to 'cpu'.
            dtype (torch.dtype, optional): 
                Data type for PyTorch tensors. Defaults to torch.float64.
            printCost (bool, optional): 
                Whether to print cost info at each iteration. Defaults to True.
            returnFull (bool, optional): 
                Whether to return full transport plan or just factors. Defaults to True.
            alpha (float, optional): 
                Mixture weight for combining W and GW costs in FRLC. Defaults to 0.2.
            initialization (str, optional): 
                Strategy for initializing the low-rank factors. Defaults to 'Full'.
            proportions (list of type torch.Tensor, optional)
                A list of cluster proportions for each timepoint, e.g. if one wants to specify rare cell-types.
        """
        
        self.rank_list = rank_list
        self.N = len(self.rank_list)
        self.a = a
        self.b = b
        self.tau_in = tau_in
        self.tau_out = tau_out
        self.gamma = gamma
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.device = device
        self.dtype = dtype
        self.printCost = printCost
        self.returnFull = returnFull
        self.alpha = alpha
        self.initialization = initialization

        if proportions is None:
            self.proportions = (self.N+1)*[None]
        else:
            self.proportions = proportions
        
    
    def alpha_pass(self, 
                   C_factors_sequence, 
                   A_factors_sequence, 
                   Q_IC = None):
        """
        Executes the forward (alpha) pass to compute clusterings (Q) and transitions (T).
        
        The alpha pass starts from the initial time step and moves forward, 
        using `FRLC_LR_opt` to optimize Q, R, and T (or partial subsets, depending 
        on the iteration).
        
        Args:
            C_factors_sequence (list): List of tuples of low-rank cost factor tensors for each pair of time points (expression).
            A_factors_sequence (list): List of tuples of low-rank cost factor tensors for each pair of time points (spatial).
        
        Returns:
            None. Populates `self.Q_alphas` and `self.T_alphas` in place.
        """
        
        self.Q_alphas = []
        self.T_alphas = []

        C_factors, A_factors, B_factors = C_factors_sequence[0], A_factors_sequence[0], A_factors_sequence[1]
        
        r1, r2 = self.rank_list[0]

        # Whether a boundary condition is set for time 1
        if Q_IC is None:
            init_args=(None,None,None)
            updateQ = True
        else:
            init_args=(Q_IC,None,None)
            updateQ = False
        
        Q,R,T, errs = FRLC_LR_opt(C_factors,
                                  A_factors,
                                  B_factors,
                                  a=self.a,
                                  b=self.b,
                                  r=r1,
                                  r2=r2,
                                  max_iter=self.max_iter,
                                  device=self.device,
                                  returnFull=self.returnFull,
                                  alpha=self.alpha,
                                  min_iter=self.min_iter,
                                  initialization=self.initialization,
                                  tau_out=self.tau_out,
                                  tau_in=self.tau_in,
                                  gamma=self.gamma,
                                  dtype=self.dtype,
                                  updateR = True,
                                  updateQ = updateQ,
                                  updateT = True,
                                  init_args=init_args,
                                  printCost=self.printCost,
                                 _gQ=self.proportions[0],
                                 _gR=self.proportions[1])
        
        
        self.Q_alphas.append(Q)
        self.Q_alphas.append(R)
        self.T_alphas.append(T)
        
        for i in range(1, self.N-1, 1):

            C_factors, A_factors, B_factors = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            r1, r2 = self.rank_list[i]
            
            Q0 = self.Q_alphas[-1]
            init_args = (Q0, None, None)
            
            Q,R,T, errs = FRLC_LR_opt(C_factors, 
                                      A_factors,
                                      B_factors, 
                                      a=self.a, 
                                      b=self.b,
                                      r=r1,
                                      r2=r2, 
                                      max_iter=self.max_iter, 
                                      device=self.device, 
                                      returnFull=self.returnFull, alpha=self.alpha, 
                                      min_iter=self.min_iter, 
                                      initialization=self.initialization,
                                      tau_out=self.tau_out,
                                      tau_in=self.tau_in,
                                      gamma=self.gamma,
                                      dtype=self.dtype,
                                      updateR = True,
                                      updateQ = False,
                                      updateT = True,
                                      init_args=init_args,
                                      printCost=self.printCost,
                                     _gQ=self.proportions[i],
                                     _gR=self.proportions[i+1])
            
            self.Q_alphas.append(R)
            self.T_alphas.append(T)
            
        return

    
    def beta_pass(self, 
                  C_factors_sequence, 
                  A_factors_sequence,
                  R_TC = None):
        
        """
        Executes the backward (beta) pass to compute clusterings (Q) and transitions (T).
        
        The beta pass starts from the final time step and moves backward, using `FRLC_LR_opt`
        to optimize Q, R, and T (or partial subsets, depending on the iteration).

        Args:
            C_factors_sequence (list): List of tuples of low-rank cost factor tensors for each pair of time points (expression).
            A_factors_sequence (list): List of tuples of low-rank cost factor tensors for each pair of time points (spatial).
        
        Returns:
            None. Populates `self.Q_betas` and `self.T_betas` in place.
        """
        
        self.Q_betas = []
        self.T_betas = []
        
        C_factors, A_factors, B_factors = C_factors_sequence[self.N-1], A_factors_sequence[self.N-1], A_factors_sequence[self.N]
        r1, r2 = self.rank_list[self.N-1]

        if R_TC is None:
            init_args = (None, None, None)
            update_R = True
        else:
            init_args = (None, None, R_TC)
            update_R = False
        
        Q,R,T, errs = FRLC_LR_opt(C_factors,
                                  A_factors, 
                                  B_factors, 
                                  a = self.a, 
                                  b = self.b,
                                  r = r1,
                                  r2 = r2,
                                  max_iter = self.max_iter,
                                  device = self.device,
                                  returnFull = self.returnFull,
                                  alpha = self.alpha,
                                  min_iter = self.min_iter,
                                  initialization = self.initialization,
                                  tau_out = self.tau_out,
                                  tau_in = self.tau_in,
                                  gamma = self.gamma,
                                  dtype = self.dtype,
                                  updateR = update_R,
                                  updateQ = True,
                                  updateT = True,
                                  init_args = init_args,
                                  printCost = self.printCost,
                                 _gQ = self.proportions[self.N-1],
                                 _gR = self.proportions[self.N])
        
        self.Q_betas.append(R)
        self.Q_betas.append(Q)
        self.T_betas.append(T)
        
        for i in range(self.N-2, -1, -1):
            
            C_factors, A_factors, B_factors = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            r1, r2 = self.rank_list[i]
            
            R0 = self.Q_betas[-1]
            
            init_args = (None, R0, None)
            
            Q,R,T, errs = FRLC_LR_opt(C_factors,
                                      A_factors,
                                      B_factors,
                                      a=self.a,
                                      b=self.b,
                                      r=r1,
                                      r2=r2,
                                      max_iter=self.max_iter,
                                      device=self.device,
                                      returnFull=self.returnFull,
                                      alpha=self.alpha,
                                      min_iter=self.min_iter,
                                      initialization=self.initialization,
                                      tau_out=self.tau_out,
                                      tau_in=self.tau_in,
                                      gamma=self.gamma,
                                      dtype=self.dtype,
                                      updateR = False,
                                      updateQ = True,
                                      updateT = True,
                                      init_args=init_args,
                                      printCost=self.printCost,
                                     _gQ=self.proportions[i],
                                     _gR=self.proportions[i+1])
            
            self.Q_betas.append(Q)
            self.T_betas.append(T)
            
        return

    
    def impute_smoothed_transitions(self, C_factors_sequence, A_factors_sequence):

        """
        Given a finalized sequence of clusterings `self.Q_gammas`, computes the 
        transition matrices (T) between each pair of consecutive clusterings.
        
        Args:
            C_factors_sequence (list): List of tuples of low-rank cost factor tensors for each pair of time points (expression).
            A_factors_sequence (list): List of tuples of low-rank cost factor tensors for each pair of time points (spatial).
        
        Returns:
            None. Populates `self.T_gammas` in place and updates `self.errs` with cost values.
        """
        
        self.T_gammas = []
        self.errs = {'total_cost':[],
                     'W_cost':[],
                     'GW_cost': []}
        
        for i in range(0, self.N, 1):
            
            r1, r2 = self.rank_list[i]
            C_factors, A_factors, B_factors = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            
            Q0 = self.Q_gammas[i]
            R0 = self.Q_gammas[i+1]
            init_args = (Q0, R0, None)
            
            Q,R,T, _errs = FRLC_LR_opt(C_factors, 
                                       A_factors, 
                                       B_factors, 
                                       a=self.a, 
                                       b=self.b, 
                                       r=r1, 
                                       r2=r2, 
                                       max_iter=self.max_iter, 
                                       device=self.device, 
                                       returnFull=self.returnFull, 
                                       alpha=self.alpha, 
                                       min_iter=self.min_iter, 
                                       initialization=self.initialization, 
                                       tau_out=self.tau_out, 
                                       tau_in=self.tau_in, 
                                       gamma=self.gamma,
                                       dtype=self.dtype, 
                                       updateR = False, 
                                       updateQ = False, 
                                       updateT = True, 
                                       init_args=init_args, 
                                       printCost = self.printCost)
            
            self.T_gammas.append(T)
            
            if self.printCost:
                self.errs['total_cost'].append(float(_errs['total_cost'][-1]))
                self.errs['W_cost'].append(float(_errs['W_cost'][-1]))
                self.errs['GW_cost'].append(float(_errs['GW_cost'][-1]))
        
        return
    
    
    def gamma_smoothing(self,
                        C_factors_sequence,
                        A_factors_sequence,
                        Q_IC = None,
                        R_TC = None):

        """
        Performs the Forward-Backward smoothing procedure (alpha pass + beta pass) 
        followed by a multi-marginal solver to refine the clustering matrices (Q) 
        at each time step.
        
        The smoothed Qs (Q_gammas) are then used to compute the transitions (T_gammas).
        
        Args:
            C_factors_sequence (list): List of cost factor tensors for each transition.
            A_factors_sequence (list): List of distribution factor tensors for each time step.
        
        Returns:
            None. Populates `self.Q_gammas` and `self.T_gammas` with smoothed clusterings
            and transitions, respectively. Also updates `self.errs` with cost values.
        """

         # Clear Q_gammas
        self.Q_gammas = []

        # Run alpha and beta passes
        self.alpha_pass(C_factors_sequence,
                        A_factors_sequence,
                        Q_IC = Q_IC)
        
        self.beta_pass(C_factors_sequence,
                       A_factors_sequence,
                       R_TC = R_TC)
        
        self.Q_gammas.append(self.Q_alphas[0])
        
        # Multi-marginal smoothing for each time step
        for i in range(1, self.N, 1):
            
            C_factors_tm1t, A_factors_tm1t, B_factors_tm1t = C_factors_sequence[i-1], A_factors_sequence[i-1], A_factors_sequence[i]
            C_factors_ttp1, A_factors_ttp1, B_factors_ttp1 = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            
            Q_tm1 = self.Q_alphas[i - 1]
            Q_tp1 = self.Q_betas[self.N - i - 1]
            
            r = self.Q_alphas[i].shape[1]
            
            # Initialize as arguments, fixed during the optimization to infer t-variable
            init_args = (Q_tm1, Q_tp1)
            
            # Learn smoothed clustering Q_t
            Q_t, T_tm1t, T_ttp1 = FRLC_LR_opt_multimarginal(C_factors_tm1t,
                                                            A_factors_tm1t,
                                                            B_factors_tm1t,
                                                            C_factors_ttp1, 
                                                            A_factors_ttp1, 
                                                            B_factors_ttp1, 
                                                            r=r, 
                                                            max_iter=self.max_iter, 
                                                            device=self.device, 
                                                            returnFull=self.returnFull, 
                                                            alpha=self.alpha, 
                                                            min_iter = self.min_iter, 
                                                            initialization=self.initialization, 
                                                            tau_out=self.tau_out, 
                                                            tau_in=self.tau_in, 
                                                            gamma=self.gamma, 
                                                            dtype=self.dtype, 
                                                            init_args=init_args,
                                                            printCost=self.printCost,
                                                            _gQ_t=self.proportions[i])
            self.Q_gammas.append(Q_t)
        
        self.Q_gammas.append(self.Q_betas[0])
        
        # Clearing alpha matrices for space 
        self.Q_alphas = []
        self.impute_smoothed_transitions(C_factors_sequence, A_factors_sequence)
        
        return
    
    def impute_annotated_transitions(self, 
                                     C_factors_sequence, 
                                     A_factors_sequence, 
                                     Qs_annotated):

        """
        Given an externally specified sequence of clusterings (e.g., from annotations), 
        compute the optimal transition matrices (T) that connect consecutive Qs.

        Args:
            C_factors_sequence (list): List of cost factor tensors for each transition.
            A_factors_sequence (list): List of distribution factor tensors for each time step.
            Qs_annotated (list): A list of externally provided cluster matrices Q at each time step.
        
        Returns:
            None. Populates `self.Q_gammas` with `Qs_annotated` and infers `self.T_gammas`
            via `impute_smoothed_transitions`.
        """
        
        # Fix Qs from annotations
        self.Q_gammas = Qs_annotated

        # Impute optimal transition matrices between annotations
        self.impute_smoothed_transitions(C_factors_sequence, A_factors_sequence)
        
        return
        










