import torch
import matplotlib.pyplot as plt
import util_LR
import FRLC_LRDist

class WassersteinDifferentiationMapping:
    
    '''
    Code to compute the full cluster (Q/Lambda) and transition (T) sequence to minimize a joint Wasserstein objective.
    '''
    
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
    
    def __init__(self, rank_list, a=None, b=None, tau_in = 0.0001, tau_out=75, \
                  gamma=90, max_iter=200, min_iter=200, device='cpu', dtype=torch.float64, \
                 printCost=True, returnFull=True, alpha=0.2, beta=0.5, \
                  initialization='Full', init_args = None):
        
        self.rank_list=rank_list
        self.N = len(self.rank_list)
        self.a=a
        self.b=b
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
        self.beta = beta
        self.initialization = initialization
        self.init_args = init_args

    
    def alpha_pass(self, C_factors_sequence, A_factors_sequence):

        self.Q_alphas = []
        self.T_alphas = []

        C_factors, A_factors, B_factors = C_factors_sequence[0], A_factors_sequence[0], A_factors_sequence[1]
        r1, r2 = self.rank_list[0]
        
        Q,R,T, errs = FRLC_LRDist.FRLC_LR_opt(C_factors, A_factors, B_factors, a=self.a, b=self.b, \
                                                  r=r1, r2=r2, max_iter=self.max_iter, device=self.device, \
                                                 returnFull=self.returnFull, alpha=self.alpha, beta=self.beta, \
                                                min_iter=self.min_iter, initialization=self.initialization, \
                                                  tau_out=self.tau_out, tau_in=self.tau_in, gamma=self.gamma, \
                                                dtype=self.dtype, updateR = True, updateQ = True, updateT = True, \
                                              init_args=(None,None,None), printCost=self.printCost)

        self.Q_alphas.append(Q)
        self.Q_alphas.append(R)
        self.T_alphas.append(T)
        
        for i in range(1, self.N-1, 1):

            C_factors, A_factors, B_factors = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            r1, r2 = self.rank_list[i]
            
            Q0 = self.Q_alphas[-1]
            init_args = (Q0, None, None)
            
            Q,R,T, errs = FRLC_LRDist.FRLC_LR_opt(C_factors, A_factors, B_factors, a=self.a, b=self.b, \
                                                          r=r1, r2=r2, max_iter=self.max_iter, device=self.device, \
                                                         returnFull=self.returnFull, alpha=self.alpha, beta=self.beta, \
                                                        min_iter=self.min_iter, initialization=self.initialization, \
                                                          tau_out=self.tau_out, tau_in=self.tau_in, gamma=self.gamma, \
                                                        dtype=self.dtype, updateR = True, updateQ = False, updateT = True, \
                                                  init_args=init_args, printCost=self.printCost)
            self.Q_alphas.append(R)
            self.T_alphas.append(T)
            
        return

    
    def beta_pass(self, C_factors_sequence, A_factors_sequence):

        self.Q_betas = []
        self.T_betas = []
        
        C_factors, A_factors, B_factors = C_factors_sequence[self.N-1], A_factors_sequence[self.N-1], A_factors_sequence[self.N]
        r1, r2 = self.rank_list[self.N-1]
        
        Q,R,T, errs = FRLC_LRDist.FRLC_LR_opt(C_factors, A_factors, B_factors, a=self.a, b=self.b, \
                                                  r=r1, r2=r2, max_iter=self.max_iter, device=self.device, \
                                                 returnFull=self.returnFull, alpha=self.alpha, beta=self.beta, \
                                                min_iter=self.min_iter, initialization=self.initialization, \
                                                  tau_out=self.tau_out, tau_in=self.tau_in, gamma=self.gamma, \
                                                dtype=self.dtype, updateR = True, updateQ = True, updateT = True, \
                                              init_args=(None,None,None), printCost=self.printCost)
        self.Q_betas.append(R)
        self.Q_betas.append(Q)
        self.T_betas.append(T)
        
        for i in range(self.N-2, -1, -1):
            
            C_factors, A_factors, B_factors = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            r1, r2 = self.rank_list[i]
            
            R0 = self.Q_betas[-1]
            
            init_args = (None, R0, None)
            
            Q,R,T, errs = FRLC_LRDist.FRLC_LR_opt(C_factors, A_factors, B_factors, a=self.a, b=self.b, \
                                                      r=r1, r2=r2, max_iter=self.max_iter, device=self.device, \
                                                     returnFull=self.returnFull, alpha=self.alpha, beta=self.beta, \
                                                    min_iter=self.min_iter, initialization=self.initialization, \
                                                      tau_out=self.tau_out, tau_in=self.tau_in, gamma=self.gamma, \
                                                    dtype=self.dtype, updateR = False, updateQ = True, updateT = True, \
                                                  init_args=init_args, printCost=self.printCost)
            
            self.Q_betas.append(Q)
            self.T_betas.append(T)
            
        return

    
    def impute_smoothed_transitions(self, C_factors_sequence, A_factors_sequence):

        self.T_gammas = []
        self.errs = {'total_cost':[], 'W_cost':[], 'GW_cost': []}
        
        for i in range(0, self.N, 1):
            
            r1, r2 = self.rank_list[i]
            C_factors, A_factors, B_factors = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            
            Q0 = self.Q_gammas[i]
            R0 = self.Q_gammas[i+1]
            init_args = (Q0, R0, None)
            
            Q,R,T, _errs = FRLC_LRDist.FRLC_LR_opt(C_factors, A_factors, B_factors, a=self.a, b=self.b, \
                                                      r=r1, r2=r2, max_iter=self.max_iter, device=self.device, \
                                                     returnFull=self.returnFull, alpha=self.alpha, beta=self.beta, \
                                                    min_iter=self.min_iter, initialization=self.initialization, \
                                                      tau_out=self.tau_out, tau_in=self.tau_in, gamma=self.gamma, \
                                                    dtype=self.dtype, updateR = False, updateQ = False, updateT = True, \
                                                   init_args=init_args, printCost = self.printCost)
            
            self.T_gammas.append(T)
            
            if self.printCost:
                self.errs['total_cost'].append(float(_errs['total_cost'][-1]))
                self.errs['W_cost'].append(float(_errs['W_cost'][-1]))
                self.errs['GW_cost'].append(float(_errs['GW_cost'][-1]))
        
        return

    
    def gamma_smoothing(self, C_factors_sequence, A_factors_sequence):

        self.Q_gammas = []
        
        self.alpha_pass(C_factors_sequence, A_factors_sequence)
        self.beta_pass(C_factors_sequence, A_factors_sequence)

        self.Q_gammas.append(self.Q_alphas[0])
        
        for i in range(1, self.N, 1):
            
            C_factors_tm1t, A_factors_tm1t, B_factors_tm1t = C_factors_sequence[i-1], A_factors_sequence[i-1], A_factors_sequence[i]
            C_factors_ttp1, A_factors_ttp1, B_factors_ttp1 = C_factors_sequence[i], A_factors_sequence[i], A_factors_sequence[i+1]
            
            Q_tm1 = self.Q_alphas[i - 1]
            Q_tp1 = self.Q_betas[self.N - i - 1]
            
            r = self.Q_alphas[i].shape[1]
            
            # Initialize as arguments
            init_args = (Q_tm1, Q_tp1)
            
            # Learn smoothed clustering Q_t
            Q_t, T_tm1t, T_ttp1 = FRLC_LRDist.FRLC_LR_opt_multimarginal(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, \
                            C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, r=r, max_iter=self.max_iter, device=self.device, \
                                                         returnFull=self.returnFull, alpha=self.alpha, beta=self.beta, \
                                                        min_iter = self.min_iter, initialization=self.initialization, tau_out=self.tau_out, \
                                                        tau_in=self.tau_in, gamma=self.gamma, dtype=self.dtype, init_args=init_args, printCost=self.printCost)
            self.Q_gammas.append(Q_t)
        
        self.Q_gammas.append(self.Q_betas[0])
        # Clearing alpha matrices for space 
        self.Q_alphas = []
        self.impute_smoothed_transitions(C_factors_sequence, A_factors_sequence)
        
        return

    
    def impute_annotated_transitions(self, C_factors_sequence, A_factors_sequence, Qs_annotated):
        # Fix Qs from annotations
        self.Q_gammas = Qs_annotated
        # Impute optimal transition matrices between annotations
        self.impute_smoothed_transitions(C_factors_sequence, A_factors_sequence)
        
        return
        


