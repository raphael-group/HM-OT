import torch
import matplotlib.pyplot as plt
import util
import objective_grad as gd







def FRLC_LR_opt_multimarginal(C_factors_tm1t,
                              A_factors_tm1t, 
                              B_factors_tm1t,
                              C_factors_ttp1, 
                              A_factors_ttp1, 
                              B_factors_ttp1, 
                              tau_in = 0.0001, 
                              tau_out=75, 
                              gamma=90, 
                              r = 10, 
                              max_iter=200, 
                              device='cpu', 
                              dtype=torch.float64,
                              printCost=True,
                              returnFull=True,
                              alpha=0.2,
                              initialization='Full',
                              init_args = None,
                              full_grad=True,
                              convergence_criterion=True,
                              tol=5e-6,
                              min_iter = 25,
                              max_inneriters_balanced= 300,
                              max_inneriters_relaxed=50,
                              _gQ_t=None):
    
   
    N1, N2, N3 = C_factors_tm1t[0].size(dim=0), C_factors_tm1t[1].size(dim=1), C_factors_ttp1[1].size(dim=1)
    
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    one_N3 = torch.ones((N3), device=device, dtype=dtype)
    
    # Initialize tm1, tp1 couplings to input factors and yield associated inner marginals
    Q_tm1, Q_tp1 = init_args
    gQ_tm1 = Q_tm1.T @ one_N1
    gQ_tp1 =  Q_tp1.T @ one_N3
    
    r1 = Q_tm1.shape[1]
    r2 = r
    r3 = Q_tp1.shape[1]

    print(f'Ranks: r1 {r1}, r2 {r2}, r3 {r3}')
    
    k = 0
    stationarity_gap = torch.inf

    # Assume uniform marginals over spots
    a = one_N1 / N1
    b = one_N2 / N2
    c = one_N3 / N3
    
    one_r1 = torch.ones((r1), device=device, dtype=dtype)
    one_r2 = torch.ones((r2), device=device, dtype=dtype)
    one_r3 = torch.ones((r3), device=device, dtype=dtype)
    
    # Initialize inner marginals; generalized to be of differing dimensions to account for non-square latent-coupling.
    gQ_tm1 = Q_tm1.T @ one_N1
    gQ_tp1 = Q_tp1.T @ one_N3
    
    if _gQ_t == None:
        # Middle inner marginal initialized to uniform for simplicity
        gQ_t = (1/r2)*one_r2
    else:
        # Otherwise, fix cluster proportions
        gQ_t = _gQ_t
    
    # Variables to optimize: Q_t, T_tm1t, T_ttp1; take random matrix sample and project onto coupling space for each
    Q_t = util.logSinkhorn(torch.rand((N2,r2), device=device, dtype=dtype), b, gQ_t, gamma, \
                    max_iter = max_inneriters_balanced, device=device, dtype=dtype, balanced=True, unbalanced=False)
    
    T_tm1t = util.logSinkhorn(torch.rand((r1,r2), device=device, dtype=dtype), gQ_tm1, gQ_t, gamma, \
                    max_iter = max_inneriters_balanced, device=device, dtype=dtype, balanced=True, unbalanced=False)
    T_ttp1 = util.logSinkhorn(torch.rand((r2,r3), device=device, dtype=dtype), gQ_t, gQ_tp1, gamma, \
                    max_iter = max_inneriters_balanced, device=device, dtype=dtype, balanced=True, unbalanced=False)
    
    Lambda_tm1t = torch.diag(1/ (Q_tm1.T @ one_N1)) @ T_tm1t @ torch.diag(1/ (Q_t.T @ one_N2))
    Lambda_ttp1 = torch.diag(1/ (Q_t.T @ one_N2)) @ T_ttp1 @ torch.diag(1/ (Q_tp1.T @ one_N3))
    
    '''
    Preparing main loop.
    '''
    errs = {'total_cost':[], 'W_cost':[], 'triplet_cost':[], 'GW_cost': []}
    grad = torch.inf
    gamma_k = gamma
    Q_t_prev, T_tm1t_prev, T_ttp1_prev = None, None, None
    
    while (k < max_iter and (not convergence_criterion or \
                       (k < min_iter or Delta((Q_t, T_tm1t, T_ttp1), (Q_t_prev, T_tm1t_prev, T_ttp1_prev), gamma_k) > tol))):
        
        if convergence_criterion:
            # Set previous iterates to evaluate convergence at the next round
            Q_t_prev, T_tm1t_prev, T_ttp1_prev = Q_t, T_tm1t, T_ttp1
        
        if k % 25 == 0:
            print(f'Iteration: {k}')
        
        gradQ_t, gamma_k = compute_grad_A_multimarginal(C_factors_tm1t,
                                                        A_factors_tm1t,
                                                        B_factors_tm1t,
                                                        Q_tm1,
                                                        Q_t,
                                                        Lambda_tm1t,
                                                        C_factors_ttp1,
                                                        A_factors_ttp1,
                                                        B_factors_ttp1,
                                                        Q_tp1,
                                                        Lambda_ttp1,
                                                        gamma,
                                                        device=device,
                                                        alpha=alpha,
                                                        dtype=dtype,
                                                        full_grad=full_grad)
        
        Q_t = util.logSinkhorn(gradQ_t - (gamma_k**-1)*torch.log(Q_t), b, gQ_t, gamma_k, max_iter = max_inneriters_relaxed, \
                             device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)

        """
        Update gQ_t cluster proportions, unless gQ_t fixed as a model parameter.
        """
        if _gQ_t == None:
            gQ_t = Q_t.T @ one_N2
        
        """
        T_tm1t first transition matrix
        """
        gradT_tm1t, gamma_T = gd.compute_grad_B_LR(C_factors_tm1t,
                                                   A_factors_tm1t,
                                                   B_factors_tm1t,
                                                   Q_tm1,
                                                   Q_t,
                                                   Lambda_tm1t,
                                                   gQ_tm1,
                                                   gQ_t,
                                                   gamma,
                                                   device,
                                                   alpha=alpha,
                                                   dtype=dtype)
        
        T_tm1t = util.logSinkhorn(gradT_tm1t - (gamma_T**-1)*torch.log(T_tm1t), gQ_tm1, gQ_t, gamma_T, max_iter = max_inneriters_balanced, \
                             device=device, dtype=dtype, balanced=True, unbalanced=False)

        
        """
        T_ttp1 second transition matrix
        """
        gradT_ttp1, gamma_T = gd.compute_grad_B_LR(C_factors_ttp1,
                                                   A_factors_ttp1,
                                                   B_factors_ttp1,
                                                   Q_t,
                                                   Q_tp1,
                                                   Lambda_ttp1,
                                                   gQ_t,
                                                   gQ_tp1,
                                                   gamma,
                                                   device,
                                                   alpha=alpha,
                                                   dtype=dtype)
        
        T_ttp1 = util.logSinkhorn(gradT_ttp1 - (gamma_T**-1)*torch.log(T_ttp1), gQ_t, gQ_tp1, gamma_T, max_iter = max_inneriters_balanced, \
                             device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # Inner latent transition-inverse matrix
        Lambda_tm1t = torch.diag(1/ gQ_tm1) @ T_tm1t @ torch.diag(1/ gQ_t)
        Lambda_ttp1 = torch.diag(1/ gQ_t) @ T_ttp1 @ torch.diag(1/ gQ_tp1)
        
        k+=1
        
        if printCost:
            primal_cost1 = torch.trace(((Q_tm1.T @ C_factors_tm1t[0]) @ (C_factors_tm1t[1] @ Q_t)) @ Lambda_tm1t.T)
            primal_cost2 = torch.trace(((Q_t.T @ C_factors_ttp1[0]) @ (C_factors_ttp1[1] @ Q_tp1)) @ Lambda_ttp1.T)
            errs['W_cost'].append(primal_cost1.cpu() + primal_cost2.cpu())

    if printCost:
        print(f"Initial Wasserstein-sum cost: {errs['W_cost'][0]}")
        print(f"Final Wasserstein-sum cost: {errs['W_cost'][-1]}")
        plt.title('Wasserstein-sum cost (excludes GW component)')
        plt.plot(errs['W_cost'])
        plt.show()
    
    return Q_t, T_tm1t, T_ttp1


def compute_grad_A_multimarginal(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, Q_tm1, Q_t, Lambda_tm1t,
                                 C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, Q_tp1, Lambda_ttp1, \
                                 gamma, device, alpha=0.2, dtype=torch.float64, full_grad=False):
    
    gradQ_tm1t, gradR_tm1t = gd.compute_grad_A_LR(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, \
                                                      Q_tm1, Q_t, Lambda_tm1t, gamma, device=device, \
                                                   alpha=alpha, dtype=dtype, full_grad=full_grad, normalize=False)
    
    gradQ_ttp1, gradR_ttp1 = gd.compute_grad_A_LR(C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, \
                                                      Q_t, Q_tp1, Lambda_ttp1, gamma, device=device, \
                                                   alpha=alpha, dtype=dtype, full_grad=full_grad, normalize=False)
    
    gradQ_t = gradQ_ttp1 + gradR_tm1t
    
    # Handle by normalizing sum of gradients in multi-marginal case
    normalizer = torch.max(torch.abs(gradQ_t))
    gamma_k = gamma / normalizer
    
    return gradQ_t, gamma_k

