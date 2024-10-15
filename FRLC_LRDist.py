import torch
import matplotlib.pyplot as plt
import util_LR

def compute_grad_A_FRLC_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gamma, device, \
                   alpha=0.2, beta=0.5, dtype=torch.float64, full_grad=False):
    
    r = Lambda.shape[0]
    one_r = torch.ones((r), device=device, dtype=dtype)
    One_rr = torch.outer(one_r, one_r).to(device)
    N1, N2 = C_factors[0].size(0), C_factors[1].size(1)

    A1, A2 = A_factors[0], A_factors[1]
    B1, B2 = B_factors[0], B_factors[1]
    
    # A*2's low-rank factorization
    A1_tild, A2_tild = util_LR.hadamard_square_lr(A1, A2.T, device=device)
    
    # GW gradients for semi-relaxed left marginal case (R1_r = b)
    gradQ = - 4 * (A1 @ (A2 @ (Q @ Lambda@( (R.T@ B1) @ (B2 @R) )@Lambda.T)) ) + (2* A1_tild @ (A2_tild.T @ Q @ One_rr))
    gradR = - 4 * (B1 @ (B2 @ (R @ (Lambda.T@( (Q.T @ A1) @ ( A2 @ Q ))@Lambda)) ) )

    one_N1, one_N2 = torch.ones((N1), device=device, dtype=dtype), torch.ones((N2), device=device, dtype=dtype)
    if full_grad:
        # Rank-1 GW perturbation
        N1, N2 = Q.shape[0], R.shape[0]
        gQ, gR = Q.T @ one_N1, R.T @ one_N2
        
        MR = Lambda.T @ ( (Q.T @ A1) @ (A2 @ Q) ) @ Lambda @ ((R.T @ B1) @ (B2 @ R)) @ torch.diag(1/gR)
        MQ = Lambda @ ( (R.T @ B1) @ (B2 @ R) ) @ Lambda.T @ ((Q.T @ A1) @ (A2 @ Q) ) @ torch.diag(1/gQ)
        gradQ += 4*torch.outer(one_N1, torch.diag(MQ))
        gradR += 4*torch.outer(one_N2, torch.diag(MR))
    
    # Triplet gradients
    gQ, gR = Q.T @ one_N1, R.T @ one_N2
    gradQ_triplet = 2 * A1 @ (A2 @ (Q @ torch.diag(1/gQ**2)))
    gradR_triplet = 2 * B1 @ (B2 @ (R @ torch.diag(1/gR**2)))
    '''
    gradQ_triplet = 2 * ((Q @ (Lambda @ ((R.T @ B1) @ (B2 @ R) ) @ Lambda.T)) +\
                                         (( A1 @ (A2 @ Q) ) @ (Lambda @ (R.T @ R) @ Lambda.T)) )
    gradR_triplet = 2 * (( ( B2.T @ (B1.T @ R) ) @ (Lambda.T @ (Q.T @ Q) @ Lambda) ) +\
                                 ( R @ (Lambda.T @ ( (Q.T @ A2.T) @ (A1.T @ Q) ) @ Lambda) ))
    '''
    '''
    if full_grad:
        # Rank-1 triplet perturbation
        N1, N2 = Q.shape[0], R.shape[0]
        one_N1, one_N2 = torch.ones((N1), device=device, dtype=dtype), torch.ones((N2), device=device, dtype=dtype)
        gQ, gR = Q.T @ one_N1, R.T @ one_N2
        MQ = ( Lambda @ ((R.T @ B1) @ (B2 @ R)) @ Lambda.T @ (Q.T @ Q) @ torch.diag(1/gQ) + \
                                        Lambda @ (R.T @ R) @ Lambda.T @ ((Q.T @ A1) @ (A2 @ Q)) @ torch.diag(1/gQ) )
        MR = ( ((R.T @ B1) @ (B2 @ R)) @ Lambda.T @ (Q.T @ Q) @ Lambda @ torch.diag(1/gR) + \
                                      (R.T @ R) @ Lambda.T @ ((Q.T @ A1) @ (A2 @ Q)) @ Lambda @ torch.diag(1/gR))
        gradQ_triplet += -2 * torch.outer(one_N1, torch.diag(MQ))
        gradR_triplet += -2 * torch.outer(one_N2, torch.diag(MR))
    '''
    
    # merged gradients
    gradQ = (1-beta)*gradQ + beta*gradQ_triplet
    gradR = (1-beta)*gradR + beta*gradR_triplet
    
    # total gradients -- readjust cost for FGW problem by adding W gradients
    gradQW, gradRW = Wasserstein_Grad_FRLC_LR(C_factors, Q, R, Lambda, device, \
                                                   dtype=dtype, full_grad=full_grad)
    gradQ = (1-alpha)*gradQW + (alpha/2)*gradQ
    gradR = (1-alpha)*gradRW + (alpha/2)*gradR
    
    normalizer = torch.max(torch.tensor([torch.max(torch.abs(gradQ)) , torch.max(torch.abs(gradR))]))
    gamma_k = gamma / normalizer
    
    return gradQ, gradR, gamma_k

def compute_grad_B_FRLC_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gQ, gR, gamma, device, \
                   alpha=0.2, beta = 0.5, dtype=torch.float64):
    
    N1, N2 = C_factors[0].size(0), C_factors[1].size(1)
    
    A1, A2 = A_factors[0], A_factors[1]
    B1, B2 = B_factors[0], B_factors[1]
    # GW grad
    gradLambda = -4 * ( (Q.T @ A1) @ (A2 @ Q) ) @ Lambda @ ( (R.T @ B1) @ (B2 @ R) )
    
    # triplet grad
    '''
    gradLambda_triplet = 2*( (( (Q.T @ A1) @ (A2 @ Q) ) @ (Lambda @ (R.T @ R))) \
                            + ((Q.T @ Q) @ Lambda @ ((R.T @ B1) @ (B2 @ R)) ) )'''
    # merged grad
    #gradLambda = gradLambda_triplet
    gradLambda = (1-beta)*gradLambda #+ beta*gradLambda_triplet
    
    del A1,A2,B1,B2
    
    C1, C2 = C_factors[0], C_factors[1]
    # total grad
    gradLambda = (1-alpha)*( (Q.T @ C1) @ (C2 @ R) ) + (alpha/2)*gradLambda
    
    gradT = torch.diag(1/gQ) @ gradLambda @ torch.diag(1/gR) # (mass-reweighted form)
    gamma_T = gamma / torch.max(torch.abs(gradT))
    return gradT, gamma_T

def Wasserstein_Grad_FRLC_LR(C_factors, Q, R, Lambda, device, \
                   dtype=torch.float64, full_grad=True):

    C1, C2 = C_factors[0], C_factors[1]
    
    gradQ = C1 @ ((C2 @ R) @ Lambda.T)
    
    if full_grad:
        # rank-one perturbation
        N1 = Q.shape[0]
        one_N1 = torch.ones((N1), device=device, dtype=dtype)
        gQ = Q.T @ one_N1
        w1 = torch.diag( (gradQ.T @ Q) @ torch.diag(1/gQ) )
        gradQ -= torch.outer(one_N1, w1)
    
    # linear term
    gradR = C2.T @ ((C1.T @ Q) @ Lambda)
    if full_grad:
        # rank-one perturbation
        N2 = R.shape[0]
        one_N2 = torch.ones((N2), device=device, dtype=dtype)
        gR = R.T @ one_N2
        w2 = torch.diag( torch.diag(1/gR) @ (R.T @ gradR) )
        gradR -= torch.outer(one_N2, w2)
    
    return gradQ, gradR

def FRLC_LR_opt(C_factors, A_factors, B_factors, a=None, b=None, tau_in = 0.0001, tau_out=75, \
                  gamma=90, r = 10, r2=None, max_iter=200, device='cpu', dtype=torch.float64, \
                 printCost=True, returnFull=True, alpha=0.2, beta=0.5, \
                  initialization='Full', init_args = None, full_grad=True, \
                   convergence_criterion=True, tol=5e-6, min_iter = 25, \
                   max_inneriters_balanced= 300, max_inneriters_relaxed=50, \
                  diagonalize_return=False, updateQ=True, updateR=True, updateT=True):
    
    N1, N2 = C_factors[0].size(dim=0), C_factors[1].size(dim=1)
    k = 0
    stationarity_gap = torch.inf
    
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    
    if a is None:
        a = one_N1 / N1
    if b is None:
        b = one_N2 / N2
    if r2 is None:
        r2 = r

    one_r = torch.ones((r), device=device, dtype=dtype)
    one_r2 = torch.ones((r2), device=device, dtype=dtype)
    
    # Initialize inner marginals to uniform; 
    # generalized to be of differing dimensions to account for non-square latent-coupling.
    gQ = (1/r)*one_r
    gR = (1/r2)*one_r2
    
    full_rank = True if initialization == 'Full' else False
    
    if initialization == 'Full':
        full_rank = True
    elif initialization == 'Rank-2':
        full_rank = False
    else:
        full_rank = True
        print('Initialization must be either "Full" or "Rank-2", defaulting to "Full".')
        
    if init_args is None:
        Q, R, T, Lambda = initialize_couplings(a, b, gQ, gR, \
                                                    gamma, full_rank=full_rank, \
                                                device=device, dtype=dtype, \
                                                    max_iter = max_inneriters_balanced)
    else:
        # Initialize to given factors
        Q, R, T = init_args
        if Q is not None:
            gQ = Q.T @ one_N1
        if R is not None:
            gR =  R.T @ one_N2
        if Q is None or R is None or T is None:
            _Q, _R, _T, Lambda = initialize_couplings(a, b, gQ, gR, \
                                                    gamma, full_rank=full_rank, \
                                                device=device, dtype=dtype, \
                                                    max_iter = max_inneriters_balanced)
            if Q is None:
                Q = _Q
            if R is None:
                R = _R
            if T is None:
                T = _T
        
        Lambda = torch.diag(1/ (Q.T @ one_N1)) @ T @ torch.diag(1/ (R.T @ one_N2))
    
    '''
    Preparing main loop.
    '''
    errs = {'total_cost':[], 'W_cost':[], 'GW_cost': []}
    grad = torch.inf
    gamma_k = gamma
    Q_prev, R_prev, T_prev = None, None, None
    
    while (k < max_iter and (not convergence_criterion or \
                       (k < min_iter or Delta((Q, R, T), (Q_prev, R_prev, T_prev), gamma_k) > tol))):
        
        if convergence_criterion:
            # Set previous iterates to evaluate convergence at the next round
            Q_prev, R_prev, T_prev = Q, R, T
        
        if k % 25 == 0:
            print(f'Iteration: {k}')
        
        gradQ, gradR, gamma_k = compute_grad_A_FRLC_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gamma, device, \
                                   alpha=alpha, beta=beta, dtype=dtype, full_grad=full_grad)
        ### Semi-relaxed updates ###
        if updateR:
            R = logSinkhorn(gradR - (gamma_k**-1)*torch.log(R), b, gR, gamma_k, max_iter = max_inneriters_relaxed, \
                             device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
        if updateQ:
            Q = logSinkhorn(gradQ - (gamma_k**-1)*torch.log(Q), a, gQ, gamma_k, max_iter = max_inneriters_relaxed, \
                             device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)
        
        gQ, gR = Q.T @ one_N1, R.T @ one_N2
        
        gradT, gamma_T = compute_grad_B_FRLC_LR(C_factors, A_factors, B_factors, Q, R, Lambda, gQ, gR, gamma, device, \
                                       alpha=alpha, beta = beta, dtype=dtype)
        if updateT:
            T = logSinkhorn(gradT - (gamma_T**-1)*torch.log(T), gQ, gR, gamma_T, max_iter = max_inneriters_balanced, \
                             device=device, dtype=dtype, balanced=True, unbalanced=False)
        '''
        else:
            # Semi-relaxed on transition matrix (e.g. if cell-type given in advance, can "learn" growth of each)
            TmT = logSinkhorn((gradT - (gamma_T**-1)*torch.log(T)).mT, gR, gQ, gamma_T, max_iter = max_inneriters_relaxed, \
                             device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)
            T = TmT.mT'''
        
        # Inner latent transition-inverse matrix
        Lambda = torch.diag(1/gQ) @ T @ torch.diag(1/gR)
        k+=1
        
        if printCost:
            primal_cost = torch.trace(((Q.T @ C_factors[0]) @ (C_factors[1] @ R)) @ Lambda.T)
            '''
            triplet_cost = torch.trace(R@((Lambda.T @ ((Q.T @ A_factors[0]) @ (A_factors[1] @ Q)) @ Lambda)@R.T)) + \
                                torch.trace( Q @ (((Lambda @ (R.T @ B_factors[0]) @ (B_factors[1] @ R)) @ Lambda.T) @ Q.T) )'''
            X = R @ ((Lambda.T @ ((Q.T @ A_factors[0]) @ (A_factors[1] @ Q)) @ Lambda) @ (R.T @ B_factors[0])) @ B_factors[1]
            GW_cost = - 2 * torch.trace(X) # add these: one_r.T @ M1 @ one_r + one_r.T @ M2 @ one_r
            del X
            A1_tild, A2_tild = util_LR.hadamard_square_lr(A_factors[0], A_factors[1].T, device=device)
            GW_cost += torch.inner(A1_tild.T @ (Q @ one_r), A2_tild.T @ (Q @ one_r))
            del A1_tild, A2_tild
            B1_tild, B2_tild = util_LR.hadamard_square_lr(B_factors[0], B_factors[1].T, device=device)
            GW_cost += torch.inner(B1_tild.T @ (R @ one_r2), B2_tild.T @ (R @ one_r2))
            del B1_tild, B2_tild
            
            errs['W_cost'].append(primal_cost.cpu())
            #errs['triplet_cost'].append((triplet_cost).cpu())
            errs['GW_cost'].append((GW_cost).cpu())
            errs['total_cost'].append(((1-alpha)*primal_cost + alpha*GW_cost).cpu()) #beta*triplet_cost + (1-beta)*GW_cost
        
    if diagonalize_return:
        '''
        Diagonalize return to factorization of (Forrow 2019)
        '''
        Q = Q @ torch.diag(1 / gQ) @ T
        gR = R.T @ one_N2
        T = torch.diag(gR)

    if printCost:
        print(f"Initial Wasserstein cost: {errs['W_cost'][0]}, GW-cost: {errs['GW_cost'][0]}, Total cost: {errs['total_cost'][0]}")
        print(f"Final Wasserstein cost: {errs['W_cost'][-1]}, GW-cost: {errs['GW_cost'][-1]}, Total cost: {errs['total_cost'][-1]}")
        plt.plot(errs['total_cost'])
        plt.show()
    
    if returnFull:
        P = Q @ Lambda @ R.T
        return P, errs
    else:
        return Q, R, T, errs


def compute_grad_A_helper_no_norm(C_factors, A_factors, B_factors, Q, R, Lambda, device, \
                   alpha=0.2, beta=0.5, dtype=torch.float64, full_grad=False):
    
    r = Lambda.shape[0]
    one_r = torch.ones((r), device=device, dtype=dtype)
    One_rr = torch.outer(one_r, one_r).to(device)
    N1, N2 = C_factors[0].size(0), C_factors[1].size(1)

    A1, A2 = A_factors[0], A_factors[1]
    B1, B2 = B_factors[0], B_factors[1]
    
    # A*2's low-rank factorization
    A1_tild, A2_tild = util_LR.hadamard_square_lr(A1, A2.T, device=device)
    
    # GW gradients for semi-relaxed left marginal case (R1_r = b)
    gradQ = - 4 * (A1 @ (A2 @ (Q @ Lambda@( (R.T@ B1) @ (B2 @R) )@Lambda.T)) ) + (2* A1_tild @ (A2_tild.T @ Q @ One_rr))
    gradR = - 4 * (B1 @ (B2 @ (R @ (Lambda.T@( (Q.T @ A1) @ ( A2 @ Q ))@Lambda)) ) )

    one_N1, one_N2 = torch.ones((N1), device=device, dtype=dtype), torch.ones((N2), device=device, dtype=dtype)
    if full_grad:
        # Rank-1 GW perturbation
        N1, N2 = Q.shape[0], R.shape[0]
        gQ, gR = Q.T @ one_N1, R.T @ one_N2
        
        MR = Lambda.T @ ( (Q.T @ A1) @ (A2 @ Q) ) @ Lambda @ ((R.T @ B1) @ (B2 @ R)) @ torch.diag(1/gR)
        MQ = Lambda @ ( (R.T @ B1) @ (B2 @ R) ) @ Lambda.T @ ((Q.T @ A1) @ (A2 @ Q) ) @ torch.diag(1/gQ)
        gradQ += 4*torch.outer(one_N1, torch.diag(MQ))
        gradR += 4*torch.outer(one_N2, torch.diag(MR))
    
    # Triplet gradients
    gQ, gR = Q.T @ one_N1, R.T @ one_N2
    gradQ_triplet = 2 * A1 @ (A2 @ (Q @ torch.diag(1/gQ**2)))
    gradR_triplet = 2 * B1 @ (B2 @ (R @ torch.diag(1/gR**2)))
    
    # merged gradients
    gradQ = (1-beta)*gradQ + beta*gradQ_triplet
    gradR = (1-beta)*gradR + beta*gradR_triplet
    
    # total gradients -- readjust cost for FGW problem by adding W gradients
    gradQW, gradRW = Wasserstein_Grad_FRLC_LR(C_factors, Q, R, Lambda, device, \
                                                   dtype=dtype, full_grad=full_grad)
    gradQ = (1-alpha)*gradQW + (alpha/2)*gradQ
    gradR = (1-alpha)*gradRW + (alpha/2)*gradR

    return gradQ, gradR

def compute_grad_A_multimarginal(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, Q_tm1, Q_t, Lambda_tm1t,
                                 C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, Q_tp1, Lambda_ttp1, \
                                 gamma, device, \
                   alpha=0.2, beta=0.5, dtype=torch.float64, full_grad=False):

    gradQ_tm1t, gradR_tm1t = compute_grad_A_helper_no_norm(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, \
                                                      Q_tm1, Q_t, Lambda_tm1t, device=device, \
                                                   alpha=alpha, beta=beta, dtype=dtype, full_grad=full_grad)
        
    gradQ_ttp1, gradR_ttp1 = compute_grad_A_helper_no_norm(C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, \
                                                      Q_t, Q_tp1, Lambda_ttp1, device=device, \
                                                   alpha=alpha, beta=beta, dtype=dtype, full_grad=full_grad)
    #print(gradQ_ttp1.sum())
    #print(gradR_tm1t.sum())
    gradQ_t = gradQ_ttp1 + gradR_tm1t
    
    normalizer = torch.max(torch.abs(gradQ_t))
    gamma_k = gamma / normalizer
    
    return gradQ_t, gamma_k

def FRLC_LR_opt_multimarginal(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, \
                    C_factors_ttp1, A_factors_ttp1, B_factors_ttp1,                                            \
                    a=None, b=None, tau_in = 0.0001, tau_out=75, \
                  gamma=90, r = 10, max_iter=200, device='cpu', dtype=torch.float64, \
                 printCost=True, returnFull=True, alpha=0.2, beta=0.5, \
                  initialization='Full', init_args = None, full_grad=True, \
                   convergence_criterion=True, tol=5e-6, min_iter = 25, \
                   max_inneriters_balanced= 300, max_inneriters_relaxed=50, initQ_t=None):
   
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
    
    if initQ_t == None:
        # Middle inner marginal initialized to uniform for simplicity
        gQ_t = (1/r2)*one_r2
        # Variables to optimize: Q_t, T_tm1t, T_ttp1; take random matrix sample and project onto coupling space for each
        Q_t = logSinkhorn(torch.rand((N2,r2), device=device, dtype=dtype), b, gQ_t, gamma, \
                        max_iter = max_inneriters_balanced, device=device, dtype=dtype, balanced=True, unbalanced=False)
    else:
        Q_t = initQ_t
        gQ_t = Q_t.T @ one_N2
        
    T_tm1t = logSinkhorn(torch.rand((r1,r2), device=device, dtype=dtype), gQ_tm1, gQ_t, gamma, \
                    max_iter = max_inneriters_balanced, device=device, dtype=dtype, balanced=True, unbalanced=False)
    T_ttp1 = logSinkhorn(torch.rand((r2,r3), device=device, dtype=dtype), gQ_t, gQ_tp1, gamma, \
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
        gradQ_t, gamma_k = compute_grad_A_multimarginal(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, Q_tm1, Q_t, Lambda_tm1t, \
                                 C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, Q_tp1, Lambda_ttp1, \
                                    gamma, device=device, alpha=alpha, beta=beta, dtype=dtype, full_grad=full_grad)
        
        ### Update: Q_t inner clustering ###
        '''
        Q_t = logSinkhorn(gradQ_t - (gamma_k**-1)*torch.log(Q_t), b, gQ_t, gamma_k, max_iter = max_inneriters_relaxed, \
                             device=device, dtype=dtype, balanced=False, unbalanced=False, tau=tau_in)'''
        
        Q_t = logSinkhorn(gradQ_t - (gamma_k**-1)*torch.log(Q_t), b, gQ_t, gamma_k, max_iter = max_inneriters_relaxed, \
                             device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)
        
        gQ_t = Q_t.T @ one_N2
        
        ### Update: T_tm1t first transition matrix ###
        gradT_tm1t, gamma_T = compute_grad_B_FRLC_LR(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, Q_tm1, Q_t, Lambda_tm1t, gQ_tm1, gQ_t, gamma, device, \
                                       alpha=alpha, beta = beta, dtype=dtype)
        
        T_tm1t = logSinkhorn(gradT_tm1t - (gamma_T**-1)*torch.log(T_tm1t), gQ_tm1, gQ_t, gamma_T, max_iter = max_inneriters_balanced, \
                             device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        ### Update: T_ttp1 second transition matrix ###
        gradT_ttp1, gamma_T = compute_grad_B_FRLC_LR(C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, Q_t, Q_tp1, Lambda_ttp1, gQ_t, gQ_tp1, gamma, device, \
                                       alpha=alpha, beta = beta, dtype=dtype)
        T_ttp1 = logSinkhorn(gradT_ttp1 - (gamma_T**-1)*torch.log(T_ttp1), gQ_t, gQ_tp1, gamma_T, max_iter = max_inneriters_balanced, \
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

def FRLC_LR_opt_multimarginal_2(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, \
                    C_factors_ttp1, A_factors_ttp1, B_factors_ttp1,                                            \
                    a=None, b=None, tau_in = 0.0001, tau_out=75, \
                  gamma=90, r = 10, max_iter=200, device='cpu', dtype=torch.float64, \
                 printCost=True, returnFull=True, alpha=0.2, beta=0.5, \
                  initialization='Full', init_args = None, full_grad=True, \
                   convergence_criterion=True, tol=5e-6, min_iter = 25, \
                   max_inneriters_balanced= 300, max_inneriters_relaxed=50):
   
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
    
    # Middle inner marginal initialized to uniform for simplicity
    gQ_t = (1/r2)*one_r2

    # Variables to optimize: Q_t, T_tm1t, T_ttp1; take random matrix sample and project onto coupling space for each
    Q_t = logSinkhorn(torch.rand((N2,r2), device=device, dtype=dtype), b, gQ_t, gamma, \
                    max_iter = max_inneriters_balanced, device=device, dtype=dtype, balanced=True, unbalanced=False)
    T_tm1t = logSinkhorn(torch.rand((r1,r2), device=device, dtype=dtype), gQ_tm1, gQ_t, gamma, \
                    max_iter = max_inneriters_balanced, device=device, dtype=dtype, balanced=True, unbalanced=False)
    T_ttp1 = logSinkhorn(torch.rand((r2,r3), device=device, dtype=dtype), gQ_t, gQ_tp1, gamma, \
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
        
        gradQ_tm1t, gradR_tm1t, gamma_k = compute_grad_A_FRLC_LR(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, Q_tm1, Q_t, Lambda_tm1t, gamma, device, \
                                   alpha=alpha, beta=beta, dtype=dtype, full_grad=full_grad)
        
        gradQ_ttp1, gradR_ttp1, gamma_k = compute_grad_A_FRLC_LR(C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, Q_t, Q_tp1, Lambda_ttp1, gamma, device, \
                                   alpha=alpha, beta=beta, dtype=dtype, full_grad=full_grad)
        
        gradQ_t = gradQ_ttp1 + gradR_tm1t
        
        ### Update: Q_t inner clustering ###
        Q_t = logSinkhorn(gradQ_t - (gamma_k**-1)*torch.log(Q_t), b, gQ_t, gamma_k, max_iter = max_inneriters_relaxed, \
                             device=device, dtype=dtype, balanced=False, unbalanced=True, tau=tau_out, tau2=tau_in)
        gQ_t = Q_t.T @ one_N2
        
        ### Update: T_tm1t first transition matrix ###
        gradT_tm1t, gamma_T = compute_grad_B_FRLC_LR(C_factors_tm1t, A_factors_tm1t, B_factors_tm1t, Q_tm1, Q_t, Lambda_tm1t, gQ_tm1, gQ_t, gamma, device, \
                                       alpha=alpha, beta = beta, dtype=dtype)
        T_tm1t = logSinkhorn(gradT_tm1t - (gamma_T**-1)*torch.log(T_tm1t), gQ_tm1, gQ_t, gamma_T, max_iter = max_inneriters_balanced, \
                             device=device, dtype=dtype, balanced=True, unbalanced=False)

        ### Update: T_ttp1 second transition matrix ###
        gradT_ttp1, gamma_T = compute_grad_B_FRLC_LR(C_factors_ttp1, A_factors_ttp1, B_factors_ttp1, Q_t, Q_tp1, Lambda_ttp1, gQ_t, gQ_tp1, gamma, device, \
                                       alpha=alpha, beta = beta, dtype=dtype)
        T_ttp1 = logSinkhorn(gradT_ttp1 - (gamma_T**-1)*torch.log(T_ttp1), gQ_t, gQ_tp1, gamma_T, max_iter = max_inneriters_balanced, \
                             device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # Inner latent transition-inverse matrix
        Lambda_tm1t = torch.diag(1/ gQ_tm1) @ T_tm1t @ torch.diag(1/ gQ_t)
        Lambda_ttp1 = torch.diag(1/ gQ_t) @ T_ttp1 @ torch.diag(1/ gQ_tp1)
        
        k+=1
    
    return Q_t, T_tm1t, T_ttp1

def initialize_couplings(a, b, gQ, gR, gamma, \
                         full_rank=True, device='cpu', \
                         dtype=torch.float64, rank2_random=False, \
                        max_iter=50):
    '''
    ------Parameters------
    a: torch tensor
        Left outer marginal, should be positive and sum to 1.0
    b: torch tensor
        Right outer marginal, should be positive and sum to 1.0
    gQ: torch tensor
        Left inner marginal, should be positive and sum to 1.0
    gR: torch tensor
        Right inner marginal, should be positive and sum to 1.0
    gamma: float
        Step-size of the coordinate MD
    full_rank: bool
        If True, initialize a full-rank set of sub-couplings.
        Else if False, initialize with a rank-2 initialization.
    device: str
        'cpu' if running on CPU, else 'cuda' for GPU
    dtype: torch dtype
        Defaults to float64
    rank2_random: bool
        If False, use deterministic rank 2 initialization of Scetbon '21
        Else, use an initialization with randomly sampled vector on simplex.
    max_iter: int
        The maximum number of Sinkhorn iterations for initialized sub-couplings.
    '''
    N1, N2 = a.size(dim=0), b.size(dim=0)
    r, r2 = gQ.size(dim=0), gR.size(dim=0)
    one_N1 = torch.ones((N1), device=device, dtype=dtype)
    one_N2 = torch.ones((N2), device=device, dtype=dtype)
    
    if full_rank:
        '''
        A means of initializing full-rank sub-coupling matrices using randomly sampled matrices
        and Sinkhorn projection onto the polytope of feasible couplings.

        Only non-diagonal initialization for the LC-factorization and handles the case of unequal
        inner left and right ranks (non-square latent couplings).
        '''
        # 1. Q-generation
        # Generate a random (full-rank) matrix as our coupling initialization
        C_random = torch.rand((N1,r), device=device, dtype=dtype)
        '''
        # Generate a random Kernel
        xi_random = torch.exp( -C_random )
        # Generate a random coupling
        u, v = Sinkhorn(xi_random, a, gQ, N1, r, gamma, device=device, max_iter=max_iter, dtype=dtype)
        Q = torch.diag(u) @ xi_random @ torch.diag(v)
        '''
        Q = logSinkhorn(C_random, a, gQ, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # 2. R-generation
        C_random = torch.rand((N2,r2), device=device, dtype=dtype)
        '''
        xi_random = torch.exp( -C_random )
        u, v = Sinkhorn(xi_random, b, gR, N2, r2, gamma, device=device, max_iter=max_iter, dtype=dtype)
        R = torch.diag(u) @ xi_random @ torch.diag(v)'''
        R = logSinkhorn(C_random, b, gR, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # 3. T-generation
        gR, gQ = R.T @ one_N2, Q.T @ one_N1
        C_random = torch.rand((r,r2), device=device, dtype=dtype)
        '''
        xi_random = torch.exp( -C_random )
        u, v = Sinkhorn(xi_random, gQ, gR, r, r2, gamma, device=device, max_iter=max_iter, dtype=dtype)
        T = torch.diag(u) @ xi_random @ torch.diag(v)
        '''
        T = logSinkhorn(C_random, gQ, gR, gamma, max_iter = max_iter, \
                         device=device, dtype=dtype, balanced=True, unbalanced=False)
        
        # Use this to form the inner inverse coupling
        if r == r2:
            Lambda = torch.linalg.inv(T)
        else:
            Lambda = torch.diag(1/gQ) @ T @ torch.diag(1/gR)
            #also, could do: torch.diag(1/gQ) @ T @ torch.diag(1/gR)
    elif r == r2:
        '''
        Rank-2 initialization which requires equal inner ranks and gQ = gR = g.
        This is adapted from "Low-Rank Sinkhorn Factorization" at https://arxiv.org/pdf/2103.04737
        We advise setting full_rank = True and using the first initialization.
        '''
        g = gQ
        lambd = torch.min(torch.tensor([torch.min(a), torch.min(b), torch.min(g)])) / 2

        if rank2_random:
            # Take random sample from probability simplex
            a1 = random_simplex_sample(N1, device=device, dtype=dtype)
            b1 = random_simplex_sample(N2, device=device, dtype=dtype)
            g1 = random_simplex_sample(r, device=device, dtype=dtype)
        else:
            # or initialize exactly as in scetbon 21' ott-jax repo
            g1 = torch.arange(1, r + 1, device=device, dtype=dtype)
            g1 /= g1.sum()
            a1 = torch.arange(1, N1 + 1, device=device, dtype=dtype)
            a1 /= a1.sum()
            b1 = torch.arange(1, N2 + 1, device=device, dtype=dtype)
            b1 /= b1.sum()
        
        a2 = (a - lambd*a1)/(1 - lambd)
        b2 = (b - lambd*b1)/(1 - lambd)
        g2 = (g - lambd*g1)/(1 - lambd)
        
        # Generate Rank-2 Couplings
        Q = lambd*torch.outer(a1, g1).to(device) + (1 - lambd)*torch.outer(a2, g2).to(device)
        R = lambd*torch.outer(b1, g1).to(device) + (1 - lambd)*torch.outer(b2, g2).to(device)
        
        # This is already determined as g (but recomputed anyway)
        gR, gQ = R.T @ one_N2, Q.T @ one_N1
        
        # Last term adds very tiny off-diagonal component for the non-diagonal LC-factorization (o/w the matrix stays fully diagonal)
        T = (1-lambd)*torch.diag(g) + lambd*torch.outer(gR, gQ).to(device)
        Lambda = torch.linalg.inv(T)
    
    return Q, R, T, Lambda

def semi_project_Balanced(xi1, a, g, N1, r, gamma_k, tau, max_iter = 50, \
                          delta = 1e-9, device='cpu', dtype=torch.float64):
    # Lax-inner marginal
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        u_tild = u
        v_tild = v
        v = (g / (xi1.T @ u))**(tau/(tau + gamma_k**-1 ))
        u = (a / (xi1 @ v))
        i+=1
    
    return u, v



def project_Unbalanced(xi1, a, g, N1, r, gamma_k, tau, max_iter = 50, \
                       delta = 1e-9, device='cpu', dtype=torch.float64):
    '''
    Fully-relaxed Sinkhorn with relaxed left and right marginals.
    '''
    # Unbalanced
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u
    v_tild = v
    i = 0
    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        u_tild = u
        v_tild = v
        v = (g / (xi1.T @ u))**(tau/(tau + gamma_k**-1 ))
        u = (a / (xi1 @ v))**(tau/(tau + gamma_k**-1 ))
        i+=1
    
    return u, v




def Sinkhorn(xi, a, b, N1, r, gamma_k, max_iter = 50, \
             delta = 1e-9, device='cpu', dtype=torch.float64):
    '''
    A lightweight impl of Sinkhorn.
    ------Parameters------
    xi: torch tensor
        An N1 x r matrix of the exponentiated positive Sinkhorn kernel.
    a: torch tensor
        Left outer marginal, should be positive and sum to 1.0
    b: torch tensor
        Right outer marginal, should be positive and sum to 1.0
    N1: int
        Dimension 1
    r: int
        Dimension 2
    gamma_k: float
        Step-size used for scaling convergence criterion.
    max_iter: int
        Maximum number of iterations for Sinkhorn loop
    delta: float
        Used for determining convergence to marginals
    device: str
        'cpu' if running on CPU, else 'cuda' for GPU
    dtype: torch dtype
        Defaults to float64
    '''
    u = torch.ones((N1), device=device, dtype=dtype)
    v = torch.ones((r), device=device, dtype=dtype)
    u_tild = u.clone()
    v_tild = v.clone()
    i = 0

    while i == 0 or (i < max_iter and 
                     gamma_k**-1 * torch.max(torch.tensor([torch.max(torch.log(u/u_tild)),torch.max(torch.log(v/v_tild))])) > delta ):
        u_tild = u.clone()
        v_tild = v.clone()
        u = (a / (xi @ v))
        v = (b / (xi.T @ u))
        i+=1
        
    return u, v

def logSinkhorn(grad, a, b, gamma_k, max_iter = 50, \
             device='cpu', dtype=torch.float64, balanced=True, unbalanced=False, tau=None, tau2=None):
    
    log_a = torch.log(a)
    log_b = torch.log(b)

    n, m = a.size(0), b.size(0)
    
    f_k = torch.zeros((n), device=device)
    g_k = torch.zeros((m), device=device)

    epsilon = gamma_k**-1
    
    if not balanced:
        ubc = (tau/(tau + epsilon ))
        if tau2 is not None:
            ubc2 = (tau2/(tau2 + epsilon ))
    
    for i in range(max_iter):
        if balanced and not unbalanced:
            # Balanced
            f_k = f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1))
            g_k = g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0))
        elif not balanced and unbalanced:
            # Unbalanced
            f_k = ubc*(f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc2*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )
        else:
            # Semi-relaxed
            f_k = (f_k + epsilon*(log_a - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=1)) )
            g_k = ubc*(g_k + epsilon*(log_b - torch.logsumexp(Cost(f_k, g_k, grad, epsilon, device=device), axis=0)) )

    P = torch.exp(Cost(f_k, g_k, grad, epsilon, device=device))
    
    return P

def Cost(f, g, Grad, epsilon, device='cpu'):
    '''
    A matrix which is using for the broadcasted log-domain log-sum-exp trick-based updates.
    ------Parameters------
    f: torch.tensor (N1)
        First dual variable of semi-unbalanced Sinkhorn
    g: torch.tensor (N2)
        Second dual variable of semi-unbalanced Sinkhorn
    Grad: torch.tensor (N1 x N2)
        A collection of terms in our gradient for the update
    epsilon: float
        Entropic regularization for Sinkhorn
    device: 'str'
        Device tensors placed on
    '''
    return -( Grad - torch.outer(f, torch.ones(Grad.size(dim=1), device=device)) - torch.outer(torch.ones(Grad.size(dim=0), device=device), g) ) / epsilon

def Delta(vark, varkm1, gamma_k):
    '''
    Convergence criterion for FRLC.
    ------Parameters------
    vark: tuple of 3-tensors
        Tuple of coordinate MD block variables (Q,R,T) at current iter
    varkm1:  tuple of 3-tensors
        Tuple of coordinate MD block variables (Q,R,T) at previous iter
    gamma_k: float
        Coordinate MD step-size
    '''
    Q, R, T = vark
    Q_prev, R_prev, T_prev = varkm1
    error = (gamma_k**-2)*(torch.norm(Q - Q_prev) + torch.norm(R - R_prev) + torch.norm(T - T_prev))
    return error
