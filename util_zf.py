import numpy as np
import torch
from scipy.optimize import linprog


###
# from zf nbs 

def factor_mats(C, A, B, device, z=None, c=100, nidx_1=None, nidx_2=None):    
    norm1 = c
    norm2 = A.max()*c
    norm3 = B.max()*c
    
    if z is None:
        # No low-rank factorization applied to the distance matrix
        A = torch.from_numpy(A).to(device)
        B = torch.from_numpy(B).to(device)
        C_factors = (C/ (norm1), torch.eye(C.shape[1]).type(torch.DoubleTensor).to(device))
        A_factors = (A/ (norm2), torch.eye(A.shape[1]).type(torch.DoubleTensor).to(device))
        B_factors = (B/ (norm3), torch.eye(B.shape[1]).type(torch.DoubleTensor).to(device))

    else:
        # Distance matrix factored using SVD
        u, s, v = torch.svd(C)
        print('C done')
        V_C,U_C = torch.mm(u[:,:z], torch.diag(s[:z])), v[:,:z].mT
        u, s, v = torch.svd(torch.from_numpy(A).to(device))
        print('A done')
        V1_A,V1_B = torch.mm(u[:,:z], torch.diag(s[:z])), v[:,:z].mT
        u, s, v = torch.svd(torch.from_numpy(B).to(device))
        print('B done')
        V2_A,V2_B = torch.mm(u[:,:z], torch.diag(s[:z])), v[:,:z].mT
        C_factors, A_factors, B_factors = ((V_C.type(torch.DoubleTensor).to(device)/norm1, U_C.type(torch.DoubleTensor).to(device)/norm1), \
                                       (V1_A.type(torch.DoubleTensor).to(device)/norm2, V1_B.type(torch.DoubleTensor).to(device)/norm2), \
                                       (V2_A.type(torch.DoubleTensor).to(device)/norm3, V2_B.type(torch.DoubleTensor).to(device)/norm3))
    
    return C_factors, A_factors, B_factors

##
# for "registering" cell types

def confusion_matrix(partitionA, partitionB):
    cm = np.zeros((len(partitionA), len(partitionB)))

    for i in range(len(partitionA)):
        for j in range(len(partitionB)):
            cm[i,j] = len(partitionA[i] & partitionB[j])
    return cm 

def cm_custom(labels_A, labels_B):
    labels_A = np.array(labels_A)
    labels_B = np.array(labels_B)
    
    unique_A = np.unique(labels_A)
    unique_B = np.unique(labels_B)
    
    cm = np.zeros((len(unique_A), len(unique_B)))

    for i, itemA in enumerate(unique_A):
        for j, itemB in enumerate(unique_B):
            count = np.sum((labels_A == itemA) & (labels_B == itemB))
            cm[i, j] = count
            # print(f"Comparing {itemA} with {itemB}: count = {count}")
    
    return cm


def get_assignment(C):
    # Flatten the cost matrix for the objective function
    n_samples=C.shape[0]
    c = C.flatten()

    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples

    # Equality constraints matrix A_eq and vector b_eq
    # Constraints to ensure the sum of transported mass equals the probability vectors
    A_eq = []
    b_eq = []

    # Supply constraints
    for i in range(len(a)):
        constraint = np.zeros(C.shape)
        constraint[i, :] = 1
        A_eq.append(constraint.flatten())
        b_eq.append(a[i])

    # Demand constraints
    for j in range(len(b)):
        constraint = np.zeros(C.shape)
        constraint[:, j] = 1
        A_eq.append(constraint.flatten())
        b_eq.append(b[j])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Bounds for each variable (all should be non-negative)
    x_bounds = [(0, None) for _ in range(len(c))]

    # Solve the linear programming problem
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

    # Display the result
    if result.success:
        optimal_transport_plan = result.x.reshape(C.shape)
        # print("Optimal Transport Plan:")
        # print(optimal_transport_plan)
        final_cost = np.sum(optimal_transport_plan * C)
        """
        print("Final Transport Cost:", final_cost)
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(optimal_transport_plan, annot=False, cmap='Blues')
        plt.title("Optimal Transport Plan")
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        plt.xlabel("Target Distribution b")
        plt.ylabel("Source Distribution a")
        plt.show()
        """;
    else:
        print("Optimization failed:", result.message)
        return None

    return optimal_transport_plan

def get_ct_registration_dict(assignment, pred_labels_num, anno_labels_num):
    set_pl = set(pred_labels_num)
    set_an = set(anno_labels_num)

    ct_dict = {}

    ind_to_anno = {j : label for j, label in enumerate(list(set_an))}

    for pred_i, pred_label in enumerate(list(set_pl)):
        ct_dict[pred_label] = ind_to_anno[np.argmax(assignment[pred_i,:])]
    
    return ct_dict

def convert_to_gt_labels(pred_labels_num, anno_labels_num):

    cm = cm_custom(pred_labels_num, anno_labels_num)

    assignment = get_assignment(np.sum(cm) - cm)

    ct_dict = get_ct_registration_dict(assignment,
                                       pred_labels_num,
                                       anno_labels_num)

    labels_pred_converted = [ct_dict[label] for label in list(pred_labels_num)]

    return np.array(labels_pred_converted)