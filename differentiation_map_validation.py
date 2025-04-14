


from collections import defaultdict
import pandas as pd
import numpy as np

def NPMI(Tij, gi, gj, eps=1e-12):
    
    # Add epsilon to avoid division by zero or log(0)
    Tij_safe = max(Tij, eps)
    gi_safe = max(gi, eps)
    gj_safe = max(gj, eps)

    #print(f'Tij: {Tij}, gi: {gi}, gj: {gj}')
    
    pmi = np.log2(Tij_safe / (gi_safe * gj_safe))
    npmi_val = pmi / (-np.log2(Tij_safe))
    
    return npmi_val

def score_from_graph(_Qs, _Ts, _labels_Q, timepoints,
                         G, labels_G, edges_df, 
                         diagonal_from_G = False
                        ):
    edge_scores = {}
    """
    Exhaustively score every edge in a given "ground-truth" graph G by its NPMI through the joint law.
    """
    for _, row in edges_df.iterrows():
        
        if row["x"] in G.nodes and row["y"] in G.nodes:
            
            type_1 = labels_G[row["x"]]
            type_2 = labels_G[row["y"]]
            
            for i in range(len(timepoints) - 1):
                
                t1 = timepoints[i]
                t2 = timepoints[i + 1]
                
                #print(f'{t1} to {t2}')
                if i == 0:
                    Q1 = _Qs[0]
                    Label1 = _labels_Q[0]
                else:
                    Q1 = Q2
                    Label1 = Label2
                
                Q2 = _Qs[i+1]
                Label2 = _labels_Q[i+1]
                
                T12 = _Ts[i]
                
                g1 = np.sum(Q1, axis = 0)
                g2 = np.sum(Q2, axis = 0)
                
                if (type_1 in Label1 and type_2 in Label2) and (type_1 != type_2):
                    
                    #print(f'Mapping {type_1} to {type_2} at times {t1} to {t2}')
                    
                    idx1, idx2 = Label1.index(type_1), Label2.index(type_2)
                    i_max = np.argmax(T12[:, idx2])
                    
                    #print(f'HM-OT proposed transition {Label1[i_max]} to {type_2}')
                    score = NPMI(T12[idx1, idx2], \
                                 g1[idx1], g2[idx2])
                    
                    if type_2 in Label1:
                        # If a diagonal-dominant transition
                        idx1, idx2 = Label1.index(type_2), Label2.index(type_2)
                        score_diag = NPMI(T12[idx1, idx2], \
                                 g1[idx1], g2[idx2])
                        #print(f'Score: {score}, Score Diag: {score_diag}')
                    
                    # Append score for transition for each timepoint
                    key = (type_1, type_2)
                    
                    if key not in edge_scores:
                        edge_scores[key] = []
                    edge_scores[key].append(score)


    edge_scores_diagonal = {}

    if diagonal_from_G:
        """
        Iterate through diagonal edges for all nodes in Graph
        """
        for node in G.nodes:
            
            type = labels_G[node]
            print(f'Mapping diagonal transition {type} to self at adjacent timepoint.')
            
            for i in range(len(timepoints) - 1):
                
                t1 = timepoints[i]
                t2 = timepoints[i + 1]
                
                if i == 0:
                    Q1 = _Qs[0]
                    Label1 = _labels_Q[0]
                else:
                    Q1 = Q2
                    Label1 = Label2
                
                Q2 = _Qs[i+1]
                Label2 = _labels_Q[i+1]
                
                T12 = _Ts[i]
                
                g1 = np.sum(Q1, axis = 0)
                g2 = np.sum(Q2, axis = 0)
                
                if type in Label1 and type in Label2:
                    
                    print(f'Mapping {type} to self at times {t1} to {t2}')
                    idx1, idx2 = Label1.index(type), Label2.index(type)
                    
                    score = NPMI(T12[idx1, idx2], \
                                 g1[idx1], g2[idx2])
                    
                    # Append score for transition for each timepoint
                    key = type
                    if key not in edge_scores_diagonal:
                        edge_scores_diagonal[key] = []
                    edge_scores_diagonal[key].append(score)
    else:
        """
        Iterate through all diagonal edges, whether in graph or not.
        """
        for i in range(len(timepoints) - 1):
            
            t1 = timepoints[i]
            t2 = timepoints[i + 1]
            
            if i == 0:
                Q1 = _Qs[0]
                Label1 = _labels_Q[0]
            else:
                Q1 = Q2
                Label1 = Label2
            
            Q2 = _Qs[i+1]
            Label2 = _labels_Q[i+1]
            
            T12 = _Ts[i]
            
            g1 = np.sum(Q1, axis = 0)
            g2 = np.sum(Q2, axis = 0)

            for type in Label2:
                if type in Label1:
                    
                    print(f'Mapping {type} to self at times {t1} to {t2}')
                    idx1, idx2 = Label1.index(type), Label2.index(type)
                    
                    score = NPMI(T12[idx1, idx2], \
                                 g1[idx1], g2[idx2])
                    
                    # Append score for transition for each timepoint
                    key = type
                    if key not in edge_scores_diagonal:
                        edge_scores_diagonal[key] = []
                    edge_scores_diagonal[key].append(score)
    
    return edge_scores, edge_scores_diagonal


