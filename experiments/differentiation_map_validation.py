


from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity



def compute_centroids(X, Q):
    gQ = np.sum(Q, axis = 0)
    centroids = np.diag(1 / gQ) @ Q.T @ X
    return centroids

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
    observed_keys = []
    
    for _, row in edges_df.iterrows():
        
        if row["x"] in G.nodes and row["y"] in G.nodes:
            
            type_1 = labels_G[row["x"]]
            type_2 = labels_G[row["y"]]
            
            if (type_1, type_2) in observed_keys:
                # Some rows repeat (perhaps with other metadata) in Shendure et al graph
                continue
            else:
                observed_keys.append(( type_1, type_2) )
            
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
                    
                    #print(f'Proposed transition {Label1[i_max]} to {type_2}')
                    score = NPMI(T12[idx1, idx2], \
                                 g1[idx1], g2[idx2])
                    
                    '''
                    if type_2 in Label1:
                        # If a diagonal-dominant transition [Handled separately]
                        idx1, idx2 = Label1.index(type_2), Label2.index(type_2)
                        score_diag = NPMI(T12[idx1, idx2], \
                                 g1[idx1], g2[idx2])
                        #print(f'Score: {score}, Score Diag: {score_diag}')
                    '''
                    
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
                    
                    idx1, idx2 = Label1.index(type), Label2.index(type)
                    
                    score = NPMI(T12[idx1, idx2], \
                                 g1[idx1], g2[idx2])
                    
                    # Append score for transition for each timepoint
                    key = type
                    if key not in edge_scores_diagonal:
                        edge_scores_diagonal[key] = []
                    edge_scores_diagonal[key].append(score)
    
    return edge_scores, edge_scores_diagonal


def yield_differentiation_graph(nodes_df, edges_df, plotting=False, include_downstream=False):
    
    # Assume G is your graph
    G = nx.DiGraph()
    # Add nodes/edges from your Ts, etc.
    
    # Add nodes with metadata
    for _, row in nodes_df.iterrows():
        
        G.add_node(
            row["meta_group"],  # PGa_M1 etc.
            label=row["celltype_new"],
            size=row["celltype_num"]
        )
    
    # Add edges from edges.txt
    for _, row in edges_df.iterrows():
        # Ensure x and y are in the graph
        if row["x"] in G.nodes and row["y"] in G.nodes:
            G.add_edge(row["x"], row["y"], edge_type=row["edge_type"])
    
    if include_downstream:
        # Iterate over source nodes
        for src in list(G.nodes):
            for dst in nx.descendants(G, src):
                # If descendant of src, then add indirect edge (not time-indexed)
                if not G.has_edge(src, dst):
                    G.add_edge(src, dst, edge_type="indirect")
    
    # Prepare for drawing
    labels_G = {n: G.nodes[n]["label"] for n in G.nodes}
    
    if plotting:
        
        sizes = [max(G.nodes[n]["size"], 1) * 0.0005 for n in G.nodes]  # scale up for visibility
        
        root = "PGa_M1"
        levels = {}
        visited = set()
        queue = [(root, 0)]
        
        while queue:
            node, level = queue.pop(0)
            if node not in visited:
                visited.add(node)
                levels[node] = level
                for neighbor in G.successors(node):
                    queue.append((neighbor, level + 1))
        
        # Group nodes by level
        level_nodes = defaultdict(list)
        for node, lvl in levels.items():
            level_nodes[lvl].append(node)
        
        # Assign positions
        pos = {}
        x_spacing = 300.0
        y_spacing = -3
        for level, nodes in level_nodes.items():
            for i, node in enumerate(nodes):
                pos[node] = (i * x_spacing, level * y_spacing)
        
        # Fallback for unvisited nodes
        for node in G.nodes:
            if node not in pos:
                pos[node] = (0, 0)
        
        nx.draw(G, pos=pos, with_labels=True)
        
        # Clear weird background
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(15, 6), facecolor='white', dpi=400)
        
        # Draw graph
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='-|>')
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='skyblue', edgecolors='black')
        
        # Draw labels slightly above the nodes
        label_pos = {k: (v[0], v[1] + 0.3) for k, v in pos.items()}
        text_items = nx.draw_networkx_labels(G, label_pos, labels_G, font_size=3)
        
        # Rotate each label
        for _, text in text_items.items():
            text.set_rotation(30)  # or 45, 90, etc.
            
        # Cleanup
        ax.set_facecolor('white')
        plt.title("Developmental Trajectory of Cell Types", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
    return G, labels_G
    
def evaluate_coclusters(Qs_ann, Qs_u, Ts_u, Ts_s, Ts_m,
                        X1, X2, X3):
    
    Q1 = Qs_ann[0]
    Q3 = Qs_ann[2]
    
    # Centroids at t1, t2 of annotated clusters
    C1 = compute_centroids(X1, Q1)
    C3 = compute_centroids(X3, Q3)
    
    # Annotated clusters
    Q2 = Qs_ann[1]
    # Learned clusters
    Q2_U = Qs_u[1]
    
    C2 = compute_centroids(X2, Q2)
    C2_U = compute_centroids(X2, Q2_U)
    
    T12_moscot, T23_moscot = Ts_m
    T12_s, T23_s = Ts_s
    T12_u, T23_u = Ts_u
    
    C1_pred_moscot = np.diag(1 / np.sum(T12_moscot, axis=1)) @ T12_moscot @ C2
    C1_pred_s = np.diag(1 / np.sum(T12_s, axis=1)) @ T12_s @ C2
    C1_pred_u = np.diag(1 / np.sum(T12_u, axis=1)) @ T12_u @ C2_U
    
    sim_pre = cosine_similarity(C1, C1_pred_moscot).diagonal()
    #print(f"Mean cosine similarity (t2 transferred to t1), moscot transitions + annotated types: {sim_pre.mean():.3f}")
    weighted_score = np.sum(np.sum(Q1, axis=0) * sim_pre)
    print(f"Weighted cosine similarity moscot (t2 transferred to t1): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C1, C1_pred_s).diagonal()
    #print(f"Mean cosine similarity (t2 transferred to t1), hm-ot transitions + annotated types: {sim_pre.mean():.3f}")
    weighted_score = np.sum(np.sum(Q1, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot supervised (t2 transferred to t1): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C1, C1_pred_u).diagonal()
    #print(f"Mean cosine similarity (t2 transferred to t1), hm-ot unsupervised types+transitions: {sim_pre.mean():.3f}")
    weighted_score = np.sum(np.sum(Q1, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot unsupervised (t2 transferred to t1): {weighted_score:.3f}")
    
    C3_pred_moscot = np.diag(1 / np.sum(T23_moscot, axis=0)) @ T23_moscot.T @ C2
    C3_pred_s = np.diag(1 / np.sum(T23_s, axis=0)) @ T23_s.T @ C2
    C3_pred_u = np.diag(1 / np.sum(T23_u, axis=0)) @ T23_u.T @ C2_U
    
    sim_pre = cosine_similarity(C3, C3_pred_moscot).diagonal()
    #print(f"Mean cosine similarity (t2 transferred to t3), moscot transitions + annotated types: {sim_pre.mean():.3f}")
    weighted_score = np.sum(np.sum(Q3, axis=0) * sim_pre)
    print(f"Weighted cosine similarity moscot (t2 transferred to t3): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C3, C3_pred_s).diagonal()
    #print(f"Mean cosine similarity (t2 transferred to t3), hm-ot transitions + annotated types: {sim_pre.mean():.3f}")
    weighted_score = np.sum(np.sum(Q3, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot supervised (t2 transferred to t3): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C3, C3_pred_u).diagonal()
    #print(f"Mean cosine similarity (t2 transferred to t3), hm-ot unsupervised types+transitions: {sim_pre.mean():.3f}")
    weighted_score = np.sum(np.sum(Q3, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot unsupervised (t2 transferred to t3): {weighted_score:.3f}")
    
    return

def evaluate_coclusters(Qs_ann, Qs_u, Ts_u, Ts_s, Ts_m,
                        X1, X2, X3):
    
    Q1 = Qs_ann[0]
    Q3 = Qs_ann[2]
    
    # Centroids at t1, t3 of annotated clusters
    C1 = compute_centroids(X1, Q1)
    C3 = compute_centroids(X3, Q3)

    Q1_u = Qs_u[0]
    Q3_u = Qs_u[2]
    
    # Centroids at t1, t3 of inferred clusters
    C1_u = compute_centroids(X1, Q1_u)
    C3_u = compute_centroids(X3, Q3_u)
    
    # Annotated clusters
    Q2 = Qs_ann[1]
    # Learned clusters
    Q2_U = Qs_u[1]

    # Annotated and unsupervised centroids
    C2 = compute_centroids(X2, Q2)
    C2_U = compute_centroids(X2, Q2_U)

    # Transition matrices
    T12_moscot, T23_moscot = Ts_m
    T12_s, T23_s = Ts_s
    T12_u, T23_u = Ts_u

    # Predicted centroids through the differentiation map T
    C1_pred_moscot = np.diag(1 / np.sum(T12_moscot, axis=1)) @ T12_moscot @ C2
    C1_pred_s = np.diag(1 / np.sum(T12_s, axis=1)) @ T12_s @ C2
    C1_pred_u = np.diag(1 / np.sum(T12_u, axis=1)) @ T12_u @ C2_U
    
    sim_pre = cosine_similarity(C1, C1_pred_moscot).diagonal()
    weighted_score = np.sum(np.sum(Q1, axis=0) * sim_pre)
    print(f"Weighted cosine similarity moscot (t2 transferred to t1): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C1, C1_pred_s).diagonal()
    weighted_score = np.sum(np.sum(Q1, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot supervised (t2 transferred to t1): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C1_u, C1_pred_u).diagonal()
    weighted_score = np.sum(np.sum(Q1, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot unsupervised (t2 transferred to t1): {weighted_score:.3f}")
    
    C3_pred_moscot = np.diag(1 / np.sum(T23_moscot, axis=0)) @ T23_moscot.T @ C2
    C3_pred_s = np.diag(1 / np.sum(T23_s, axis=0)) @ T23_s.T @ C2
    C3_pred_u = np.diag(1 / np.sum(T23_u, axis=0)) @ T23_u.T @ C2_U
    
    sim_pre = cosine_similarity(C3, C3_pred_moscot).diagonal()
    weighted_score = np.sum(np.sum(Q3, axis=0) * sim_pre)
    print(f"Weighted cosine similarity moscot (t2 transferred to t3): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C3, C3_pred_s).diagonal()
    weighted_score = np.sum(np.sum(Q3, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot supervised (t2 transferred to t3): {weighted_score:.3f}")
    
    sim_pre = cosine_similarity(C3_u, C3_pred_u).diagonal()
    weighted_score = np.sum(np.sum(Q3, axis=0) * sim_pre)
    print(f"Weighted cosine similarity hm-ot unsupervised (t2 transferred to t3): {weighted_score:.3f}")
    
    return


