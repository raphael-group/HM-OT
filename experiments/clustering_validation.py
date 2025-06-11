


from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compute_centroids(X, Q):
    gQ = np.sum(Q, axis = 0)
    centroids = np.diag(1 / gQ) @ Q.T @ X
    return centroids

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

def get_Jaccard_Table(adata_1, adata_2, 
                      clustering1, clustering2,
                     cluster_label='cluster',
                     n_top_genes = 50, 
                      print_DE=False, 
                      overlap_lst=None,
                     genes_to_print=None):
    
    if overlap_lst is not None:
        
        mask1 = np.isin(clustering1, overlap_lst)
        mask2 = np.isin(clustering2, overlap_lst)
        adata1 = adata_1[mask1].copy()
        adata2 = adata_2[mask2].copy()
        clustering1 = np.array(clustering1)[mask1]
        clustering2 = np.array(clustering2)[mask2]
        
    else:
        adata1 = adata_1.copy()
        adata2 = adata_2.copy()
    
    adata1.obs[cluster_label] = clustering1
    adata2.obs[cluster_label] = clustering2

    # Normalize, log1p
    sc.pp.normalize_total(adata1)
    sc.pp.log1p(adata1)
    sc.pp.normalize_total(adata2)
    sc.pp.log1p(adata2)
    
    adata1.obs[cluster_label] = adata1.obs[cluster_label].astype('category')
    adata2.obs[cluster_label] = adata2.obs[cluster_label].astype('category')
    
    sc.tl.rank_genes_groups(adata1, groupby=cluster_label, method='wilcoxon')
    sc.tl.rank_genes_groups(adata2, groupby=cluster_label, method='wilcoxon')
    
    sc.pl.rank_genes_groups(adata1, n_genes=n_top_genes,
                            sharey=False, title='t1 DE')
    sc.pl.rank_genes_groups(adata2, n_genes=n_top_genes,
                            sharey=False, title='t2 DE')
    '''
    top_genes1 = get_top_de_genes(adata1, 
                                 top_n=n_top_genes)
    top_genes2 = get_top_de_genes(adata2, 
                                 top_n=n_top_genes)
    
    if print_DE:
        print('Printing DE Genes for Time 1.')
        for cluster, df in top_genes1.items():
            print(f"\n=== Top DE Genes for Cluster {cluster} ===")
            print(df.to_string(index=False))
        print('Printing DE Genes for Time 2.')
        for cluster, df in top_genes2.items():
            print(f"\n=== Top DE Genes for Cluster {cluster} ===")
            print(df.to_string(index=False))'''

    top_1 = get_top_de_genes(adata1, top_n=n_top_genes)
    top_2 = get_top_de_genes(adata2, top_n=n_top_genes)
    
    clusters1 = sorted(top_1.keys())
    clusters2 = sorted(top_2.keys())
    
    jaccard_matrix = np.zeros((len(clusters1), len(clusters2)))
    
    for i, c1 in enumerate(clusters1):
        genes1 = set(top_1[c1]['gene'])
        for j, c2 in enumerate(clusters2):
            genes2 = set(top_2[c2]['gene'])
            intersection = genes1 & genes2
            union = genes1 | genes2
            jaccard_matrix[i, j] = len(intersection) / len(union) if union else 0.0
            
            if c1 == c2 and print_DE is True:
                print(f"\n=== Cluster {c1} ===")
                print(f"Intersection genes ({len(intersection)}):\n", sorted(intersection))
                
                df1 = top_1[c1]
                df2 = top_2[c1]
                
                # slice the two DE tables down to just those intersecting genes
                df1_int = df1[df1['gene'].isin(intersection)].reset_index(drop=True)
                df2_int = df2[df2['gene'].isin(intersection)].reset_index(drop=True)

                if genes_to_print is None:
                    print("\nTop DE stats for time 1 (intersection):")
                    print(df1_int.to_string(index=False))
                    print("\nTop DE stats for time 2 (intersection):")
                    print(df2_int.to_string(index=False))
                else:
                    df1_sub = df1_int[df1_int['gene'].isin(genes_to_print)].reset_index(drop=True)
                    df2_sub = df2_int[df2_int['gene'].isin(genes_to_print)].reset_index(drop=True)
                    if not df1_sub.empty or not df2_sub.empty:
                        if not df1_sub.empty:
                            print(" Time 1 DE stats for selected genes:")
                            print(df1_sub.to_string(index=False))
                        if not df2_sub.empty:
                            print(" Time 2 DE stats for selected genes:")
                            print(df2_sub.to_string(index=False))
                    else:
                        print(f'No genes in list represented in cluster {c1}!')
    
    jaccard_df = pd.DataFrame(jaccard_matrix, index=clusters1, columns=clusters2)
    
    return jaccard_df


def plot_jaccard_matrix(
    jaccard_df,
    ax,
    annot_text=False,
    title=None,
    cmap=None,
    x_lab='Time 1 Cluster',
    y_lab='Time 2 Cluster'):
    
    if cmap is None:
        cmap = sns.light_palette("navy", as_cmap=True)
    
    fig = ax.get_figure()
    # get axis size in inches, then convert to pixels
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_px = bbox.height  * fig.dpi
    width_px  = bbox.width   * fig.dpi
    
    cell_h = height_px / jaccard_df.shape[0]
    cell_w = width_px  / jaccard_df.shape[1]
    # choose a fraction of the smaller dimension (e.g. 50%)
    fontsize = (min(cell_h, cell_w) * 0.05)

    sns.heatmap(
        jaccard_df,
        annot=annot_text,
        fmt=".2f" if annot_text else None,
        cmap=cmap,
        cbar_kws={"label": "Jaccard Index"},
        annot_kws={"fontsize": fontsize},
        linewidths=0.5,
        linecolor='white',
        square=True,
        ax=ax,
        vmin=0, vmax=1,      # ensure both share same color scale
    )
    
    ax.set_xlabel(x_lab, fontsize=13)
    ax.set_ylabel(y_lab, fontsize=13)
    
    if title:
        ax.set_title(title, fontsize=12, weight='bold', pad=6)
    ax.tick_params(labelsize=10)
    
    return ax
    
def get_top_de_genes(adata, top_n=10):
    
    de_results = adata.uns['rank_genes_groups']
    groups = de_results['names'].dtype.names
    top_genes = {}
    
    for group in groups:
        top_genes[group] = pd.DataFrame({
            'gene': de_results['names'][group][:top_n],
            'logfc': de_results['logfoldchanges'][group][:top_n],
            'pval': de_results['pvals'][group][:top_n],
            'score': de_results['scores'][group][:top_n]
        })
    
    return top_genes

def get_zscore_corr_summary(adata_1, adata_2, clustering1, clustering2, 
                             cluster_label='cluster', n_top_genes=50,
                             overlap_lst=None, plot=False):
    """
    Computes Z-score correlations across DE genes for co-clustered groups between timepoints.
    Returns a DataFrame of results with optional scatterplots.
    """
    if overlap_lst is not None:
        mask1 = np.isin(clustering1, overlap_lst)
        mask2 = np.isin(clustering2, overlap_lst)
        adata1 = adata_1[mask1].copy()
        adata2 = adata_2[mask2].copy()
        clustering1 = np.array(clustering1)[mask1]
        clustering2 = np.array(clustering2)[mask2]
    else:
        adata1 = adata_1.copy()
        adata2 = adata_2.copy()
    
    adata1.obs[cluster_label] = clustering1
    adata2.obs[cluster_label] = clustering2
    
    # Normalize and log1p
    sc.pp.normalize_total(adata1)
    sc.pp.log1p(adata1)
    sc.pp.normalize_total(adata2)
    sc.pp.log1p(adata2)

    adata1.obs[cluster_label] = adata1.obs[cluster_label].astype('category')
    adata2.obs[cluster_label] = adata2.obs[cluster_label].astype('category')

    sc.tl.rank_genes_groups(adata1, groupby=cluster_label, method='wilcoxon')
    sc.tl.rank_genes_groups(adata2, groupby=cluster_label, method='wilcoxon')
    
    def get_de_table(adata):
        de = adata.uns['rank_genes_groups']
        groups = de['names'].dtype.names
        df_dict = {}
        for group in groups:
            df_dict[group] = pd.DataFrame({
                'gene': de['names'][group],
                'zscore': de['scores'][group]
            })
        return df_dict

    de1 = get_de_table(adata1)
    de2 = get_de_table(adata2)
    
    shared_clusters = overlap_lst if overlap_lst is not None else sorted(set(de1.keys()).intersection(de2.keys()))
    
    records = []

    for c in shared_clusters:
        df1 = de1[c].set_index('gene')
        df2 = de2[c].set_index('gene')
        
        common_genes = df1.index.intersection(df2.index)
        if len(common_genes) < 10:
            continue

        z1 = df1.loc[common_genes, 'zscore']
        z2 = df2.loc[common_genes, 'zscore']
        r, p = pearsonr(z1, z2)
        records.append({
            'cluster': c,
            'pearson_r': r,
            'pval': p,
            'n_genes': len(common_genes)
        })

        if plot:
            plt.figure(figsize=(5, 4))
            plt.scatter(z1, z2, alpha=0.6)
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.xlabel('Z-scores at time 1')
            plt.ylabel('Z-scores at time 2')
            plt.title(f'Cluster {c}: Z-score correlation\nr = {r:.2f}, p = {p:.2e}')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(records)

'''
def plot_zscore_correlation_per_cluster(adata_1, adata_2, clustering1, clustering2, 
                                         cluster_label='cluster', n_top_genes=50, overlap_lst=None):
    if overlap_lst is not None:
        mask1 = np.isin(clustering1, overlap_lst)
        mask2 = np.isin(clustering2, overlap_lst)
        adata1 = adata_1[mask1].copy()
        adata2 = adata_2[mask2].copy()
        clustering1 = np.array(clustering1)[mask1]
        clustering2 = np.array(clustering2)[mask2]
    else:
        adata1 = adata_1.copy()
        adata2 = adata_2.copy()
    
    adata1.obs[cluster_label] = clustering1
    adata2.obs[cluster_label] = clustering2
    
    sc.pp.normalize_total(adata1)
    sc.pp.log1p(adata1)
    sc.pp.normalize_total(adata2)
    sc.pp.log1p(adata2)

    adata1.obs[cluster_label] = adata1.obs[cluster_label].astype('category')
    adata2.obs[cluster_label] = adata2.obs[cluster_label].astype('category')

    # Run DE
    sc.tl.rank_genes_groups(adata1, groupby=cluster_label, method='wilcoxon')
    sc.tl.rank_genes_groups(adata2, groupby=cluster_label, method='wilcoxon')

    # All genes (not just top)
    def get_de_table(adata):
        de = adata.uns['rank_genes_groups']
        groups = de['names'].dtype.names
        df_dict = {}
        for group in groups:
            df_dict[group] = pd.DataFrame({
                'gene': de['names'][group],
                'zscore': de['scores'][group]
            })
        return df_dict
    
    de1 = get_de_table(adata1)
    de2 = get_de_table(adata2)
    
    shared_clusters = overlap_lst if overlap_lst is not None else set(de1.keys()).intersection(de2.keys())
    
    # Plot correlations
    for c in shared_clusters:
        df1 = de1[c].set_index('gene')
        df2 = de2[c].set_index('gene')
        
        common_genes = df1.index.intersection(df2.index)
        if len(common_genes) < 10:
            continue  # Skip underpowered clusters
        
        z1 = df1.loc[common_genes, 'zscore']
        z2 = df2.loc[common_genes, 'zscore']
        r, p = pearsonr(z1, z2)

        # Plot
        plt.figure(figsize=(5, 4))
        plt.scatter(z1, z2, alpha=0.6)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel('Z-scores at time 1')
        plt.ylabel('Z-scores at time 2')
        plt.title(f'Cluster {c}: Z-score correlation\nr = {r:.2f}, p = {p:.2e}')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()'''

        
