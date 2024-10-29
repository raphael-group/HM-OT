# HM-OT
Hidden-Markov Optimal Transport (HM-OT)

Given a time-series of datasets $( X )_{i=1}^{N}$, HM-OT learns a series of latent representations of each dataset $( Q_{i} )_{i=1}^{N}$ and a series of latent Markov transition kernels $( \Tilde{T}^{(i,i+1)} )_{i=1}^{N-1}$ for the clusters these representations map to. For single-cell transcriptomics, this jointly learns latent cell-states for all times t, and the transition matrix between these cell-states which is of least-action with respect to the optimal transport (OT) principle.
