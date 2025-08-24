# Hidden Markov Optimal Transport (HM-OT)



**HM-OT** is a scalable algorithm for learning **cell-state transitions** in time-series single-cell and spatial transcriptomics datasets using the principle of **optimal transport**.


Given a time-series of datasets $X_1, X_2, \dots, X_N$, HM-OT learns:


- A sequence of **latent representations** $Q_1, Q_2, \dots, Q_N$ for each timepoint
- A set of **Markov transition kernels** $\tilde{T}^{(i,i+1)}$ between adjacent timepoints


For single-cell transcriptomics, this corresponds to learning:


- A set of **latent cell states** at each time
- The **least-action transition maps** that explain cellular differentiation across time


<p align="center">
 <img src="images/Figure1.png" alt="Figure 1: HM-OT Schematic" width="1200"/>
</p>


*(Figure 1: HM-OT infers cell-state transitions between timepoints using optimal transport with a Hidden Markov structure.)*


To get started, clone the repository and install dependencies with:
```bash
git clone https://github.com/raphael-group/HM-OT.git
cd HM-OT
# Install dependencies
pip install -r requirements.txt
```


The main method for running HM-OT is in src/HiddenMarkovOT.py. See the demo notebook in notebooks/ for a full example.




**Folder Structure**
```bash
HM-OT/
├── src/                    # Source directory
│   ├── FRLC/               # Low-rank OT solver
│   ├── utils/              # Utility functions
       └── clustering.py   # Functions for computing clusterings
       └── util_LR.py   # Utilities for HM-OT preprocessing
       └── util_zf.py   # Other utilities
│   └── HiddenMarkovOT.py   # Main HM-OT interface
   └── plotting.py   # Plotting functions
├── notebooks/              # Example notebooks
   └── HM_OT_SingleCell_Demo.ipynb # Demo for diff-maps on single-cell
├── images/                 # Visual figures
│   └── Figure1.pdf
├── requirements.txt
└── README.md
```

## Code & Environment

All experiments were run using frozen snapshots of this repository. The table below lists which commit was used for each set of experiments.

| Experiments   | Commit SHA | Notes                  |
|---------------|------------|------------------------|
| 4.1           | [`7fb6785`](https://github.com/<user>/<repo>/commit/7fb67851c4cbaf0e9787d021134dc8541171756d) | Includes Figure 2    |
| 4.2           | [`49dfdbe`](https://github.com/<user>/<repo>/commit/49dfdbbeeed413dd79544d18d8ce00aa115becf7) | Includes Figure 3    |
| 4.3a          | [`7fb6785`](https://github.com/<user>/<repo>/commit/7fb67851c4cbaf0e9787d021134dc8541171756d) | Includes Figure 4b, 4c   |
| 4.3b          | [`dbb00d2`](https://github.com/<user>/<repo>/commit/dbb00d26283085b444060eb4a182f14a6904ca84) | Includes Figure 4d, 4e   |
| 4.4.1         | [`dbb00d2`](https://github.com/<user>/<repo>/commit/dbb00d26283085b444060eb4a182f14a6904ca84) | Includes Figure 5      |
| 4.4.2         | [`dbb00d2`](https://github.com/<user>/<repo>/commit/dbb00d26283085b444060eb4a182f14a6904ca84) | Includes Figure 6      |
| 4.5           | [`3d2f012`](https://github.com/<user>/<repo>/commit/3d2f0126bea2b118d6b79b7e0adc86bc464469fd) | Includes Figure 7    |
| S5.3 (Zebrafish) | [`3d2f012`](https://github.com/<user>/<repo>/commit/3d2f0126bea2b118d6b79b7e0adc86bc464469fd) | Includes Figure S18, S22    |

Additional details:
- **Environment:** see [`requirements.txt`](requirements.txt).  
- **Randomness:** Algorithms include randomized components (e.g. initialization, solvers). Unless otherwise stated, we fixed `seed=42`. Re-running with different seeds produces small numerical variability that does not change qualitative conclusions. Where applicable, we report mean ± s.d. over N runs (or maxima over N runs).
  
**Contact**

If you have any questions or difficulties at all, feel free to reach out to Peter Halmos (ph3641@princeton.edu) or Julian Gold (jg7090@princeton.edu). We're happy to help!


If this work has been helpful to your research, feel free to cite our preprint:
```
@article{Halmos2025,
 title = {Learning Latent Trajectories in Developmental Time Series with Hidden-Markov Optimal Transport},
 url = {http://dx.doi.org/10.1101/2025.02.14.638351},
 DOI = {10.1101/2025.02.14.638351},
 publisher = {Cold Spring Harbor Laboratory},
 author = {Halmos,  Peter and Gold,  Julian and Liu,  Xinhao and Raphael,  Benjamin J.},
 year = {2025},
 month = feb
}
