# 🚀 GMM vs. K-Means: When Geometry Breaks Algorithms

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![Simulation](https://img.shields.io/badge/Simulation-Monte_Carlo_%7C_Bootstrap-success.svg)

K-Means is the most widely used clustering algorithm in the industry due to its speed and simplicity, but it has a structural Achilles' heel: **it assumes the world is made of perfect, symmetric spheres**. 

What happens when real-world data presents elliptical clusters, unequal densities, or severe overlap? In this project, we empirically demonstrate why rigid boundary algorithms (Hard Clustering) collapse in the face of spatial complexity, and how **Gaussian Mixture Models (GMM)** manage to decode the latent structure through probabilistic assignment (Soft Clustering).

---

## 🧠 Project Highlights

This repository is not just a model comparison; it is a complete statistical experimentation environment. It includes:

- ⚙️ **Data Generating Mechanism (DGM):** A Gaussian mixture simulation engine in $\mathbb{R}^2$ to create controlled geometric "stress tests."
- 🎲 **Monte Carlo Simulation:** Execution of hundreds of replicas to evaluate the expected value of extrinsic metrics (ARI, Homogeneity, Completeness).
- 🔄 **Label Switching Resolution:** Implementation of the *Hungarian Algorithm* (linear sum assignment) to solve the random permutation of labels during parametric instability analysis.
- 📊 **Advanced Visualizations:** 3D rendering of Probability Density Functions (PDF) with eigenvector projection, fuzzy boundary mapping (alpha mapping), and parametric confidence ellipses.

---

## 📂 Repository Architecture

The source code is structured following MLOps best practices, cleanly separating mathematical logic from visualization and analysis.

~~~text
.
├── notebooks/              # Notebooks
│   ├── Estudio.ipynb       # Main notebook featuring narrative, analysis, and conclusions
├── src/                    # Reusable Python modules
│   ├── data_generation.py  # DGM logic and spatial perturbation
│   ├── simulation.py       # Monte Carlo pipeline and metrics collection
│   ├── bootstrap.py        # Non-parametric resampling and centroid anchoring
│   └── utils.py            # Visual toolkit (Seaborn / Matplotlib 3D)
├── requirements.txt
└── README.md
~~~

---

## 🧪 Stress Tests: The 5 Scenarios

The latent space was progressively deformed to evaluate the breaking point of both algorithms:

| # | Scenario | Geometry | The Algorithmic Challenge |
|---|----------|----------|---------------------------|
| 1 | **Control** | Perfect, separated spheres | Baseline. A technical tie is expected. |
| 2 | **Heteroscedasticity** | Strongly unequal variances | K-Means is prone to "stealing" points from highly dispersed clusters. |
| 3 | **Anisotropy** | Ellipses with varied orientations | K-Means splits ellipses in half. GMM adjusts full covariance matrices. |
| 4 | **Overlap** | Close, intersecting ellipses | Acid test to evaluate uncertainty across decision boundaries. |
| 5 | **Total Collapse** | Pure noise and extreme overlap | Evaluation of *Overfitting* and parametric variance. |

---

## 📈 Key Findings (Bias-Variance Trade-off)

1. **The K-Means Bias:** Faced with anisotropy (Scenario 3), K-Means exhibits a high geometric bias, shifting its centroids far from the true population parameter ($\mu$) by forcing Voronoi cells where they do not belong.
2. **GMM's Robustness:** GMM remains unbiased against unequal variances and covariances, achieving vastly superior partition quality (ARI).
3. **The Complexity Trap:** In pure noise scenarios (Scenario 5), GMM's flexibility works against it (high parametric variance). It overfits the noise in every Bootstrap iteration, whereas K-Means maintains erroneous but stable estimations.

---

## ⚙️ Reproduction Guide

To run the simulations and generate the 3D graphics locally, the following dependencies are required:

~~~bash
# Clone the repository
git clone https://github.com/tu-usuario/gmm-vs-kmeans-simulation.git
cd gmm-vs-kmeans-simulation

# Install dependencies
pip install -r requirements.txt

# Launch the interactive environment
jupyter notebook Estudio.ipynb
~~~

---

## 👥 About the Authors

This project was developed as the final study for the **Numerical Calculus and Simulation** course within the **B.S. in Data Science** program at **Universidad Austral**.

**Data Science Team:**
* Marcos Ziadi
* Juan Martín Leoni
* Facundo Rubiolo
* Máximo Verdondoni

*Special thanks to our teaching staff (Marcos Prunello and Julián L'Heureux) for their academic guidance throughout the methodological design.*
