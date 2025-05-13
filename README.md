# NMTF for Breast Cancer Subtyping

This repository contains the implementation of a two-phase computational workflow designed for multi-omics data integration and subtype-specific association prediction in breast cancer, using Non-negative Matrix Tri-Factorization (NMTF).

The code supports the study presented in the paper:

**Inferring Breast Cancer Subtype Associations Using an Original Omics Integration Based on Non-negative Matrix Tri-Factorization**  
Camila Riccio-Rengifo, Silvia Cascianelli, Gaia Ceddia, Marco Masseroli  
*CIBB 2023, Springer Lecture Notes in Computer Science, LNCS 14513, Chapter 19*

## 📌 Overview

The workflow is composed of two main stages:
1. **Patient Subtyping** via masked NMTF decomposition of a multi-partite network built from TCGA breast cancer data.
2. **Subtype Association Prediction** for genes and miRNAs using subtype-specific subnetworks and node embeddings (Node2Vec) combined with machine learning classifiers (e.g., XGBoost, Random Forest).

## 🗂 Structure

- `mask_train_test.py`: code to apply stratified masking on patient labels.
- `method_NMTF.py`: core implementation of the NMTF algorithm and its update rules.
- `NMTF_subnetworks.py`: subnetwork generation and node embedding with Node2Vec.
- `parallel_initialization.py`: parallel SVD-based initialization for NMTF.
- `pecanpy_embedding.py`: embedding with PecanPy implementation of Node2Vec.

## 📖 Reference

If you use this code, please cite:

@incollection{riccio2025nmtf,
  author    = {Riccio-Rengifo, Camila and Cascianelli, Silvia and Ceddia, Gaia and Masseroli, Marco},
  title     = {Inferring Breast Cancer Subtype Associations Using an Original Omics Integration Based on Non-negative Matrix Tri-Factorization},
  booktitle = {Computational Intelligence Methods for Bioinformatics and Biostatistics},
  editor    = {Vettoretti, Mario and Tavazzi, Emanuela and Longato, Enrico and Baruzzo, Giulio and Bellato, Michele},
  series    = {Lecture Notes in Computer Science},
  volume    = {14513},
  year      = {2025},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-031-90714-2_19},
  url       = {https://doi.org/10.1007/978-3-031-90714-2_19}
}
