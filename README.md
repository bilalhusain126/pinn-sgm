# PINN-SGM

Physics-Informed Neural Networks for Score-Based Generative Models in Mathematical Finance.

This repository accompanies the reading course report *Fokker-Planck Constrained Score-Based Generative Models* (FM9561, April 2026).

## Overview

This project combines Physics-Informed Neural Networks (PINNs) with Score-Based Generative Models (SGMs) to extract theoretical score functions from the Fokker-Planck equation and use them to regularize diffusion model training. The Merton structural model is used as the test case throughout.

## Notebooks

The three experiments in the report:

- `notebooks/experiment_1_density_pinn.ipynb` — Density PINN with score extraction on the 1D Merton model (Method I).
- `notebooks/experiment_2_score_pinn.ipynb` — Score-PINN on the 2D Merton model (Method II).
- `notebooks/experiment_3_sgm_comparison.ipynb` — Three-way comparison of Baseline DSM, Lagrangian-constrained DSM, and FP-Diffusion on the 1D Merton model.
