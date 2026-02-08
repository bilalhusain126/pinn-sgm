# PINN-SGM-Finance: Theory-Constrained Score Estimation for Quantitative Finance

A unified framework combining Physics-Informed Neural Networks (PINNs) and Score-Based Generative Models (SGMs) for theory-guided probability density estimation in financial applications.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This package implements a novel methodology that integrates **theoretical constraints from stochastic calculus** with **data-driven generative modeling** to improve tail probability estimation in sparse-data regimes. The framework:

1. **Solves the Fokker-Planck equation** using Physics-Informed Neural Networks (PINNs)
2. **Extracts theoretical score functions** from the PINN solution via automatic differentiation
3. **Blends theoretical and empirical scores** for hybrid score-based generation

### Key Features

- 🎯 **Physics-Informed**: Enforces PDE constraints during neural network training
- 📊 **Fokker-Planck Solver**: Specialized implementation for the Merton structural model
- 🔬 **Score Extraction**: Automatic differentiation-based score computation
- 🌊 **Hybrid Score Fields**: Blends theory and data for improved tail estimation
- 📈 **Financial Applications**: Designed for credit risk and option pricing
- 🚀 **GPU Acceleration**: Full PyTorch implementation with CUDA support

---

## Mathematical Foundation

### 1. The Fokker-Planck Equation

Consider a stochastic process $X_t \in \mathbb{R}^d$ governed by the Itô SDE:

```math
dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t
```

The probability density function $p(x, t)$ of $X_t$ satisfies the **Fokker-Planck equation**:

```math
\frac{\partial p}{\partial t} = -\sum_{i=1}^{d} \frac{\partial}{\partial x_i}[\mu_i(x,t) p] + \frac{1}{2}\sum_{i=1}^{d}\sum_{j=1}^{d} \frac{\partial^2}{\partial x_i \partial x_j}[D_{ij}(x,t) p]
```

where $D(x,t) = \sigma(x,t) \sigma(x,t)^\top$ is the diffusion tensor.

### 2. Merton Structural Model

For firm asset value $V_t$ following Geometric Brownian Motion:

```math
dV_t = \mu V_t dt + \sigma V_t dW_t
```

We transform to log-space $X_t = \ln V_t$, yielding the advection-diffusion equation:

```math
\frac{\partial p}{\partial t} + \alpha \frac{\partial p}{\partial x} - \frac{\sigma^2}{2} \frac{\partial^2 p}{\partial x^2} = 0
```

where $\alpha = \mu - \frac{\sigma^2}{2}$ is the effective drift.

**Analytical Solution**: For constant coefficients with initial condition $\delta(x - x_0)$:

```math
p(x, t) = \mathcal{N}(x; x_0 + \alpha t, \sigma^2 t)
```

### 3. Physics-Informed Neural Networks (PINNs)

Approximate the solution using a neural network $u_\theta(x, t)$ and minimize the composite loss:

```math
\mathcal{L}(\theta) = \mathcal{L}_{\text{PDE}}(\theta) + \mathcal{L}_{\text{IC}}(\theta) + \mathcal{L}_{\text{BC}}(\theta)
```

where:
- **PDE Loss**: $\mathcal{L}_{\text{PDE}} = \frac{1}{N_f} \sum_{i=1}^{N_f} |\mathcal{F}[u_\theta](x_i, t_i)|^2$
- **Initial Condition Loss**: $\mathcal{L}_{\text{IC}} = \frac{1}{N_0} \sum_{i=1}^{N_0} |u_\theta(x_i, 0) - u_0(x_i)|^2$
- **Boundary Condition Loss**: $\mathcal{L}_{\text{BC}} = \frac{1}{N_b} \sum_{i=1}^{N_b} |\mathcal{B}[u_\theta](x_i, t_i)|^2$

**Key Innovation**: Automatic differentiation computes exact derivatives $\frac{\partial u_\theta}{\partial x}$, $\frac{\partial^2 u_\theta}{\partial x^2}$ needed for the PDE residual.

### 4. Score Function Extraction

The **score function** is the gradient of the log-density:

```math
s(x, t) = \nabla_x \log p(x, t)
```

Given a trained PINN $p_\theta(x, t)$, extract the theoretical score via:

```math
s_{\text{theory}}(x, t) \approx \frac{\nabla_x p_\theta(x, t)}{p_\theta(x, t)}
```

### 5. Hybrid Score Field

Blend empirical score $s_\theta(x, t)$ (learned from data) with theoretical score $s_{\text{theory}}(x, t)$ (from PINN):

```math
\hat{s}(x, t) = (1 - \phi_t) s_\theta(x, t) + \phi_t s_{\text{theory}}(x, t)
```

where $\phi_t \in [0, 1]$ is a time-dependent weight:
- **$\phi_0 = 0$**: Near the mean, use empirical score (captures market idiosyncrasies)
- **$\phi_T = 1$**: In the tails, use theoretical score (enforces structural constraints)

### 6. Modified Langevin Dynamics

Use the hybrid score in the Langevin corrector step:

```math
x_{i+1} = x_i + \epsilon \hat{s}(x_i, t) + \sqrt{2\epsilon} z_i, \quad z_i \sim \mathcal{N}(0, I)
```

This guides the diffusion process toward theoretically consistent tail behavior.

---

## Installation

### From Source

```bash
git clone https://github.com/bilalsalehhusain/pinn-sgm-finance.git
cd pinn-sgm-finance
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0

---

## Quick Start

### Example 1: Solve Fokker-Planck Equation for Merton Model

```python
import torch
from pinn_sgm.equations import FokkerPlanckMerton
from pinn_sgm.solvers import PINNSolver
from pinn_sgm.nets import DensityMLP
from pinn_sgm.config import MertonModelConfig, PINNConfig, TrainingConfig

# Configure Merton model parameters
merton_config = MertonModelConfig(
    mu=0.05,       # Asset drift
    sigma=0.2,     # Asset volatility
    x0=0.0         # Initial log-asset value
)

# Create Fokker-Planck equation
equation = FokkerPlanckMerton(
    config=merton_config,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Create neural network
network = DensityMLP(
    input_dim=2,
    hidden_dims=[64, 64, 64],
    activation='tanh'
)

# Configure PINN solver
pinn_config = PINNConfig(
    T=1.0,
    x_min=-5.0,
    x_max=5.0,
    num_collocation=10000,
    num_initial=1000,
    enforce_normalization=True
)

training_config = TrainingConfig(
    epochs=5000,
    learning_rate=1e-3,
    batch_size=1024
)

# Create and train solver
solver = PINNSolver(
    equation=equation,
    network=network,
    pinn_config=pinn_config,
    training_config=training_config
)

results = solver.train()
print(f"Final loss: {results['final_loss']:.6f}")

# Evaluate on test grid
x_test = torch.linspace(-5, 5, 200).unsqueeze(-1)
t_test = torch.full((200, 1), 0.5)  # At t=0.5
p_pred = solver.predict(x_test, t_test)
```

### Example 2: Extract and Visualize Score Function

```python
from pinn_sgm.utils import ScoreExtractor, plot_score_field

# Extract score from trained PINN
score_extractor = ScoreExtractor(
    network=solver.network,
    device=solver.device
)

# Compute score at specific points
x = torch.tensor([[0.0], [1.0], [-1.0]])
t = torch.tensor([[0.5], [0.5], [0.5]])
scores = score_extractor(x, t)
print(f"Scores: {scores}")

# Visualize score field evolution
fig = plot_score_field(
    score_extractor,
    x_range=(-5.0, 5.0),
    time_points=[0.1, 0.5, 1.0]
)
fig.savefig('score_evolution.png')
```

### Example 3: Hybrid Score for Diffusion Models

```python
from pinn_sgm.utils import hybrid_score
from pinn_sgm.config import ScoreModelConfig

# Configure hybrid scoring
score_config = ScoreModelConfig(
    phi_start=0.0,     # Use empirical score at t=0
    phi_end=1.0,       # Use theoretical score at t=T
    interpolation='linear'
)

# Define empirical score network (learned from data)
def empirical_score(x, t):
    # Your trained score network here
    return score_network(x, t)

# Compute hybrid score
x = torch.randn(100, 1)
t = torch.rand(100, 1)
s_hybrid = hybrid_score(
    x, t,
    score_empirical=empirical_score,
    score_theoretical=score_extractor,
    config=score_config,
    T=1.0
)
```

---

## Package Structure

```
pinn-sgm-finance/
├── src/
│   └── pinn_sgm/
│       ├── __init__.py           # Package initialization, logging setup
│       ├── config.py              # Configuration dataclasses
│       ├── equations/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract PDE interface
│       │   └── fokker_planck.py   # Fokker-Planck implementations
│       ├── solvers/
│       │   ├── __init__.py
│       │   └── pinn_solver.py     # PINN solver with automatic differentiation
│       ├── nets/
│       │   ├── __init__.py
│       │   └── mlp.py             # Neural network architectures
│       └── utils/
│           ├── __init__.py
│           ├── score_extraction.py  # Score computation and hybrid blending
│           └── visualizations.py    # Plotting utilities
├── notebooks/
│   └── demo_merton_model.ipynb    # Demonstration notebook
├── tests/
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Advanced Usage

### Custom PDE Equations

Implement your own PDE by subclassing `BasePDE`:

```python
from pinn_sgm.equations import BasePDE
import torch

class CustomPDE(BasePDE):
    def __init__(self, param1, param2, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2

    def pde_residual(self, x, t, u, u_t, u_x, u_xx):
        # Implement your PDE residual
        residual = u_t + self.param1 * u_x - self.param2 * u_xx
        return residual

    def initial_condition(self, x):
        # Implement initial condition
        return torch.exp(-x**2)
```

### Time-Dependent Weight Functions

Customize the hybrid score blending:

```python
from pinn_sgm.config import ScoreModelConfig

# Exponential weight: rapid transition from empirical to theoretical
config_exp = ScoreModelConfig(
    phi_start=0.0,
    phi_end=1.0,
    interpolation='exponential'
)

# Sigmoid weight: sharp transition at midpoint
config_sigmoid = ScoreModelConfig(
    phi_start=0.0,
    phi_end=1.0,
    interpolation='sigmoid'
)
```

### Save and Load Models

```python
# Save trained model
solver.save('trained_pinn_merton.pth')

# Load model
solver_loaded = PINNSolver(equation, network, pinn_config, training_config)
solver_loaded.load('trained_pinn_merton.pth')
```

---

## Visualization Gallery

The package includes comprehensive visualization utilities:

- **Density Evolution**: `plot_density_evolution()` - Compare PINN vs. analytical solutions
- **Score Fields**: `plot_score_field()` - Visualize $\nabla_x \log p(x, t)$
- **Heatmaps**: `plot_density_heatmap()` - 2D spatiotemporal evolution
- **3D Surfaces**: `plot_3d_surface()` - 3D visualization of $p(x, t)$
- **Error Analysis**: `plot_error_analysis()` - Absolute and relative errors
- **Training History**: `plot_training_history()` - Loss components over epochs

---

## References

### Academic Papers

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.

2. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B.** (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. International Conference on Learning Representations (ICLR).

3. **Merton, R. C.** (1974). *On the Pricing of Corporate Debt: The Risk Structure of Interest Rates*. The Journal of Finance, 29(2), 449-470.

### PhD Research Document

This implementation is based on:
- **Chapter 2**: Preliminaries (Fokker-Planck Equation, PINNs, SGMs)
- **Chapter 3**: Theory-Constrained Score Estimation

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{husain2025pinnsgm,
  author = {Husain, Bilal Saleh},
  title = {PINN-SGM-Finance: Theory-Constrained Score Estimation for Quantitative Finance},
  year = {2025},
  url = {https://github.com/bilalsalehhusain/pinn-sgm-finance},
  institution = {Western University}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Bilal Saleh Husain**
PhD Student, Financial Mathematics
Western University, London, ON, Canada
Email: bhusain@uwo.ca

---

## Acknowledgments

- Supervisor: [Your Advisor's Name]
- Western University, Department of Statistical and Actuarial Sciences
- FM9561 - Special Topics in Mathematical Finance

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Future Enhancements

- [ ] Multi-dimensional Fokker-Planck solvers
- [ ] Integration with popular diffusion model libraries (diffusers, score_sde)
- [ ] Variance reduction techniques for Monte Carlo comparison
- [ ] Real market data calibration examples
- [ ] Option pricing with smile dynamics
- [ ] Credit risk portfolio models

---

**Last Updated**: February 2025
