# J-PIKAN: A Physics-Informed Kolmogorov-Arnold Network Based on Jacobi Orthogonal Polynomials for Solving Fluid Dynamics

## Citation

```bibtex
@article{xiong2025j,
  title={J-PIKAN: A Physics-Informed KAN Network Based on Jacobi Orthogonal Polynomials for solving Fluid Dynamics},
  author={Xiong, Xiong and Lu, Kang and Zhang, Zhuo and Zeng, Zheng and Zhou, Sheng and Deng, Zichen and Hu, Rongchun},
  journal={Communications in Nonlinear Science and Numerical Simulation},
  pages={109414},
  year={2025},
  publisher={Elsevier}
}
```

## Overview

**J-PIKAN** is a novel physics-informed neural network framework that addresses fundamental limitations of traditional MLP-based Physics-Informed Neural Networks (PINNs). By leveraging **Jacobi orthogonal polynomials** as learnable activation functions within a Kolmogorov-Arnold Network (KAN) architecture, J-PIKAN achieves superior performance in solving complex fluid dynamics problems.

### Key Advantages

- **Superior Accuracy**: Delivers 1-2 orders of magnitude improvement in solution accuracy compared to baseline MLPs across different equation types
- **Parameter Efficiency**: Requires only 50% of the parameters compared to basic MLPs while maintaining superior accuracy
- **Improved Optimization**: Exhibits more favorable optimization characteristics with reduced numerical ill-conditioning during training
- **Versatility**: Successfully handles diverse fluid dynamics phenomena from 1D shock waves to high Reynolds number 2D flows

## Methodology

### Jacobi Orthogonal Polynomials

The J-PIKAN framework employs Jacobi orthogonal polynomials $J_n^{(\alpha,\beta)}(x)$ as basis functions for learnable activation functions. These polynomials are defined on the finite interval $[-1,1]$ with parameters $\alpha, \beta > -1$, and satisfy the orthogonality relation:

$$
\int_{-1}^1 (1-x)^\alpha(1+x)^\beta J_m^{(\alpha,\beta)}(x)J_n^{(\alpha,\beta)}(x)dx = \delta_{nm}c_n
$$

For a deep KAN architecture with $L$ layers, each univariate function $\phi_{ij}^{(l)}(x)$ in layer $l$ is defined as:

$$
\phi_{ij}^{(l)}(x) = \sum_{n=0}^N c_{n,ij}^{(l)} J_n^{(\alpha,\beta)}(\tanh(x))
$$

where $c_{n,ij}^{(l)}$ are trainable coefficients and $\tanh(x)$ ensures input normalization to the orthogonality interval $[-1,1]$.

### Network Architecture

The complete deep J-PIKAN network can be expressed as:

$$
u_{\theta}(\mathbf{x}) = \boldsymbol{\Phi}^{(L-1)} \circ \boldsymbol{\Phi}^{(L-2)} \circ \cdots \circ \boldsymbol{\Phi}^{(1)} \circ \boldsymbol{\Phi}^{(0)}(\mathbf{x})
$$

### Loss Function

The total loss function for J-PIKAN incorporates physical constraints:

$$
\mathcal{L}_{J-PIKAN}(\theta) = \lambda_{IC} \mathcal{L}_{IC}(\theta) + \lambda_{BC} \mathcal{L}_{BC}(\theta) + \lambda_{PDE} \mathcal{L}_{PDE}(\theta)
$$

where $\lambda_{IC}$, $\lambda_{BC}$, and $\lambda_{PDE}$ are weighting factors for initial condition, boundary condition, and PDE residual losses, respectively.

## Code Structure

### Implementation: Kovasznay Flow Example

The provided code (`JacobiPINN_Kovasznay_NS.py`) demonstrates J-PIKAN's application to the steady-state Kovasznay flow problem, governed by the 2D incompressible Navier-Stokes equations:

$$
u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0
$$

$$
u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0
$$

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

### Key Components

#### 1. Analytical Solution (`kovasznay_solution`)

```python
def kovasznay_solution(x, y, Re=40):
    """Compute analytical solution of Kovasznay flow"""
    nu = 1/Re
    lamb = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2)
    
    u = 1 - np.exp(lamb*x)*np.cos(2*np.pi*y)
    v = (lamb/(2*np.pi))*np.exp(lamb*x)*np.sin(2*np.pi*y)
    p = 0.5*(1 - np.exp(2*lamb*x))
    
    return u, v, p
```

#### 2. Physics-Informed Neural Network Class (`PhysicsInformedNN`)

**Network Initialization:**
- Accepts Jacobi polynomial models (`JacobiPINN1` or `JacobiPINN2`) with different parameter combinations
- Configures both Adam and L-BFGS optimizers for two-stage training
- Tracks training history for loss and error metrics

**PDE Residual Computation (`net_f`):**
```python
def net_f(self, x):
    """Compute PDE residuals - Steady-state NS equations"""
    u, v, p = self.net_uvp(x)
    
    # Compute partial derivatives using automatic differentiation
    u_x = torch.autograd.grad(u, x, ...)[0][:, 0:1]
    # ... [other derivatives]
    
    # Steady-state NS equations
    f_u = (u*u_x + v*u_y) + p_x - self.nu*(u_xx + u_yy)
    f_v = (u*v_x + v*v_y) + p_y - self.nu*(v_xx + v_yy)
    f_p = u_x + v_y
    
    return f_u, f_v, f_p
```

**Two-Stage Training Strategy:**
1. **Adam Optimizer**: Global stochastic optimization with adaptive learning rates
2. **L-BFGS Optimizer**: Local refinement using quasi-Newton methods for high-precision solutions

#### 3. Data Generation

**Boundary Points:**
- 101 points sampled along each boundary edge (404 total)
- Analytical solution enforced at boundary locations

**Interior Collocation Points:**
- 2000+ randomly sampled points within the domain
- Used for PDE residual loss computation

### Network Configuration

For the Kovasznay flow benchmark:

```python
nu = 1/40              # Viscosity coefficient (Re=40)
degree = 4             # Polynomial degree
size = 30              # Neurons per layer
hidden_layer = 4       # Number of hidden layers
layers = [2, 30, 30, 30, 30, 3]  # Network architecture: [input, hidden..., output]
```

**Total Parameters**: 14,250

## Experimental Results

### Performance on Kovasznay Flow (Re=40)

The paper presents comprehensive comparisons across multiple basis functions using identical architecture [2, 30, 30, 30, 30, 3] with 14,250 parameters:

| Model | Network Architecture | N. Params | Rel. $l_2$ Error $u$ | Rel. $l_2$ Error $v$ | Rel. $l_2$ Error $p$ |
|-------|---------------------|-----------|---------------------|---------------------|---------------------|
| Jacobi($\alpha=\beta=2$) | [2, 30, 30, 30, 30, 3] | 14250 | $(1.5\pm 0.4)\times 10^{-3}$ | $(8.2\pm 2.3)\times 10^{-3}$ | $(7.8\pm 2.1)\times 10^{-3}$ |
| Jacobi($\alpha=\beta=1$) | [2, 30, 30, 30, 30, 3] | 14250 | $(1.9\pm 0.5)\times 10^{-3}$ | $(9.3\pm 2.5)\times 10^{-3}$ | $(8.5\pm 2.4)\times 10^{-3}$ |
| Chebyshev | [2, 30, 30, 30, 30, 3] | 14250 | $(2.7\pm 0.7)\times 10^{-3}$ | $(9.8\pm 2.9)\times 10^{-3}$ | $(8.1\pm 2.2)\times 10^{-3}$ |
| Legendre | [2, 30, 30, 30, 30, 3] | 14250 | $(6.5\pm 2.2)\times 10^{-1}$ | $(1.0\pm 0.3)\times 10^{0}$ | $(9.8\pm 3.1)\times 10^{-1}$ |
| Hermite | [2, 30, 30, 30, 30, 3] | 14250 | $(1.9\pm 0.6)\times 10^{-3}$ | $(9.3\pm 2.7)\times 10^{-3}$ | $(4.2\pm 1.3)\times 10^{-3}$ |
| Taylor | [2, 30, 30, 30, 30, 3] | 14250 | $(6.6\pm 2.3)\times 10^{-1}$ | $(1.7\pm 0.6)\times 10^{0}$ | $(8.7\pm 2.9)\times 10^{-1}$ |

**Key Findings:**
- Jacobi polynomials ($\alpha=\beta=2$) achieve **optimal performance** across all velocity and pressure components
- $u$-component error: $1.5 \times 10^{-3}$ (best accuracy)
- Consistently outperforms Legendre and Taylor polynomials by **2-3 orders of magnitude**
- Demonstrates robust performance with low standard deviation across multiple random seeds

### Comprehensive Benchmark Suite

The paper evaluates J-PIKAN across five canonical fluid dynamics problems:

| Test Cases | N. Params | Adam Iter. | L-BFGS Iter. | Rel. $l_2$ Error |
|------------|-----------|------------|--------------|------------------|
| Burgers | 4300 | $1\times10^4$ | 10000 | $1.1\times 10^{-4}$ |
| KdV | 6300 | $1\times10^4$ | 10000 | $3.7\times 10^{-3}$ |
| Taylor-Green vortex | 6600 | $1\times10^4$ | 10000 | $u$: $4.6\times 10^{-3}$<br>$v$: $6.5\times 10^{-3}$<br>$p$: $8.1\times 10^{-3}$ |
| Kovasznay flow | 6500 | - | 10000 | $u$: $1.1\times 10^{-3}$<br>$v$: $6.3\times 10^{-3}$<br>$p$: $7.6\times 10^{-3}$ |
| Lid-driven cavity(Re=1000) | 6500 | - | 9000 | $1.3\times 10^{-2}$ |

### High Reynolds Number Performance

J-PIKAN demonstrates exceptional capability in handling challenging high Reynolds number flows:

**Lid-Driven Cavity Flow Comparison (Re=1000):**

| Model | Network Architecture | N. Params | Relative $l_2$ Error |
|-------|---------------------|-----------|---------------------|
| Jacobi($\alpha=\beta=2$) | [2, 20, 20, 20, 3] | 4500 | $(4.2\pm 1.3)\times 10^{-2}$ |
| Jacobi($\alpha=\beta=1$) | [2, 20, 20, 20, 3] | 4500 | $(4.1\pm 1.1)\times 10^{-2}$ |
| Chebyshev | [2, 20, 20, 20, 3] | 4500 | $(4.4\pm 1.2)\times 10^{-2}$ |
| MLP | [2, 50, 50, 50, 50, 50, 3] | 10503 | $(2.2\pm 0.8)\times 10^{-1}$ |

**Key Advantage**: J-PIKAN achieves **4× lower error** than MLPs while using only **43% of the parameters** (4500 vs 10503).

**At Re=4000:**
- J-PIKAN [2,30,30,30,30,4] with 14,250 parameters: $6.8 \times 10^{-2}$ error
- MLP [2,100,100,100,100,4] with 21,106 parameters: $1.8 \times 10^{-1}$ error
- **Result**: 32.5% parameter reduction with 62.2% accuracy improvement

## Installation and Usage

### Prerequisites

```bash
pip install torch numpy matplotlib scipy pyDOE
```

### Repository Structure

```
J-PIKAN/
├── KAN_nn/
│   ├── __init__.py
│   ├── jacobi_a1b1.py    # Jacobi polynomials (α=β=1)
│   ├── jacobi_a2b2.py    # Jacobi polynomials (α=β=2)
│   ├── jacobi_a3b3.py    # Jacobi polynomials (α=β=3)
│   ├── chebyshev.py      # Chebyshev polynomials
│   ├── legendre.py       # Legendre polynomials
│   ├── hermite.py        # Hermite polynomials
│   ├── fourier.py        # Fourier series
│   └── bspline.py        # B-spline functions
├── JacobiPINN_Kovasznay_NS.py
├── JacobiPINN_taylor2D_NS.py
└── data/
```

### Running the Kovasznay Flow Example

```python
# Basic usage
python JacobiPINN_Kovasznay_NS.py
```

### Customizing Network Configuration

Modify the main function parameters:

```python
# Network architecture
size = 30              # Number of neurons per layer
hidden_layer = 4       # Number of hidden layers
degree = 4             # Polynomial degree
layers = [2] + [size] * hidden_layer + [3]

# Training configuration
epoch_ADAM = 2000      # Adam optimization iterations (set to 0 to skip)
epoch_LBFGS = 20000    # L-BFGS optimization iterations

# Physical parameters
nu = 1/40              # Viscosity coefficient (Re=40)
```

### Selecting Different Basis Functions

```python
from KAN_nn.jacobi_a1b1 import JacobiPINN1  # α=β=1
from KAN_nn.jacobi_a2b2 import JacobiPINN2  # α=β=2
from KAN_nn.chebyshev import ChebyshevPINN
from KAN_nn.legendre import LegendrePINN

# Define model dictionary
models_dict = OrderedDict({
    'Jacobi_a2b2': JacobiPINN2,
    'Jacobi_a1b1': JacobiPINN1,
    'Chebyshev': ChebyshevPINN,
})
```

### Output Files

The code generates:

1. **Visualization**: Contour plots comparing true vs. predicted velocity and pressure fields
2. **Training History**: Loss and error evolution plots
3. **Comparison Plots**: Bar charts comparing training time and L2 errors across models
4. **MAT File**: Complete results saved for further analysis

## Computational Efficiency Analysis

### Training Time Comparison (Kovasznay Flow)

Based on the paper's comprehensive analysis:

- **J-PIKAN (Jacobi α=β=2)**: 143-187 seconds
- **Chebyshev**: ~180 seconds
- **Legendre**: ~185 seconds
- **MLPs**: 906 seconds

**Result**: J-PIKAN achieves **5× faster training** than MLPs for steady-state problems while maintaining superior accuracy.

### Polynomial Degree Sensitivity

Optimal polynomial degree analysis from the paper:

| Polynomial Degree | Burgers | KdV | Kovasznay Flow |
|-------------------|---------|-----|----------------|
| 2 | $2.3\times 10^{-3}$ | $1.8\times 10^{-2}$ | $4.83\times 10^{-2}$ |
| **4** | **$5.8\times 10^{-4}$** | **$8.6\times 10^{-3}$** | **$5.83\times 10^{-3}$** |
| 6 | $6.2\times 10^{-4}$ | $9.1\times 10^{-3}$ | $6.15\times 10^{-3}$ |
| 8 | $7.1\times 10^{-4}$ | $9.8\times 10^{-3}$ | $6.92\times 10^{-3}$ |

**Recommendation**: Polynomial degree **4** provides optimal accuracy-complexity trade-off across diverse problems.

## Optimization Characteristics

### Hessian Eigenvalue Analysis

J-PIKAN exhibits superior optimization conditioning:

- **Maximum Hessian eigenvalue at convergence**:
  - J-PIKAN (Jacobi): < $1 \times 10^4$
  - MLPs: ≈ $1 \times 10^5$
  
- **Result**: **10× better conditioning** creates smoother loss landscapes and improved training stability

### Convergence Properties

For the Burgers equation:
- J-PIKAN reaches $10^{-4}$ training loss within $5 \times 10^3$ iterations
- MLPs remain at $10^{-2}$ loss level
- **90.8% error reduction** compared to MLPs after $2 \times 10^4$ iterations

## Key Advantages Demonstrated in the Paper

### 1. Superior Accuracy

**Burgers Equation** (network [2,20,20,1]):
- J-PIKAN ($\alpha=\beta=2$): $5.8 \times 10^{-4}$ error with 2,300 parameters
- MLP [2,40,40,40,40,1]: $5.3 \times 10^{-3}$ error with 5,081 parameters
- **Result**: 9× accuracy improvement with 55% fewer parameters

**KdV Equation** (network [2,20,20,20,20,1]):
- J-PIKAN: $8.6 \times 10^{-3}$ error with 5,040 parameters
- MLP [2,50,50,50,50,50,1]: $6.9 \times 10^{-2}$ error with 10,401 parameters
- **Result**: 8× accuracy improvement with 52% fewer parameters

### 2. Parameter Efficiency

Across all benchmarks, J-PIKAN consistently achieves comparable or superior accuracy with **~50% fewer parameters** than traditional MLPs.

### 3. Optimization Stability

The orthogonality properties of Jacobi polynomials lead to:
- More favorable loss landscapes
- Reduced numerical ill-conditioning
- Faster and more stable convergence

### 4. Versatility

Successfully handles:
- 1D nonlinear equations (Burgers, KdV)
- 2D unsteady flows (Taylor-Green vortex)
- Steady-state flows (Kovasznay)
- High Reynolds number complex flows (Lid-driven cavity, Re=4000)

## Practical Guidelines

### Parameter Selection

Based on extensive experimental validation:

1. **Jacobi Parameters**: $\alpha = \beta = 2$ or $\alpha = \beta = 1$ consistently achieve optimal performance
2. **Polynomial Degree**: Degree 4 provides best accuracy-complexity balance
3. **Learning Rate Schedule**: Cosine annealing demonstrates superior convergence across all problems
4. **Network Depth**: 4 hidden layers sufficient for most fluid dynamics problems

### When to Use J-PIKAN

J-PIKAN is particularly effective for:
- Problems requiring high accuracy with limited computational resources
- Complex nonlinear PDEs with sharp gradients or shock waves
- High Reynolds number flows
- Multi-scale phenomena requiring spectral convergence properties
- Applications where optimization stability is critical

## Limitations and Future Directions

As noted in the paper:

### Current Limitations
- Significant memory consumption for high-order polynomial implementations
- Training stability issues with very high-order polynomials in discontinuous problems
- Computational overhead for problems with third-order derivatives

### Future Research Directions
1. Variable separation architectures for reduced computational complexity
2. Adaptive polynomial degree selection strategies
3. Hybrid approaches combining different basis functions
4. Extension to asymmetric parameter combinations ($\alpha \neq \beta$)

## Conclusion

J-PIKAN represents a significant advancement in physics-informed neural networks for fluid dynamics, addressing fundamental limitations of traditional MLP-based approaches through the strategic use of Jacobi orthogonal polynomials. The comprehensive experimental validation across diverse benchmarks demonstrates:

- **1-2 orders of magnitude accuracy improvement** over baseline MLPs
- **50% parameter reduction** while maintaining superior accuracy
- **Improved optimization conditioning** with 10× better Hessian eigenvalue characteristics
- **Robust performance** across shock waves, solitons, and high Reynolds number flows

The provided code and methodology offer a practical, efficient framework for solving complex fluid dynamics problems using physics-informed deep learning.



## Acknowledgments

This work was conducted at:
- School of Mathematics and Statistics, Northwestern Polytechnical University
- Department of Engineering Mechanics, Northwestern Polytechnical University
- MIIT Key Laboratory of Dynamics and Control of Complex Systems

---

*Published in Communications in Nonlinear Science and Numerical Simulation, 2025, Elsevier*

