# Algorithm Reference Guide

Complete documentation for all 11 metaheuristic algorithms implemented in this project.

---

## Table of Contents

1. [Phase 1: Model Optimization Algorithms](#phase-1-model-optimization-algorithms)
   - [DOE (Taguchi Method)](#doe-taguchi-method)
   - [Particle Swarm Optimization (PSO)](#particle-swarm-optimization-pso)
   - [Grey Wolf Optimizer (GWO)](#grey-wolf-optimizer-gwo)
   - [Whale Optimization Algorithm (WOA)](#whale-optimization-algorithm-woa)
   - [Simulated Annealing (SA)](#simulated-annealing-sa)
   - [Tabu Search](#tabu-search)
   - [Ant Colony Optimization (ACO)](#ant-colony-optimization-aco)
2. [Phase 2: Meta-Optimization & XAI Algorithms](#phase-2-meta-optimization--xai-algorithms)
   - [Cuckoo Search](#cuckoo-search)
   - [Genetic Algorithm (GA)](#genetic-algorithm-ga)
   - [Harmony Search](#harmony-search)
   - [Firefly Algorithm](#firefly-algorithm)

---

## Phase 1: Model Optimization Algorithms

### DOE (Taguchi Method)

**Category**: Systematic Design  
**Inventor**: Genichi Taguchi (1950s)

#### Description
Design of Experiments using Taguchi's orthogonal arrays. Systematically explores the parameter space with minimal experiments while maximizing information gain.

#### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `levels` | 3 | 2-5 | Number of levels per factor |
| `factors` | 4 | 1-10 | Number of hyperparameters |

#### Usage
```python
from algorithms import DOE

doe = DOE(factors=['lstm_units', 'dropout', 'lr', 'batch_size'],
          levels=3)
best_params = doe.optimize(objective_function)
```

#### Results
- **Best Accuracy**: 71.78%
- **Runtime**: 70 seconds
- **Iterations**: 9 (orthogonal array L9)

---

### Particle Swarm Optimization (PSO)

**Category**: Swarm Intelligence  
**Inventors**: James Kennedy & Russell Eberhart (1995)

#### Description
Simulates social behavior of bird flocking. Each particle adjusts its position based on personal experience and swarm knowledge.

#### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_particles` | 10 | 5-50 | Number of particles |
| `c1` | 2.0 | 0.5-3.0 | Cognitive coefficient |
| `c2` | 2.0 | 0.5-3.0 | Social coefficient |
| `w` | 0.7 | 0.1-0.9 | Inertia weight |
| `w_decay` | 0.99 | 0.9-1.0 | Inertia decay rate |

#### Update Equations
```
v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

#### Usage
```python
from algorithms import PSO

pso = PSO(n_particles=10, c1=2.0, c2=2.0, w=0.7)
best_params = pso.optimize(objective_function, iterations=15)
```

#### Results
- **Best Accuracy**: 72.76%
- **Optimal Config**: LSTM=64, Dropout=0.28, LR=0.0053
- **Runtime**: 270 seconds

---

### Grey Wolf Optimizer (GWO)

**Category**: Swarm Intelligence  
**Inventors**: Seyedali Mirjalili, Seyed Mohammad Mirjalili & Andrew Lewis (2014)

#### Description
Mimics the leadership hierarchy and hunting mechanism of grey wolves. The pack is divided into alpha (best), beta (second best), delta (third best), and omega (rest).

#### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_wolves` | 10 | 5-30 | Population size |
| `a_decay` | 2.0 | 1.5-3.0 | Linear decay from a to 0 |

#### Hunting Mechanism
```python
# Encircling prey
D_alpha = |C1 * X_alpha - X|
X1 = X_alpha - A1 * D_alpha

# Position update (average of alpha, beta, delta influence)
X_new = (X1 + X2 + X3) / 3
```

#### Results
- **Best Accuracy**: 72.76%
- **Optimal Config**: LSTM=64, Dropout=0.31, LR=0.0052
- **Runtime**: 323 seconds

---

### Whale Optimization Algorithm (WOA)

**Category**: Swarm Intelligence  
**Inventors**: Seyedali Mirjalili & Andrew Lewis (2016)

#### Description
Simulates the bubble-net hunting strategy of humpback whales. Combines shrinking encircling and spiral updating.

#### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_whales` | 10 | 5-30 | Population size |
| `b` | 1.0 | 0.5-2.0 | Spiral shape constant |

#### Spiral Update
```python
if p < 0.5:
    X_new = X_best - A * D  # Encircling
else:
    X_new = D * exp(b*l) * cos(2*pi*l) + X_best  # Spiral
```

#### Results
- **Best Accuracy**: 72.94%
- **Optimal Config**: LSTM=32, Dropout=0.20, LR=0.001
- **Runtime**: 319 seconds

---

### Simulated Annealing (SA)

**Category**: Single-Solution  
**Inventors**: Scott Kirkpatrick, C. Daniel Gelatt & Mario P. Vecchi (1983)

#### Description
Inspired by the annealing process in metallurgy. Accepts worse solutions with decreasing probability as "temperature" decreases.

#### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `T_init` | 100 | 10-1000 | Initial temperature |
| `T_min` | 0.01 | 0.001-1 | Minimum temperature |
| `alpha` | 0.95 | 0.8-0.99 | Cooling rate |

#### Acceptance Probability
```python
if delta < 0:
    accept = True
else:
    accept = random() < exp(-delta / T)
```

#### Results
- **Best Accuracy**: 72.72%
- **Optimal Config**: LSTM=64, Dropout=0.35, LR=0.005
- **Runtime**: 353 seconds

---

### Tabu Search

**Category**: Memory-Based  
**Inventor**: Fred Glover (1986)

#### Description
Uses memory structures to avoid cycling and guide the search. Maintains a tabu list of recently visited solutions.

#### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tabu_size` | 7 | 3-20 | Tabu list length |
| `n_neighbors` | 10 | 5-30 | Neighbors per iteration |

#### Memory Mechanism
```python
if candidate not in tabu_list:
    if fitness(candidate) > fitness(current):
        current = candidate
        tabu_list.append(candidate)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
```

#### Results
- **Best Accuracy**: 73.43% (WINNER)
- **Optimal Config**: LSTM=128, Dropout=0.45, LR=0.001
- **Runtime**: 902 seconds

---

### Ant Colony Optimization (ACO)

**Category**: Swarm Intelligence  
**Inventors**: Marco Dorigo (1992)

#### Description
Used for feature selection. Ants deposit pheromones on paths, guiding future ants toward better feature subsets.

#### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_ants` | 20 | 10-50 | Number of ants |
| `alpha` | 1.0 | 0.5-2.0 | Pheromone importance |
| `beta` | 2.0 | 1.0-3.0 | Heuristic importance |
| `rho` | 0.5 | 0.1-0.9 | Evaporation rate |

---

## Phase 2: Meta-Optimization & XAI Algorithms

### Cuckoo Search

**Category**: Swarm Intelligence  
**Inventors**: Xin-She Yang & Suash Deb (2009)

#### Description
Based on brood parasitism of cuckoo birds. Uses Levy flights for efficient exploration.

#### Usage: Meta-Optimization
```python
from algorithms import CuckooSearch

# Optimize PSO parameters
bounds = {'c1': (1.0, 3.0), 'c2': (1.0, 3.0), 'w': (0.3, 0.9)}
cs = CuckooSearch(n_nests=15, pa=0.25)
optimal_pso_params = cs.optimize(pso_objective, bounds)
```

#### Results
- Optimized PSO from 72.76% to 74.8%
- Found: c1=1.8, c2=2.1, w=0.6

---

### Genetic Algorithm (GA)

**Category**: Evolutionary  
**Inventor**: John Holland (1975)

#### Description
Used to optimize SHAP parameters. Evolves a population through selection, crossover, and mutation.

#### Optimized Parameters
- SHAP n_samples: 150
- SHAP max_evals: 350
- Quality Score: 0.8234

---

### Harmony Search

**Category**: Music-Inspired  
**Inventors**: Zong Woo Geem, Joong Hoon Kim & G.V. Loganathan (2001)

#### Description
Used to optimize LIME parameters. Mimics musicians improvising to find harmony.

#### Optimized Parameters
- LIME kernel_width: 1.2
- LIME num_features: 12
- Quality Score: 0.8156

---

### Firefly Algorithm

**Category**: Swarm Intelligence  
**Inventor**: Xin-She Yang (2008)

#### Description
Used to optimize Grad-CAM parameters. Based on flashing behavior of fireflies.

#### Optimized Parameters
- Grad-CAM layer_index: -2
- Grad-CAM threshold: 0.45
- Quality Score: 0.8412

---

## References

1. Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. IEEE ICNN.
2. Mirjalili, S. et al. (2014). Grey wolf optimizer. Advances in Engineering Software.
3. Mirjalili, S. & Lewis, A. (2016). The whale optimization algorithm. Advances in Engineering Software.
4. Glover, F. (1986). Future paths for integer programming. Computers & Operations Research.
5. Kirkpatrick, S. et al. (1983). Optimization by simulated annealing. Science.
6. Yang, X.S. & Deb, S. (2009). Cuckoo search via Levy flights. NaBIC.
7. Holland, J.H. (1975). Adaptation in natural and artificial systems. MIT Press.
8. Geem, Z.W. et al. (2001). A new heuristic optimization algorithm. Simulation.
9. Yang, X.S. (2008). Nature-inspired metaheuristic algorithms. Luniver Press.
10. Dorigo, M. (1992). Optimization, learning and natural algorithms. PhD Thesis.
