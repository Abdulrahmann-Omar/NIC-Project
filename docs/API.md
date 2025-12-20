# API Reference

Technical API documentation for the NIC Project modules.

---

## 3D Visualizations Module

**Location**: `visualizations/algorithm_3d.py`

### Functions

#### `create_pso_animation(n_particles=20, n_iterations=30, bounds=(-5, 5))`

Creates an animated 3D visualization of Particle Swarm Optimization.

**Parameters**:
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_particles` | int | 20 | Number of particles in swarm |
| `n_iterations` | int | 30 | Animation frames |
| `bounds` | tuple | (-5, 5) | Search space bounds |

**Returns**: `plotly.graph_objects.Figure`

**Example**:
```python
from algorithm_3d import create_pso_animation

fig = create_pso_animation(n_particles=15, n_iterations=25)
fig.write_html("pso_animation.html")
fig.show()
```

---

#### `create_gwo_animation(n_wolves=15, n_iterations=25, bounds=(-5, 5))`

Creates an animated 3D visualization of Grey Wolf Optimizer.

**Parameters**:
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_wolves` | int | 15 | Number of wolves in pack |
| `n_iterations` | int | 25 | Animation frames |
| `bounds` | tuple | (-5, 5) | Search space bounds |

**Returns**: `plotly.graph_objects.Figure`

**Wolf Colors**:
- Red: Alpha (best)
- Orange: Beta (2nd best)
- Yellow: Delta (3rd best)
- Gray: Omega (rest)

---

#### `create_search_space_comparison()`

Creates side-by-side 3D comparison of test functions.

**Returns**: `plotly.graph_objects.Figure` with 3 subplots:
1. Sphere function (unimodal)
2. Rastrigin function (multimodal)
3. Rosenbrock function (valley)

---

### Test Functions

#### `sphere_function(x, y)`
Simple convex function: `f(x,y) = x² + y²`

#### `rastrigin_function(x, y)`
Highly multimodal: `f(x,y) = 20 + x² - 10cos(2πx) + y² - 10cos(2πy)`

#### `rosenbrock_function(x, y)`
Banana valley: `f(x,y) = (1-x)² + 100(y-x²)²`

---

## Dashboard Module

**Location**: `dashboard/app.py`

### Page Structure

| Page | Function | Description |
|------|----------|-------------|
| Home | Overview | Project metrics & architecture |
| Live Prediction | Inference | Real-time sentiment analysis |
| Algorithm Comparison | Visualization | Phase 1 results charts |
| XAI Explorer | Explanations | SHAP/LIME/Grad-CAM views |
| Convergence Analysis | Charts | Algorithm convergence plots |
| 3D Visualizations | Interactive | PSO/GWO 3D animations |
| Bonus Features | Extra | Statistical tests, downloads |

### Helper Functions

#### `safe_image(filename, caption=None)`
Displays image if exists, otherwise shows placeholder.

#### `load_trained_model()`
Loads best available Keras model from multiple paths.

#### `load_results()`
Loads CSV result files with fallback to empty DataFrames.

---

## Data Files

### Phase 1 Results (`results/phase1_results.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `Algorithm` | str | Algorithm name |
| `Best_Accuracy` | float | Best validation accuracy |
| `LSTM_Units` | int | Optimal LSTM units |
| `Dropout` | float | Optimal dropout rate |
| `Learning_Rate` | float | Optimal learning rate |
| `Runtime_Seconds` | float | Total optimization time |
| `Iterations` | int | Number of iterations run |

### XAI Results (`results/phase2_xai_results.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `XAI_Method` | str | SHAP/LIME/Grad-CAM/IG |
| `Optimizer` | str | Algorithm used for optimization |
| `Quality_Score` | float | Explanation quality (0-1) |
| `Stability` | float | Consistency score (0-1) |
| `Computational_Cost` | str | Low/Medium/High |

---

## Configuration Files

### Jupyter Book (`book/_config.yml`)

```yaml
title: Nature-Inspired Computation for Deep Learning
author: Abdulrahman Omar
execute:
  execute_notebooks: auto
repository:
  url: https://github.com/Abdulrahmann-Omar/NIC-Project
```

### Streamlit Requirements (`dashboard/requirements.txt`)

```
streamlit>=1.28
plotly>=5.18
pandas>=2.0
numpy>=1.24
scipy>=1.11
tensorflow>=2.10
```
