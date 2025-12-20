# ❓ Frequently Asked Questions

Common questions and answers about the NIC Project.

---

## Table of Contents

- [General Questions](#general-questions)
- [Installation Issues](#installation-issues)
- [Usage Questions](#usage-questions)
- [Performance Questions](#performance-questions)
- [Results Questions](#results-questions)
- [Contributing Questions](#contributing-questions)

---

## General Questions

### What is this project?

This project implements 11 nature-inspired metaheuristic algorithms to optimize a BiLSTM neural network for sentiment analysis. Instead of manual hyperparameter tuning or brute-force grid search, we use algorithms inspired by bird flocks (PSO), wolf packs (GWO), whale hunting (WOA), and more.

### Who is it for?

- **ML Researchers**: Compare metaheuristic algorithms on a common task
- **Students**: Learn optimization algorithms with working implementations
- **Data Scientists**: Quickly tune models without manual effort
- **Developers**: Integrate optimizers into your own projects

### What can I do with it?

1. **Optimize your models**: Use PSO, GWO, or other algorithms to find optimal hyperparameters
2. **Compare algorithms**: Benchmark multiple optimizers on your problem
3. **Learn algorithms**: Study implementations and visualizations
4. **Generate explanations**: Use optimized XAI methods for interpretability

### Why nature-inspired algorithms?

| Approach | Pros | Cons |
|----------|------|------|
| **Grid Search** | Exhaustive | O(n^k) time complexity |
| **Random Search** | Fast | No guidance |
| **Bayesian** | Smart | Expensive per iteration |
| **Nature-Inspired** | Intelligent + Fast | Requires tuning |

Nature-inspired algorithms balance exploration (searching new areas) and exploitation (refining good solutions) naturally.

---

## Installation Issues

### "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**:
```bash
pip install tensorflow>=2.10
```

For GPU support:
```bash
pip install tensorflow[and-cuda]
```

### "pip install fails with version conflicts"

**Solution**: Use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### "CUDA/GPU not detected"

**Check**: 
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Solutions**:
1. Install CUDA Toolkit 11.x
2. Install cuDNN 8.x
3. Verify with `nvidia-smi`

### "Modal authentication failed"

**Solution**:
```bash
modal setup  # Opens browser for authentication
```

If issues persist, try:
```bash
modal token new
```

### "Streamlit won't start"

**Solutions**:
1. Check port availability: `netstat -an | findstr 8501`
2. Clear cache: `streamlit cache clear`
3. Reinstall: `pip install --upgrade streamlit`

---

## Usage Questions

### Which algorithm should I use?

| Scenario | Recommended | Why |
|----------|-------------|-----|
| **Best accuracy** | Tabu Search | Memory prevents cycling |
| **Fast results** | PSO | Quick convergence |
| **Balanced** | GWO | Good exploration/exploitation |
| **Black-box model** | SA | Works with any objective |
| **Discrete parameters** | GA | Natural for discrete spaces |

### How do I tune algorithm parameters?

**Option 1**: Use defaults (good starting point)
```python
pso = PSO()  # Uses sensible defaults
```

**Option 2**: Meta-optimization (recommended)
```python
from algorithms import CuckooSearch
cs = CuckooSearch()
best_pso_params = cs.optimize(pso_objective, pso_search_space)
```

**Option 3**: Manual tuning guidelines
| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| Population | 5 (fast) | 30 (thorough) |
| Iterations | 10 (fast) | 50 (thorough) |
| w (inertia) | 0.3 (exploitation) | 0.9 (exploration) |

### How do I add my own model?

```python
def my_objective(params):
    model = create_my_model(
        hidden_size=params['hidden_size'],
        lr=params['learning_rate']
    )
    accuracy = train_and_evaluate(model)
    return accuracy  # Return metric to MAXIMIZE

search_space = {
    'hidden_size': [32, 64, 128],
    'learning_rate': (0.0001, 0.01)
}

pso = PSO()
best_params = pso.optimize(my_objective, search_space)
```

### How do I customize the dashboard?

Edit `dashboard/app.py` and add new pages:
```python
if page == "My Custom Page":
    st.header("Custom Analysis")
    # Add your content
```

Then commit and push - Streamlit Cloud will auto-deploy.

---

## Performance Questions

### Why is optimization slow?

**Causes**:
1. Model training is slow (main bottleneck)
2. Too many particles/iterations
3. CPU instead of GPU

**Solutions**:
1. Reduce epochs per evaluation: `epochs=3` instead of `10`
2. Use fewer particles: `n_particles=5`
3. Use GPU: Modal.com H100 or Colab T4
4. Enable early stopping

### How can I speed up training?

```python
# Use early stopping
from tensorflow.keras.callbacks import EarlyStopping

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True)
]

model.fit(X, y, epochs=50, callbacks=callbacks)
```

### What are the memory requirements?

| Component | RAM Required |
|-----------|-------------|
| Model (BiLSTM) | 2-4 GB |
| PSO (10 particles) | 200 MB |
| SHAP explanations | 1-2 GB |
| Dashboard | 500 MB |

**Minimum**: 8 GB RAM (CPU) or 16 GB GPU memory

### Can I run on CPU only?

Yes, but slower:
```bash
export CUDA_VISIBLE_DEVICES=""
python run_optimization.py
```

Expected slowdown: 5-10x compared to GPU.

---

## Results Questions

### How do I reproduce the results?

```bash
git clone https://github.com/Abdulrahmann-Omar/NIC-Project.git
cd NIC-Project
pip install -r requirements.txt
modal run src/phase1_modal_observable.py
```

All experiments use seed=42 for reproducibility.

### Why are my results different?

**Possible causes**:
1. **Different hardware**: GPU differences affect floating-point operations
2. **TensorFlow version**: Update to TensorFlow 2.x
3. **Random seed**: Ensure seed=42 is set
4. **Data split**: Use the same train/test split

**Fix**:
```python
import numpy as np
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

### Is the improvement statistically significant?

Yes! We performed paired t-tests:

| Comparison | p-value | Significant at α=0.05 |
|------------|:-------:|:---------------------:|
| Tabu vs PSO | 0.0089 | ✅ Yes |
| Tabu vs WOA | 0.0213 | ✅ Yes |
| Meta-PSO vs PSO | 0.0043 | ✅ Yes |

### Can I use these results in my paper?

Yes! Please cite:

```bibtex
@software{omar2024nic,
  author = {Omar, Abdulrahman},
  title = {Nature-Inspired Computation for Deep Learning Optimization},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Abdulrahmann-Omar/NIC-Project}
}
```

---

## Contributing Questions

### How do I contribute?

1. Fork the repository
2. Create feature branch: `git checkout -b feature/NewAlgorithm`
3. Make changes with tests
4. Submit PR to `master` branch

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

### What coding standards should I follow?

- **Style**: PEP 8
- **Docstrings**: Google format
- **Type hints**: Required for public functions
- **Tests**: Required for new features

```python
def optimize(
    self,
    objective: Callable[[Dict], float],
    search_space: Dict[str, Any],
    iterations: int = 15
) -> Tuple[Dict, float]:
    """
    Run optimization.
    
    Args:
        objective: Function to maximize.
        search_space: Parameter bounds.
        iterations: Number of iterations.
        
    Returns:
        Tuple of (best_params, best_score).
    """
```

### How long does PR review take?

Typically 2-3 days. Faster for:
- Bug fixes
- Documentation improvements
- Small feature additions

### Can I add a new algorithm?

Yes! Create a new file in `algorithms/` following the template:

```python
from .base import BaseOptimizer

class MyAlgorithm(BaseOptimizer):
    def __init__(self, ...):
        ...
    
    def optimize(self, objective, search_space, iterations):
        # Your implementation
        return best_params, best_score
```

Submit PR with:
- Implementation
- Unit tests
- Documentation
- Example usage

---

## Still Have Questions?

- 🐛 **Bug Report**: [Open an issue](https://github.com/Abdulrahmann-Omar/NIC-Project/issues/new?template=bug_report.md)
- 💡 **Feature Request**: [Open an issue](https://github.com/Abdulrahmann-Omar/NIC-Project/issues/new?template=feature_request.md)
- 💬 **Discussion**: [Start a discussion](https://github.com/Abdulrahmann-Omar/NIC-Project/discussions)
- 📧 **Email**: [Contact author](mailto:abdu.omar.muhammad@gmail.com)
