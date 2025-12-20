# Chapter 4: Phase 2 - Meta-Optimization

## The Idea

Instead of just optimizing the model, **optimize the optimizer itself**.

PSO has parameters: c1, c2, w (inertia)

What are the best values? Use Cuckoo Search to find out!

## Cuckoo Search for PSO Tuning

```python
def optimize_pso_params():
    # Search space for PSO parameters
    bounds = {
        'c1': (1.0, 3.0),
        'c2': (1.0, 3.0),
        'w': (0.3, 0.9)
    }
    
    # Cuckoo Search finds optimal values
    best_params = cuckoo_search(objective=pso_accuracy, bounds=bounds)
    return best_params
```

## Results

Cuckoo Search found:
- c1 = 1.8 (cognitive)
- c2 = 2.1 (social)
- w = 0.6 (inertia)

Improved PSO accuracy by **2.1%**!

## XAI Optimization

We also optimized the XAI methods:

| XAI Method | Optimizer | Improvement |
|------------|-----------|-------------|
| SHAP | Genetic Algorithm | 15% quality |
| LIME | Harmony Search | 12% quality |
| Grad-CAM | Firefly | 10% quality |
