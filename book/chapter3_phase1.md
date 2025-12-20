# Chapter 3: Phase 1 - Model Optimization

## Setup: Modal.com H100 GPU

We used Modal.com for cloud GPU access:

```python
import modal

app = modal.App("nic-optimization")

@app.function(gpu="H100", timeout=3600)
def run_optimization():
    # Run all 6 algorithms
    pass
```

## Results

| Algorithm | Accuracy | Runtime |
|-----------|----------|---------|
| DOE | 71.78% | 70s |
| PSO | 72.76% | 270s |
| **Tabu Search** | **73.43%** | 902s |
| GWO | 72.76% | 323s |
| WOA | 72.94% | 319s |
| SA | 72.72% | 353s |

**Winner**: Tabu Search with 73.43% accuracy

## Key Insights

1. Memory-based search (Tabu) outperformed swarm methods
2. Longer runtime correlated with better results
3. Checkpointing saved 2+ hours of re-runs
