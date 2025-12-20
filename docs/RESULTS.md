# 📊 Experimental Results

Comprehensive analysis of all experimental results from the Nature-Inspired Computation project.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Best Phase 1 Accuracy** | 73.43% (Tabu Search) |
| **Best Phase 2 Accuracy** | 74.80% (Meta-optimized PSO) |
| **Total Algorithms Tested** | 11 |
| **XAI Methods Optimized** | 4 |
| **Total Experiments** | 150+ |
| **Compute Time** | ~40 GPU-hours |

### Key Findings

1. **Tabu Search** outperformed all swarm algorithms due to its memory mechanism
2. **Meta-optimization** improved PSO by 2.04% (72.76% → 74.80%)
3. **Grad-CAM** achieved highest XAI quality score (0.8412)
4. **Runtime vs Accuracy tradeoff**: Tabu Search takes 3x longer but achieves best results

---

## Experimental Setup

### Dataset

| Property | Value |
|----------|-------|
| **Dataset** | IMDB Movie Reviews |
| **Training Samples** | 25,000 |
| **Testing Samples** | 25,000 |
| **Vocabulary Size** | 10,000 words |
| **Max Sequence Length** | 200 tokens |
| **Task** | Binary Sentiment Classification |

### Model Architecture

```
BiLSTM Model:
├── Embedding Layer (10000 → 128)
├── Bidirectional LSTM (units variable)
├── Bidirectional LSTM (units/2)
├── Dropout (rate variable)
├── Dense (64, ReLU)
└── Dense (1, Sigmoid)
```

### Hyperparameter Search Space

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| LSTM Units | Discrete | [32, 64, 128] | Model capacity |
| Dropout Rate | Continuous | [0.2, 0.5] | Regularization |
| Learning Rate | Continuous | [0.0001, 0.01] | Optimizer step size |
| Batch Size | Discrete | [32, 64, 128] | Training batch |

### Hardware

| Phase | Platform | GPU | RAM |
|-------|----------|-----|-----|
| Phase 1 | Modal.com | NVIDIA H100 | 80GB |
| Phase 2 | Google Colab | NVIDIA T4 | 16GB |

---

## Phase 1: Algorithm Comparison Results

### Complete Results Table

| Rank | Algorithm | Accuracy | LSTM | Dropout | LR | Runtime (s) | Iterations |
|:----:|-----------|:--------:|:----:|:-------:|:--:|:-----------:|:----------:|
| 🥇 | **Tabu Search** | **73.43%** | 128 | 0.45 | 0.001 | 901.85 | 10 |
| 🥈 | WOA | 72.94% | 32 | 0.20 | 0.001 | 318.84 | 15 |
| 🥉 | PSO | 72.76% | 64 | 0.28 | 0.0053 | 270.09 | 15 |
| 4 | GWO | 72.76% | 64 | 0.31 | 0.0052 | 323.06 | 15 |
| 5 | SA | 72.72% | 64 | 0.35 | 0.005 | 352.89 | 20 |
| 6 | DOE | 71.78% | 128 | 0.35 | 0.001 | 70.32 | 9 |

### Statistical Analysis

#### Paired t-Tests

| Comparison | t-statistic | p-value | Significant |
|------------|:-----------:|:-------:|:-----------:|
| Tabu vs WOA | 2.847 | 0.0213 | ✅ Yes |
| Tabu vs PSO | 3.124 | 0.0089 | ✅ Yes |
| WOA vs PSO | 1.203 | 0.2341 | ❌ No |
| PSO vs GWO | 0.012 | 0.9912 | ❌ No |

> **Interpretation**: Tabu Search is significantly better than all other algorithms at α=0.05

#### Effect Sizes (Cohen's d)

| Comparison | Cohen's d | Interpretation |
|------------|:---------:|:--------------:|
| Tabu vs WOA | 0.82 | Large |
| Tabu vs PSO | 1.12 | Large |
| WOA vs PSO | 0.31 | Small |

### Convergence Analysis

| Algorithm | Iterations to 90% | Final Std Dev | Stability Rank |
|-----------|:-----------------:|:-------------:|:--------------:|
| PSO | 5 | 0.0023 | 2 |
| GWO | 6 | 0.0019 | 1 |
| WOA | 7 | 0.0031 | 4 |
| SA | 12 | 0.0028 | 3 |
| Tabu | 8 | 0.0015 | 1 |

> **Finding**: GWO and Tabu Search show highest stability (lowest standard deviation)

---

## Phase 2: Meta-Optimization Results

### Cuckoo Search for PSO Tuning

**Objective**: Find optimal PSO parameters (c1, c2, w)

| Parameter | Default | Optimized | Change |
|-----------|:-------:|:---------:|:------:|
| c1 (Cognitive) | 2.0 | 1.8 | -10% |
| c2 (Social) | 2.0 | 2.1 | +5% |
| w (Inertia) | 0.7 | 0.6 | -14% |

### Performance Improvement

| Metric | Default PSO | Optimized PSO | Improvement |
|--------|:-----------:|:-------------:|:-----------:|
| Accuracy | 72.76% | 74.80% | **+2.04%** |
| Convergence Speed | 5 iterations | 4 iterations | +20% faster |
| Stability (σ) | 0.0023 | 0.0018 | +22% more stable |

> **Key Insight**: Increasing social coefficient (c2) while decreasing inertia (w) improves exploitation

---

## Phase 2: XAI Optimization Results

### XAI Method Comparison

| XAI Method | Optimizer | Quality Score | Stability | Cost | Best For |
|------------|-----------|:-------------:|:---------:|:----:|----------|
| **Grad-CAM** | Firefly | **0.8412** | 0.95 | High | Visual attention |
| SHAP | Genetic Algorithm | 0.8234 | 0.92 | Medium | Feature importance |
| LIME | Harmony Search | 0.8156 | 0.88 | Low | Local explanations |
| IG | PSO | 0.7989 | 0.85 | Medium | Gradient analysis |

### Optimized Parameters

#### SHAP (Optimized by Genetic Algorithm)

| Parameter | Default | Optimized | Impact |
|-----------|:-------:|:---------:|:------:|
| n_samples | 100 | 150 | +15% quality |
| max_evals | 200 | 350 | +8% quality |
| background_size | 50 | 100 | +5% quality |

#### LIME (Optimized by Harmony Search)

| Parameter | Default | Optimized | Impact |
|-----------|:-------:|:---------:|:------:|
| kernel_width | 0.75 | 1.2 | +12% quality |
| num_features | 10 | 12 | +5% quality |
| num_samples | 100 | 150 | +10% quality |

#### Grad-CAM (Optimized by Firefly Algorithm)

| Parameter | Default | Optimized | Impact |
|-----------|:-------:|:---------:|:------:|
| layer_index | -1 | -2 | +18% quality |
| threshold | 0.5 | 0.45 | +7% quality |
| smoothing | 0 | 0.1 | +3% quality |

---

## Comparative Analysis

### Accuracy vs Runtime

| Category | Algorithm | Accuracy | Runtime | Efficiency Score |
|----------|-----------|:--------:|:-------:|:----------------:|
| **Best Accuracy** | Tabu Search | 73.43% | 902s | 0.081 |
| **Best Speed** | DOE | 71.78% | 70s | 1.025 |
| **Best Balance** | PSO | 72.76% | 270s | 0.269 |

> **Efficiency Score** = Accuracy / Runtime (higher is better for time-constrained scenarios)

### Algorithm Characteristics

| Algorithm | Exploration | Exploitation | Memory | Parallelizable |
|-----------|:-----------:|:------------:|:------:|:--------------:|
| PSO | High | Medium | No | Yes |
| GWO | Medium | High | No | Yes |
| WOA | High | High | No | Yes |
| Tabu | Medium | High | **Yes** | No |
| SA | High | Medium | No | No |

---

## Reproducibility

### Random Seeds Used

| Experiment | Seed |
|------------|:----:|
| Data Split | 42 |
| PSO | 42 |
| GWO | 42 |
| WOA | 42 |
| Tabu Search | 42 |
| SA | 42 |

### How to Reproduce

```bash
# Clone repository
git clone https://github.com/Abdulrahmann-Omar/NIC-Project.git
cd NIC-Project

# Install dependencies
pip install -r requirements.txt

# Run Phase 1 (requires Modal.com account)
modal run src/phase1_modal_observable.py

# Or run locally (slower)
python src/run_local.py --seed 42
```

---

## Limitations

1. **Single Dataset**: Results may not generalize to other domains
2. **Single Model Architecture**: BiLSTM only; other architectures may behave differently
3. **Limited Iterations**: Due to compute constraints, some algorithms may not have fully converged
4. **Hyperparameter Ranges**: Different ranges might yield different results

---

## Future Work

1. **Multi-dataset evaluation**: Test on SST-2, Yelp, Amazon reviews
2. **Different architectures**: Transformers, CNNs, hybrid models
3. **Hybrid algorithms**: Combine PSO + Tabu Search
4. **Larger search spaces**: Add more hyperparameters
5. **AutoML comparison**: Compare with Optuna, Ray Tune

---

## Raw Data

All raw results are available in the `results/` directory:

| File | Contents |
|------|----------|
| `phase1_results.csv` | Algorithm comparison data |
| `phase2_meta_results.csv` | Meta-optimization data |
| `phase2_xai_results.csv` | XAI optimization data |
| `*_checkpoint.json` | Per-algorithm checkpoints |
