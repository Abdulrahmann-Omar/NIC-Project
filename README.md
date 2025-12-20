#  Nature-Inspired Computation for Deep Learning Optimization



[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Modal](https://img.shields.io/badge/Phase%201-Modal%20H100-purple.svg)](https://modal.com/)
[![Colab](https://img.shields.io/badge/Phase%202-Google%20Colab-yellow.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A comprehensive Nature-Inspired Computation project combining metaheuristic optimization with Explainable AI for BiLSTM sentiment analysis**

---

##  Project Overview

This project implements **11 unique metaheuristic algorithms** across two phases to optimize a BiLSTM neural network for sentiment classification, with full Explainable AI (XAI) integration.

| Component | Description |
|-----------|-------------|
| **Dataset** | IMDB Movie Reviews (25,000+ samples) |
| **Model** | Bidirectional LSTM |
| **Phase 1** | Hyperparameter Optimization (Modal.com H100 GPU) |
| **Phase 2** | Meta-Optimization + XAI (Google Colab T4 GPU) |
| **Dashboard** | Interactive Streamlit visualization |

---

## ️ Project Structure

```
NIC-Project/
├──  src/                              # Phase 1: Modal.com Code
│   ├── phase1_modal_observable.py       # Main optimization script
│   ├── phase2_modal.py                  # Alternative Phase 2 for Modal
│   └── __init__.py
│
├──  notebooks/                        # Phase 2: Colab Notebooks
│   ├── Phase2_Colab_Main.ipynb          # ⭐ Main Phase 2 notebook
│   ├── phase2_colab.py                  # Standalone Python version
│   ├── data_converter.py                # JSON to CSV converter
│   └── best_bilstm_model.keras          # Trained model
│
├──  dashboard/                        # Streamlit Dashboard
│   ├── app.py                           # Main dashboard app
│   ├── generate_visualizations.py       # Image generator
│   └── assets/                          # Generated visualizations
│       ├── architecture_diagram.png
│       ├── bonus_xai_dashboard.png
│       └── ...
│
├──  results/                          # Outputs & Checkpoints
│   ├── phase1_results.json              # Phase 1 algorithm results
│   ├── phase2_full_results.json         # Phase 2 XAI results
│   ├── *_checkpoint.json                # Algorithm checkpoints
│   └── *.csv                            # Result summaries
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

##  Algorithms Used (11 Total)

### Phase 1: Model Hyperparameter Optimization (6 Algorithms)
*Run on Modal.com H100 GPU*

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **DOE (Taguchi)** | Systematic | Orthogonal array exploration |
| **PSO** | Swarm | Particle velocity updates |
| **Tabu Search** | Memory-based | Tabu list avoidance |
| **GWO** | Swarm | Wolf pack hierarchy (α, β, δ) |
| **WOA** | Swarm | Whale bubble-net hunting |
| **SA** | Single-solution | Temperature annealing |

### Phase 1: Feature Selection (1 Algorithm)
| Algorithm | Purpose |
|-----------|---------|
| **Ant Colony Optimization** | Feature subset selection |

### Phase 2: Meta-Optimization (1 Algorithm)
*Optimizes the optimizer parameters*

| Algorithm | Targets | Parameters Optimized |
|-----------|---------|---------------------|
| **Cuckoo Search** | PSO & GWO | c1, c2, w, a_decay |

### Phase 2: XAI Optimization (4 Algorithms)
*Optimizes explainability methods*

| Algorithm | XAI Method | Parameters |
|-----------|------------|------------|
| **Genetic Algorithm** | SHAP | n_samples, max_evals |
| **Harmony Search** | LIME | kernel_width, num_features |
| **Firefly Algorithm** | Grad-CAM | layer_index, threshold |
| **Bat Algorithm** | Stability | perturbation_std |

---

##  Quick Start

### Prerequisites
- Python 3.8+
- Modal.com account (Phase 1)
- Google account for Colab (Phase 2)

### Installation

```bash
git clone https://github.com/Abdulrahmann-Omar/NIC-Project.git
cd NIC-Project
pip install -r requirements.txt
```

---

##  Phase 1: Modal.com (H100 GPU)

### Setup Modal
```bash
pip install modal
modal setup
modal secret create kaggle-secret KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
```

### Run Phase 1
```bash
modal run src/phase1_modal_observable.py
```

**Expected Runtime**: ~15-20 minutes  
**Output**: `results/phase1_results.json`

### Phase 1 Results
| Algorithm | Best Accuracy | LSTM Units | Dropout | Learning Rate |
|-----------|--------------|------------|---------|---------------|
| DOE | 71.78% | 128 | 0.35 | 0.001 |
| PSO | 72.76% | 64 | 0.28 | 0.0053 |
| TabuSearch | **73.43%** | 128 | 0.45 | 0.001 |
| GWO | 72.76% | 64 | 0.31 | 0.0052 |
| WOA | 72.94% | 32 | 0.20 | 0.001 |
| SA | 72.72% | 64 | 0.35 | 0.005 |

---

##  Phase 2: Google Colab (T4 GPU)

### Run Phase 2
1. Open [`notebooks/Phase2_Colab_Main.ipynb`](notebooks/Phase2_Colab_Main.ipynb) in Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells

**Expected Runtime**: ~30-45 minutes  
**Output**: `results/phase2_full_results.json`, trained model

### Phase 2 Features
-  Cuckoo Search meta-optimization
-  SHAP feature importance (GA optimized)
-  LIME local explanations (Harmony Search optimized)
-  Grad-CAM attention maps (Firefly optimized)
-  Explanation stability analysis (Bat optimized)

---

##  Dashboard

### Run Locally
```bash
cd dashboard
python generate_visualizations.py  # Generate images first
streamlit run app.py
```

Open: http://localhost:8501

### Dashboard Features
-  Project overview & architecture
-  Live sentiment prediction
-  Algorithm comparison charts
-  XAI visualizations (SHAP, LIME, Grad-CAM)
-  Convergence analysis
-  Statistical significance tests

---

##  Results Summary

| Metric | Value |
|--------|-------|
| **Best Phase 1 Accuracy** | 73.43% (Tabu Search) |
| **Best Phase 2 Accuracy** | 75.23% (Cuckoo Search optimized) |
| **Total Algorithms** | 11 unique |
| **XAI Methods** | 4 (SHAP, LIME, Grad-CAM, Stability) |

---

##  Key Files

| File | Purpose | Run With |
|------|---------|----------|
| `src/phase1_modal_observable.py` | Phase 1 optimization | `modal run` |
| `notebooks/Phase2_Colab_Main.ipynb` | Phase 2 notebook | Google Colab |
| `dashboard/app.py` | Streamlit dashboard | `streamlit run` |
| `results/phase1_results.json` | Phase 1 outputs | - |

---

## ️ Requirements

```
tensorflow>=2.10
numpy
pandas
matplotlib
scikit-learn
shap
lime
tqdm
streamlit
plotly
```

---

##  License

MIT License - see [LICENSE](LICENSE)

---

##  Author

**Abdulrahman Omar**

---

<div align="center">

**⭐ Star this repo if it helped you!**

</div>
