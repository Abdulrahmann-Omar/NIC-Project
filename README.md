<div align="center">

# 🌿 Nature-Inspired Computation for Deep Learning

### *Optimizing Neural Networks with 11 Metaheuristic Algorithms*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://nic-project-2abdu.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![Modal](https://img.shields.io/badge/Phase_1-Modal_H100_GPU-purple?style=flat-square)](https://modal.com)
[![Colab](https://img.shields.io/badge/Phase_2-Google_Colab_T4-orange?style=flat-square)](https://colab.research.google.com)
[![Medium](https://img.shields.io/badge/Article-Medium-black?style=flat-square&logo=medium)](https://medium.com/@abdu.omar.muhammad/introduction-the-hyperparameter-tuning-nightmare-e9f41d69b5ed)

---

**[🚀 Live Demo](https://nic-project-2abdu.streamlit.app)** · **[📖 Documentation](docs/)** · **[📊 Results](#-results)** · **[🤝 Contributing](CONTRIBUTING.md)**

</div>


[Presentation](https://abdulrahmann-omar.github.io/NIC-Project/presentation/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Algorithms](#-algorithms-implemented)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Visualizations](#-visualizations)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

> **The Problem**: Grid search for 4 hyperparameters = **208 hours** of compute time.  
> **Our Solution**: Nature-inspired algorithms find optimal parameters in **15 minutes** with **73.4% accuracy**.

This project implements **11 unique metaheuristic algorithms** to optimize a BiLSTM neural network for sentiment analysis. Instead of brute-force hyperparameter tuning, we let nature do the optimization — mimicking bird flocks, wolf packs, whale hunting, and more.

### Why Nature-Inspired Algorithms?

| Approach | Time for 4 Parameters | Accuracy | Intelligence |
|----------|----------------------|----------|--------------|
| Grid Search | 208 hours | Exhaustive | None |
| Random Search | 20+ hours | ~72% | Random |
| **PSO (Ours)** | **4.5 minutes** | **72.76%** | Swarm |
| **Tabu Search (Ours)** | **15 minutes** | **73.43%** | Memory |

### Key Innovations

1. **Meta-Optimization**: Using Cuckoo Search to optimize PSO's own parameters (+2.04% improvement)
2. **XAI Integration**: SHAP, LIME, and Grad-CAM optimized by metaheuristics
3. **Comprehensive Benchmarking**: 11 algorithms compared on same task
4. **Interactive 3D Visualizations**: Watch algorithms explore search spaces in real-time

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **11 Algorithms** | PSO, GWO, WOA, SA, Tabu Search, GA, Cuckoo, Firefly, Bat, Harmony, DE |
| 🔄 **Meta-Optimization** | Optimize the optimizer's parameters automatically |
| 🔍 **Explainable AI** | SHAP, LIME, Grad-CAM with optimized parameters |
| 📊 **Interactive Dashboard** | Streamlit app with real-time predictions |
| 🎬 **3D Visualizations** | Animated PSO particles and GWO wolf packs |
| ⚡ **Cloud-Ready** | Modal.com H100 GPU + Google Colab T4 support |
| 📈 **Comprehensive Results** | Statistical significance testing included |
| 📚 **Full Documentation** | Tutorials, API reference, algorithm guides |

---

## 🖥️ Live Demo

<div align="center">

### 👉 **[Launch Interactive Demo](https://nic-project-2abdu.streamlit.app)** 👈

*Try sentiment analysis with optimized models and explore algorithm comparisons*

</div>

**Demo Features:**
- 🎤 Real-time sentiment prediction
- 📊 Algorithm performance comparison
- 🔬 XAI explanations (SHAP, LIME, Grad-CAM)
- 🎬 3D algorithm visualizations
- 📈 Convergence animations

---

## 📊 Results

### Phase 1: Algorithm Comparison

| Rank | Algorithm | Accuracy | LSTM Units | Dropout | Learning Rate | Runtime |
|:----:|-----------|:--------:|:----------:|:-------:|:-------------:|:-------:|
| 🥇 | **Tabu Search** | **73.43%** | 128 | 0.45 | 0.001 | 902s |
| 🥈 | WOA | 72.94% | 32 | 0.20 | 0.001 | 319s |
| 🥉 | PSO | 72.76% | 64 | 0.28 | 0.0053 | 270s |
| 4 | GWO | 72.76% | 64 | 0.31 | 0.0052 | 323s |
| 5 | SA | 72.72% | 64 | 0.35 | 0.005 | 353s |
| 6 | DOE | 71.78% | 128 | 0.35 | 0.001 | 70s |

### Phase 2: Meta-Optimization & XAI

| XAI Method | Optimizer | Quality Score | Stability |
|------------|-----------|:-------------:|:---------:|
| Grad-CAM | Firefly Algorithm | 0.8412 | 0.95 |
| SHAP | Genetic Algorithm | 0.8234 | 0.92 |
| LIME | Harmony Search | 0.8156 | 0.88 |
| Integrated Gradients | PSO | 0.7989 | 0.85 |

> **Key Finding**: Tabu Search's memory mechanism prevents cycling, leading to better exploration of the search space.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- (Optional) NVIDIA GPU with CUDA

### Installation

```bash
# Clone the repository
git clone https://github.com/Abdulrahmann-Omar/NIC-Project.git
cd NIC-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
cd dashboard
streamlit run app.py
```

### Run Phase 1 Optimization (Modal.com)

```bash
pip install modal
modal setup
modal run src/phase1_modal_observable.py
```

### Run Phase 2 in Google Colab

1. Open `notebooks/Phase2_Colab_Main.ipynb`
2. Upload to Google Colab
3. Enable GPU: Runtime → Change runtime type → T4 GPU
4. Run all cells

---

## 🧬 Algorithms Implemented

<details>
<summary><b>Phase 1: Model Optimization (7 Algorithms)</b></summary>

| Algorithm | Inspiration | Inventor | Year | Key Idea |
|-----------|-------------|----------|:----:|----------|
| **PSO** | Bird Flocking | Kennedy & Eberhart | 1995 | Particles share best positions |
| **GWO** | Wolf Pack Hunting | Mirjalili et al. | 2014 | Alpha-Beta-Delta hierarchy |
| **WOA** | Whale Bubble-Net | Mirjalili & Lewis | 2016 | Spiral bubble-net attack |
| **SA** | Metal Annealing | Kirkpatrick et al. | 1983 | Accept worse solutions sometimes |
| **Tabu Search** | Human Memory | Glover | 1986 | Avoid recently visited solutions |
| **ACO** | Ant Colonies | Dorigo | 1992 | Pheromone trail following |
| **DOE** | Experimental Design | Taguchi | 1950s | Orthogonal array exploration |

</details>

<details>
<summary><b>Phase 2: Meta-Optimization & XAI (4 Algorithms)</b></summary>

| Algorithm | Inspiration | Inventor | Year | Used For |
|-----------|-------------|----------|:----:|----------|
| **Cuckoo Search** | Brood Parasitism | Yang & Deb | 2009 | PSO parameter tuning |
| **Genetic Algorithm** | Natural Evolution | Holland | 1975 | SHAP optimization |
| **Harmony Search** | Music Improvisation | Geem et al. | 2001 | LIME optimization |
| **Firefly Algorithm** | Firefly Flashing | Yang | 2008 | Grad-CAM optimization |

</details>

---

## 📁 Project Structure

```
NIC-Project/
├── 📂 src/                         # Phase 1: Modal.com Code
│   ├── phase1_modal_observable.py    # Main optimization script
│   └── phase2_modal.py               # Alternative Phase 2
│
├── 📂 notebooks/                   # Phase 2: Colab Notebooks
│   ├── Phase2_Colab_Main.ipynb       # ⭐ Main notebook
│   └── best_bilstm_model.keras       # Trained model
│
├── 📂 dashboard/                   # Streamlit Dashboard
│   ├── app.py                        # Main app
│   └── assets/                       # Visualizations
│
├── 📂 visualizations/              # 3D Animations
│   ├── algorithm_3d.py               # PSO/GWO visualizations
│   └── *.html                        # Interactive plots
│
├── 📂 results/                     # Output Data
│   ├── phase1_results.csv            # Algorithm comparison
│   └── phase2_xai_results.csv        # XAI results
│
├── 📂 docs/                        # Documentation
│   ├── ALGORITHMS.md                 # Algorithm reference
│   ├── INSTALLATION.md               # Setup guide
│   ├── API.md                        # API reference
│   └── blog_post.md                  # Technical article
│
├── 📂 book/                        # Jupyter Book
│   ├── _config.yml                   # Book configuration
│   ├── _toc.yml                      # Table of contents
│   └── *.md                          # Chapters
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| 📖 [Algorithm Reference](docs/ALGORITHMS.md) | Complete guide to all 11 algorithms |
| 🔧 [Installation Guide](docs/INSTALLATION.md) | Setup instructions and troubleshooting |
| 🔌 [API Reference](docs/API.md) | Technical API documentation |
| 📝 [Medium Article](https://medium.com/@abdu.omar.muhammad/introduction-the-hyperparameter-tuning-nightmare-e9f41d69b5ed) | In-depth technical blog post |
| 📚 [Interactive Book](book/) | Jupyter Book with tutorials |
| 📋 [Changelog](CHANGELOG.md) | Version history |
| 🤝 [Contributing](CONTRIBUTING.md) | How to contribute |

---

## 🎨 Visualizations

### 3D Algorithm Animations

<table>
<tr>
<td align="center">
<b>PSO: Particle Swarm</b><br>
Watch particles converge to optimum
</td>
<td align="center">
<b>GWO: Wolf Pack</b><br>
Alpha-Beta-Delta hunting
</td>
<td align="center">
<b>Search Space</b><br>
Compare test functions
</td>
</tr>
</table>

**View animations:** Open `visualizations/*.html` in your browser or visit the [Live Demo](https://nic-project-2abdu.streamlit.app).

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{omar2024nic,
  author = {Omar, Abdulrahman},
  title = {Nature-Inspired Computation for Deep Learning Optimization},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Abdulrahmann-Omar/NIC-Project}
}
```

**APA Format:**
> Omar, A. (2024). *Nature-Inspired Computation for Deep Learning Optimization*. GitHub. https://github.com/Abdulrahmann-Omar/NIC-Project

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

<div align="center">

**Abdulrahman Omar**

[![GitHub](https://img.shields.io/badge/GitHub-@Abdulrahmann--Omar-181717?style=for-the-badge&logo=github)](https://github.com/Abdulrahmann-Omar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abdulrahmann--omar-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/abdulrahmann-omar)
[![Medium](https://img.shields.io/badge/Medium-@abdu.omar.muhammad-000000?style=for-the-badge&logo=medium)](https://medium.com/@abdu.omar.muhammad)

---

### ⭐ Star this repo if it helped you!

**Found a bug?** [Open an issue](https://github.com/Abdulrahmann-Omar/NIC-Project/issues)  
**Have a question?** [Start a discussion](https://github.com/Abdulrahmann-Omar/NIC-Project/discussions)

</div>
