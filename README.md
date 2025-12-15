# Nature-Inspired Computation for BiLSTM Optimization

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Modal](https://img.shields.io/badge/modal-H100_GPU-orange.svg)](https://modal.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **BiLSTM hyperparameter optimization using metaheuristic algorithms deployed on Modal.com H100 GPU**

## 📖 Overview

This project implements a comprehensive comparison of **7 metaheuristic algorithms** for optimizing BiLSTM neural network hyperparameters on the Sentiment140 dataset. The implementation features state-of-the-art observability, automatic checkpointing, and cloud GPU deployment on Modal.com.

### Key Features

- 🚀 **H100 GPU Acceleration** - Leveraging Modal.com's cutting-edge GPU infrastructure
- 🧬 **7 Metaheuristic Algorithms** - DOE, PSO, Tabu Search, GWO, WOA, DE, SA
- 💾 **Automatic Checkpointing** - Resume training from failures seamlessly
- 📊 **Comprehensive Observability** - Real-time progress tracking with detailed metrics
- 📈 **Publication-Ready Visualizations** - Automated comparison charts and heatmaps
- 🔄 **Two-Phase Approach** - Hyperparameter optimization + Meta-optimization + XAI

---

## 🎯 Problem Statement

Optimizing deep learning models like BiLSTM for NLP tasks requires extensive hyperparameter tuning. Traditional grid search is computationally expensive. This project demonstrates how **nature-inspired metaheuristic algorithms** can efficiently explore the hyperparameter space, achieving superior performance with significantly less compute time.

**Dataset**: Sentiment140 (1.6M tweets)  
**Task**: Binary sentiment classification  
**Model**: Bidirectional LSTM with embedding layer  
**Optimization Target**: LSTM units, dropout rate, learning rate

---

## 🏗️ Project Structure

```
nature-inspired-computation/
├── 📁 src/                          # Source code
│   ├── phase1_modal.py              # Phase 1: Hyperparameter optimization
│   ├── phase1_modal_observable.py   # Phase 1 with detailed observability
│   ├── phase2_modal.py              # Phase 2: Meta-optimization + XAI
│   └── utils.py                     # Helper functions (if needed)
│
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── Phase-1.ipynb                # Phase 1 development notebook
│   └── Phase-2.ipynb                # Phase 2 development notebook
│
├── 📁 docs/                         # Documentation
│   ├── README_PHASE1.md             # Phase 1 detailed guide
│   ├── visualization_guide.md       # Visualization interpretation
│   ├── observability_guide.md       # Observability features
│   └── production_guide.md          # Production deployment guide
│
├── 📁 results/                      # Results and outputs (gitignored)
│   ├── phase1_results.csv
│   ├── phase1_comparison.png
│   └── checkpoints/
│
├── 📁 config/                       # Configuration files
│   └── modal_config.py              # Modal.com configuration
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                    # Git ignore rules
├── 📄 LICENSE                       # MIT License
└── 📄 README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Modal.com account (with $30 free credit)
- Kaggle API credentials

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nature-inspired-computation.git
   cd nature-inspired-computation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Modal.com**
   ```bash
   modal setup
   ```

4. **Configure Kaggle credentials**
   ```bash
   modal secret create kaggle-secret \
       KAGGLE_USERNAME=your_username \
       KAGGLE_KEY=your_api_key
   ```

### Running Phase 1

```bash
# Standard version
modal run src/phase1_modal.py

# Observable version (detailed logs + progress bars)
modal run src/phase1_modal_observable.py
```

Expected runtime: **10-15 minutes** on H100 GPU

### Running Phase 2

```bash
modal run src/phase2_modal.py
```

---

## 🧬 Algorithms Implemented

### Phase 1: Hyperparameter Optimization

| Algorithm | Type | Population | Iterations | Key Feature |
|-----------|------|------------|------------|-------------|
| **DOE (Taguchi L2)** | Systematic | 2 | 1 | Orthogonal array exploration |
| **PSO** | Swarm | 5 | 3 | Particle velocity updates |
| **Tabu Search** | Memory-based | 1 | 8 | Tabu list (tenure=5) |
| **GWO** | Swarm | 5 | 3 | Pack hierarchy (α, β, δ) |
| **WOA** | Swarm | 5 | 3 | Bubble-net feeding strategy |
| **DE** | Evolutionary | 5 | 3 | Mutation (F=0.8) + Crossover (CR=0.9) |
| **SA** | Single-solution | 1 | 24 | Temperature annealing (α=0.9) |

### Phase 2: Meta-Optimization + XAI

- **Cuckoo Search**: Optimize algorithm hyperparameters (population size, iterations)
- **XAI Optimization**: 4 algorithms for feature importance (GA, Harmony Search, Firefly, Bat)

---

## 📊 Results

### Sample Output

```
==================================================
FINAL RESULTS
==================================================
Algorithm      Best_Accuracy  LSTM  Dropout    LR      Time(s)
DOE-Taguchi    0.7601        128   0.350   0.001000    320
PSO            0.7823        128   0.285   0.003421    280
Tabu_Search    0.7791        64    0.312   0.004123    310
GWO            0.7767        128   0.298   0.002987    265
WOA            0.7754        64    0.308   0.003876    290
DE             0.7809        128   0.291   0.003654    275
SA             0.7732        64    0.324   0.004521    305
```

**Best Algorithm**: PSO with **78.23% accuracy**

### Visualizations

The system automatically generates a comprehensive 6-panel comparison figure:

![Visualization Example](docs/assets/example_visualization.png)

1. **Accuracy Bar Chart** - Direct performance comparison
2. **Execution Time** - Computational cost analysis
3. **Accuracy vs Time Scatter** - Pareto-optimal identification
4. **LSTM Units Distribution** - Architecture consensus
5. **Hyperparameter Heatmap** - Parameter correlation analysis
6. **Summary Statistics** - Key metrics at a glance

---

## 💾 Checkpoint System

The project features automatic checkpoint saving to Modal Volumes:

```
/checkpoints/ (persists across runs)
├── preprocessed_data.npz        # Cached dataset (15MB)
├── DOE_checkpoint.json          # DOE results + timestamp
├── PSO_checkpoint.json          # PSO results + timestamp
└── phase1_results.json          # Cumulative results
```

**Resume from Failure**:
```bash
# Just re-run - automatically resumes from last checkpoint
modal run src/phase1_modal.py
```

---

## 🔬 Observability Features

### Terminal Output
- 🎨 Emoji indicators for each algorithm (🐝 PSO, 🐺 GWO, 🐋 WOA, etc.)
- 📊 tqdm progress bars with real-time metrics
- ⏱️ Timing information for each experiment
- 🎯 Best score tracking per iteration

### Logging Levels
- `INFO`: Algorithm progress, checkpoints, GPU status
- `DEBUG`: Detailed iteration metrics (available in observable version)
- `ERROR`: Graceful error handling with stack traces

See [Observability Guide](docs/observability_guide.md) for details.

---

## 💰 Cost Estimation

### Modal.com H100 Pricing
- **Rate**: ~$3-5/hour
- **Runtime**: ~10-15 minutes for Phase 1
- **Estimated cost**: **$0.50-1.25 per run**

With $30 credit:
- **~24-60 full experiments** possible
- Use checkpoints to avoid re-running completed work

### Alternative Platforms
| Platform | GPU | Cost | Runtime |
|----------|-----|------|---------|
| **Modal.com** | H100 | $0.50-1.25 | 10-15 min |
| Google Colab Pro+ | A100 | $50/month | 30-45 min |
| AWS SageMaker | A100 | ~$5/hour | 20-30 min |
| GCP Vertex AI | A100 | ~$3-4/hour | 20-30 min |

---

## 🛠️ Development

### Code Style
```bash
# Format code
black src/

# Check style
flake8 src/
```

### Testing
```bash
# Run unit tests (if implemented)
pytest tests/

# Run specific test
pytest tests/test_algorithms.py -v
```

---

## 📚 Documentation

- **[Phase 1 Guide](docs/README_PHASE1.md)** - Detailed setup and troubleshooting
- **[Production Guide](docs/production_guide.md)** - Deployment best practices
- **[Visualization Guide](docs/visualization_guide.md)** - Chart interpretation
- **[Observability Guide](docs/observability_guide.md)** - Logging and monitoring

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Modal.com** for providing H100 GPU infrastructure
- **Kaggle** for the Sentiment140 dataset
- **TensorFlow/Keras** for deep learning framework
- Nature-inspired algorithms research community

---

## 📧 Contact

**Author**: Abdalrahman Mohammed  
**Email**: your.email@example.com  
**GitHub**: [@your-username](https://github.com/your-username)

---

## 📈 Roadmap

- [x] Phase 1: Hyperparameter optimization (7 algorithms)
- [x] Checkpoint system with Modal Volumes
- [x] Comprehensive observability
- [x] Publication-ready visualizations
- [ ] Phase 2: Meta-optimization + XAI
- [ ] Performance benchmarking report
- [ ] Algorithm parameter sensitivity analysis
- [ ] Additional datasets (Twitter, IMDB)
- [ ] Multi-GPU distributed training
- [ ] Web dashboard for real-time monitoring

---



<div align="center">

**If this project helped you, please consider giving it a ⭐!**

Made with ❤️ and ☕ by Abdulrahman Omar

</div>
