# Changelog

All notable changes to the NIC Project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Hybrid algorithm implementations (PSO + Tabu Search)
- Multi-objective optimization support
- Transformer model optimization
- Optuna integration for comparison

---

## [2.1.0] - 2024-12-20

### Added
- 📚 **Comprehensive Documentation Suite**
  - Professional README.md with badges and visual appeal
  - [Tutorials](docs/TUTORIAL.md) with 7 step-by-step guides
  - [Algorithm Reference](docs/ALGORITHMS.md) for all 11 algorithms
  - [API Reference](docs/API.md) with function documentation
  - [Results Analysis](docs/RESULTS.md) with statistics
  - [Deployment Guide](docs/DEPLOYMENT.md) for multiple platforms
  - [FAQ](docs/FAQ.md) addressing common questions
  - [Installation Guide](docs/INSTALLATION.md) with troubleshooting

- 🎨 **Professional Project Website**
  - GitHub Pages landing page (`docs/index.html`)
  - Modern CSS styling (`docs/style.css`)
  - Responsive design for all devices

- 📝 **Technical Blog Post**
  - [Medium article](https://medium.com/@abdu.omar.muhammad/introduction-the-hyperparameter-tuning-nightmare-e9f41d69b5ed)
  - 3,500+ words with code examples
  - Original researcher credits

- 📖 **Interactive Jupyter Book**
  - 6 chapters covering all project aspects
  - Step-by-step explanations
  - Ready for GitHub Pages deployment

### Changed
- Updated README with collapsible sections
- Improved navigation in Streamlit dashboard
- Enhanced error handling for missing data

### Fixed
- Empty DataFrame error in Algorithm Comparison page
- Path resolution for deployed Streamlit app
- Git tracking of result files

---

## [2.0.0] - 2024-12-15

### Added
- 🎬 **3D Algorithm Visualizations**
  - PSO particle swarm animation
  - GWO wolf pack hunting visualization
  - Search space comparison (Sphere, Rastrigin, Rosenbrock)
  - Embedded in Streamlit dashboard

- 🔄 **Phase 2: Meta-Optimization**
  - Cuckoo Search for PSO parameter tuning
  - Achieved 2.04% improvement over default PSO
  - Optimized c1, c2, and w parameters

- 🔍 **XAI Integration**
  - SHAP with Genetic Algorithm optimization
  - LIME with Harmony Search optimization
  - Grad-CAM with Firefly Algorithm optimization
  - Integrated Gradients with PSO optimization

- 📊 **Interactive Dashboard**
  - Real-time sentiment prediction
  - Algorithm comparison charts
  - XAI visualization explorer
  - Convergence animations
  - 3D visualization page

### Changed
- Moved Phase 2 from Modal to Google Colab
- Reorganized project structure
- Updated model architecture documentation

---

## [1.1.0] - 2024-12-10

### Added
- ⏸️ **Checkpointing System**
  - Modal Volume for persistent storage
  - Resume capability after interruption
  - JSON checkpoint format

- 📈 **Enhanced Observability**
  - Rich terminal output with progress bars
  - Real-time status updates
  - Algorithm-specific logging
  - Convergence tracking

- 🐜 **Ant Colony Optimization**
  - Feature selection implementation
  - Pheromone trail mechanism
  - Configurable parameters

### Changed
- Improved error handling in Modal functions
- Better timeout management
- Enhanced logging format

### Fixed
- Memory leak in extended optimization runs
- Checkpoint file path issues on Windows

---

## [1.0.0] - 2024-12-01

### Added
- 🚀 **Initial Release**

- 🧬 **6 Metaheuristic Algorithms**
  - Particle Swarm Optimization (PSO)
  - Grey Wolf Optimizer (GWO)
  - Whale Optimization Algorithm (WOA)
  - Simulated Annealing (SA)
  - Tabu Search
  - Design of Experiments (DOE/Taguchi)

- 🧠 **BiLSTM Model**
  - Bidirectional LSTM for sentiment analysis
  - Configurable architecture
  - IMDB dataset integration

- ☁️ **Modal.com Integration**
  - H100 GPU support
  - Serverless execution
  - Kaggle dataset access

- 📊 **Results Tracking**
  - CSV and JSON output formats
  - Per-algorithm metrics
  - Convergence data

- 📁 **Project Structure**
  - Organized folder hierarchy
  - Modular code design
  - Clean separation of concerns

### Technical Details
- Python 3.8+ compatible
- TensorFlow 2.x backend
- NumPy, Pandas, Matplotlib dependencies

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 2.1.0 | 2024-12-20 | Documentation overhaul |
| 2.0.0 | 2024-12-15 | 3D visualizations, XAI, dashboard |
| 1.1.0 | 2024-12-10 | Checkpointing, ACO, observability |
| 1.0.0 | 2024-12-01 | Initial release with 6 algorithms |

---

## Migration Guides

### Upgrading from 1.x to 2.x

1. **New folder structure**:
   - `src/` now contains Modal code only
   - `notebooks/` for Colab notebooks
   - `dashboard/` for Streamlit app

2. **New dependencies**:
   ```bash
   pip install plotly shap lime
   ```

3. **Running dashboard**:
   ```bash
   cd dashboard
   streamlit run app.py
   ```

---

## Roadmap

### v2.2.0 (Planned)
- [ ] Hybrid algorithm implementations
- [ ] Additional test functions
- [ ] Performance optimizations

### v3.0.0 (Future)
- [ ] PyPI package release
- [ ] Multi-objective support
- [ ] Neural Architecture Search

---

[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
