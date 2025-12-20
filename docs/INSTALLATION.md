# Installation & Setup Guide

Complete guide to set up the NIC Project locally.

---

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- (Optional) NVIDIA GPU with CUDA for faster training

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Abdulrahmann-Omar/NIC-Project.git
cd NIC-Project
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Additional Dependencies

For full functionality:

```bash
# Core ML
pip install tensorflow numpy pandas scikit-learn

# Visualization
pip install plotly matplotlib

# XAI
pip install shap lime

# Dashboard
pip install streamlit

# 3D Visualizations
pip install plotly
```

---

## Running the Dashboard

```bash
cd dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Running Phase 1 (Modal.com)

### Prerequisites
1. Create account at [modal.com](https://modal.com)
2. Install Modal: `pip install modal`
3. Set up authentication: `modal setup`

### Run
```bash
modal run src/phase1_modal_observable.py
```

---

## Running Phase 2 (Google Colab)

1. Open `notebooks/Phase2_Colab_Main.ipynb`
2. Upload to Google Colab
3. Enable GPU: Runtime > Change runtime type > T4 GPU
4. Run all cells

---

## Running 3D Visualizations

### Generate HTML Files
```bash
cd visualizations
python algorithm_3d.py
```

### View in Browser
Open the generated HTML files:
- `pso_3d_animation.html`
- `gwo_3d_animation.html`
- `search_space_comparison.html`

---

## Project Structure

```
NIC-Project/
├── src/                    # Phase 1 Modal code
│   ├── phase1_modal_observable.py
│   └── phase2_modal.py
├── notebooks/              # Jupyter notebooks
│   └── Phase2_Colab_Main.ipynb
├── dashboard/              # Streamlit app
│   ├── app.py
│   └── assets/
├── visualizations/         # 3D visualizations
│   ├── algorithm_3d.py
│   └── *.html
├── results/                # Output data
│   ├── phase1_results.csv
│   └── phase2_*.csv
├── docs/                   # Documentation
│   ├── ALGORITHMS.md
│   ├── INSTALLATION.md
│   └── blog_post.md
└── book/                   # Jupyter Book
    └── *.md
```

---

## Troubleshooting

### TensorFlow GPU Issues
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Modal Authentication
```bash
modal setup
# Follow browser prompts to authenticate
```

### Streamlit Not Loading
```bash
# Clear cache
streamlit cache clear
# Restart
streamlit run app.py
```

---

## Environment Variables

For Modal.com Kaggle integration:

```bash
modal secret create kaggle-secret \
  KAGGLE_USERNAME=your_username \
  KAGGLE_KEY=your_api_key
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Abdulrahmann-Omar/NIC-Project/issues)
- **Documentation**: [Algorithm Reference](ALGORITHMS.md)
- **Live Demo**: [Streamlit App](https://nic-project-2abdu.streamlit.app)
