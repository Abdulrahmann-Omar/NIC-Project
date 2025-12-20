# 🚀 Deployment Guide

Deploy the NIC Project to various platforms.

---

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud](#streamlit-cloud)
3. [Docker](#docker)
4. [Google Colab](#google-colab)
5. [Modal.com](#modalcom)
6. [GitHub Pages](#github-pages)
7. [Heroku](#heroku)

---

## Local Deployment

### Quick Start

```bash
# Clone and setup
git clone https://github.com/Abdulrahmann-Omar/NIC-Project.git
cd NIC-Project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run dashboard
cd dashboard
streamlit run app.py
```

### Access

Open http://localhost:8501 in your browser.

### Custom Port

```bash
streamlit run app.py --server.port 8080
```

### Production Mode

```bash
streamlit run app.py --server.headless true --server.enableCORS false
```

---

## Streamlit Cloud

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Step 1: Prepare Repository

Ensure these files exist:
- `dashboard/app.py` - Main application
- `dashboard/requirements.txt` - Dependencies

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select repository: `Abdulrahmann-Omar/NIC-Project`
5. Set main file path: `dashboard/app.py`
6. Click "Deploy"

### Step 3: Configure

In Streamlit Cloud settings:

**Secrets** (if needed):
```toml
[kaggle]
username = "your_username"
key = "your_api_key"
```

**Custom domain** (optional):
- Go to Settings → Sharing
- Add custom domain

### Live Demo

🔗 [https://nic-project-2abdu.streamlit.app](https://nic-project-2abdu.streamlit.app)

---

## Docker

### Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY dashboard/ ./dashboard/
COPY results/ ./results/
COPY visualizations/ ./visualizations/

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t nic-project .

# Run container
docker run -p 8501:8501 nic-project
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./results:/app/results
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
```

```bash
docker-compose up -d
```

---

## Google Colab

### Run Phase 2 Notebook

1. Open [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub
3. Enter: `https://github.com/Abdulrahmann-Omar/NIC-Project`
4. Select `notebooks/Phase2_Colab_Main.ipynb`
5. Runtime → Change runtime type → T4 GPU
6. Run all cells

### One-Click Open

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abdulrahmann-Omar/NIC-Project/blob/master/notebooks/Phase2_Colab_Main.ipynb)

### Install Dependencies in Colab

```python
!pip install -q tensorflow numpy pandas matplotlib plotly shap lime
```

---

## Modal.com

### Setup

```bash
pip install modal
modal setup  # Authenticate via browser
```

### Deploy Phase 1

```bash
modal run src/phase1_modal_observable.py
```

### Persistent Deployment

```bash
modal deploy src/phase1_modal_observable.py
```

### Configure Secrets

```bash
# Kaggle credentials (for dataset access)
modal secret create kaggle-secret \
  KAGGLE_USERNAME=your_username \
  KAGGLE_KEY=your_api_key
```

---

## GitHub Pages

### For Documentation Site

#### Step 1: Create MkDocs Config

Create `mkdocs.yml`:

```yaml
site_name: NIC Project Documentation
site_url: https://abdulrahmann-omar.github.io/NIC-Project/
repo_url: https://github.com/Abdulrahmann-Omar/NIC-Project

theme:
  name: material
  palette:
    primary: deep purple
    accent: purple
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
    - search.suggest

nav:
  - Home: index.md
  - Getting Started:
    - Installation: INSTALLATION.md
    - Quick Start: TUTORIAL.md
  - Algorithms: ALGORITHMS.md
  - API Reference: API.md
  - Results: RESULTS.md
  - About:
    - Contributing: ../CONTRIBUTING.md
    - Changelog: ../CHANGELOG.md
```

#### Step 2: Install MkDocs

```bash
pip install mkdocs mkdocs-material
```

#### Step 3: Build and Deploy

```bash
# Build site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### For Static Landing Page

Already created in `docs/` folder. Enable in GitHub:

1. Go to repository Settings
2. Pages → Source → Deploy from branch
3. Branch: `master`, Folder: `/docs`
4. Save

**Site URL**: https://abdulrahmann-omar.github.io/NIC-Project/

---

## Heroku

### Prerequisites

- Heroku CLI installed
- Heroku account

### Step 1: Create Heroku Files

**Procfile**:
```
web: sh setup.sh && streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh**:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**runtime.txt**:
```
python-3.10.12
```

### Step 2: Deploy

```bash
heroku login
heroku create nic-project
git push heroku master
heroku open
```

---

## Environment Variables

### Required for All Deployments

| Variable | Description | Example |
|----------|-------------|---------|
| `PYTHONPATH` | Module path | `/app` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `STREAMLIT_SERVER_PORT` | Dashboard port | 8501 |
| `KAGGLE_USERNAME` | Kaggle API username | - |
| `KAGGLE_KEY` | Kaggle API key | - |

---

## Troubleshooting

### Streamlit Cloud Issues

**"Module not found"**:
- Check `dashboard/requirements.txt` has all dependencies
- Reboot app in Streamlit Cloud dashboard

**"FileNotFoundError"**:
- Paths must be relative to `app.py`
- Use `pathlib.Path(__file__).parent` for reliable paths

### Docker Issues

**"Port already in use"**:
```bash
docker run -p 8502:8501 nic-project
```

**"Out of memory"**:
- Increase Docker memory limit in Docker Desktop settings

### Modal Issues

**"Authentication failed"**:
```bash
modal setup  # Re-authenticate
```

**"Timeout"**:
- Increase timeout in `@app.function(timeout=7200)`

---

## Health Checks

### Local Check

```bash
curl http://localhost:8501/_stcore/health
```

### Docker Check

Add to `docker-compose.yml`:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

---

## Monitoring

### Streamlit Cloud

- View logs in Streamlit Cloud dashboard
- "Manage app" → "Logs"

### Docker

```bash
docker logs -f container_name
```

### Heroku

```bash
heroku logs --tail --app nic-project
```
