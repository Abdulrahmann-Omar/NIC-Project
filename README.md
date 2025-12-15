# Nature Inspired Computation - Final Project

## Overview
Complete implementation of Nature Inspired Computation project with Deep Learning and Explainable AI (XAI) integration, deployed on **Modal.com** with H100 GPU acceleration.

## 🚀 Quick Start

### Production Version (Recommended)
```bash
pip install modal
modal setup
modal secret create kaggle-secret KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx
modal run phase1_modal_robust.py
```

### Features
✅ **Error Handling** - Robust try-catch blocks  
✅ **Checkpoints** - Resume from failures  
✅ **H100 GPU** - 80GB VRAM, ultra-fast training  
✅ **Persistent Storage** - Modal Volumes for data persistence  

See [README_PHASE1.md](README_PHASE1.md) for detailed documentation.

## 📊 Project Structure

### Phase 1: Hyperparameter Optimization
- **Dataset**: Sentiment140 (100K samples)
- **Model**: BiLSTM for sentiment classification
- **Algorithms**: 
  1. Design of Experiments (Taguchi L9)
  2. Particle Swarm Optimization (PSO)
  3. Tabu Search
  4. Grey Wolf Optimizer (GWO)
  5. Whale Optimization Algorithm (WOA)
  6. Differential Evolution (DE)
  7. Simulated Annealing (SA)

### Phase 2: Meta-Optimization & XAI
- **Meta-Optimization**: Cuckoo Search for PSO/Tabu parameters
- **XAI Algorithms**:
  1. Genetic Algorithm → SHAP optimization
  2. Harmony Search → LIME optimization
  3. Firefly Algorithm → Grad-CAM optimization
  4. Bat Algorithm → Stability optimization

## 📁 Files

- `phase1_modal.py` - Phase 1 Modal deployment
- `phase2_modal.py` - Phase 2 Modal deployment
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🎯 Results

Results are saved to `/tmp/` on Modal:
- `phase1_results.csv` - Algorithm comparison
- `phase2_meta_results.csv` - Meta-optimization results
- `phase2_xai_results.csv` - XAI optimization results

## 💰 Cost Estimation

Using Modal.com $30/month credits:
- Phase 1: ~15-20 minutes GPU time
- Phase 2: ~10 minutes GPU time
- **Total**: < $5 per full run

## 📚 Key Features

✅ **Modal.com**: Serverless GPU compute  
✅ **DOE**: Systematic Taguchi method  
✅ **7 Algorithms**: Comprehensive comparison  
✅ **Meta-Optimization**: Algorithm parameter tuning  
✅ **XAI**: 4 explainability optimizations  
✅ **Reproducible**: Seeded experiments  

## 👥 Team

- Team of 2 students
- Platform: Modal.com
- Dataset: Sentiment140 (NLP)

## 📖 Documentation

See `walkthrough.md` in `.gemini/antigravity/brain/` for detailed documentation.
