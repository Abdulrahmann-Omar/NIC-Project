"""
═══════════════════════════════════════════════════════════════════════════════
NATURE INSPIRED COMPUTATION - PHASE 2
Advanced Optimization (Cuckoo Search) & Explainable AI (XAI)
H100 GPU Accelerated

HIGH-LEVEL ARCHITECTURE:
────────────────────────────────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL.COM CLOUD EXECUTION                         │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────────────────┐   │
│  │  H100 GPU     │→│  TensorFlow  │→│  Checkpoint System (Modal Volume) │   │
│  │  80GB VRAM    │  │  XLA + AMP   │  │  Persistent Storage             │   │
│  └───────────────┘  └──────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

OBJECTIVES:
1. Meta-Heuristic: Cuckoo Search (CS) for Hyperparameter Optimization
2. XAI Analysis: SHAP, LIME, and Saliency Maps on the Best Model
3. Observability: Structured Logging & Checkpointing (Matching Phase 1)
═══════════════════════════════════════════════════════════════════════════════
"""

import modal
import os
import json
import time
import logging
import math
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: MODAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

app = modal.App("nic-phase2-h100")

# Image with TensorFlow (CUDA), XAI libs, and standard tools
image = (
    modal.Image
    .from_registry("nvidia/cuda:12.2.0-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("unzip", "wget", "git")
    .pip_install(
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "tqdm",
        "matplotlib",
        "seaborn",
        "kaggle",
        "tensorflow[and-cuda]",
        "shap",
        "lime",
        "tf-keras-vis",
        "opencv-python-headless"
    )
)

volume = modal.Volume.from_name("nic-checkpoints", create_if_missing=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA STRUCTURES & LOGGING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AlgorithmResult:
    """Structured result from a single optimization algorithm."""
    algorithm_name: str
    best_accuracy: float
    best_lstm_units: int
    best_dropout: float
    best_learning_rate: float
    execution_time_seconds: float
    iterations_completed: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class HyperparameterConfig:
    """BiLSTM model hyperparameters."""
    lstm_units: int
    dropout_rate: float
    learning_rate: float
    
    def __repr__(self) -> str:
        return f"LSTM={self.lstm_units}, Drop={self.dropout_rate:.3f}, LR={self.learning_rate:.6f}"

def setup_logger(name: str = "NIC-Phase2") -> logging.Logger:
    """Configure structured logging with timestamps."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class CheckpointManager:
    """Handles persistent storage and retrieval of training state."""
    
    def __init__(self, checkpoint_dir: Path, logger: logging.Logger):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
    def save_checkpoint(self, algorithm_name: str, data: Dict[str, Any]) -> bool:
        try:
            checkpoint_file = self.checkpoint_dir / f"{algorithm_name}_checkpoint.json"
            full_data = {
                "algorithm": algorithm_name,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            temp_file = checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(full_data, f, indent=2)
            temp_file.replace(checkpoint_file)
            self.logger.info(f"✅ Checkpoint saved: {checkpoint_file.name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint for {algorithm_name}: {e}")
            return False
    
    def load_checkpoint(self, algorithm_name: str) -> Optional[Dict[str, Any]]:
        checkpoint_file = self.checkpoint_dir / f"{algorithm_name}_checkpoint.json"
        if not checkpoint_file.exists():
            return None
        try:
            with open(checkpoint_file, 'r') as f:
                full_data = json.load(f)
            self.logger.info(f"✅ Loaded checkpoint for {algorithm_name}")
            return full_data['data']
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint for {algorithm_name}: {e}")
            return None

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    secrets=[modal.Secret.from_name("kaggle-secret")],
    volumes={"/checkpoints": volume}
)
def run_phase2_training() -> Dict[str, Any]:
    """
    ═══════════════════════════════════════════════════════════════════════════
    MAIN PIPELINE - PHASE 2 (CS + XAI)
    ═══════════════════════════════════════════════════════════════════════════
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
    from sklearn.metrics import accuracy_score
    import shap
    import lime
    from lime import lime_tabular
    
    # Setup
    logger = setup_logger("NIC-Phase2")
    checkpoint_dir = Path("/checkpoints")
    checkpoint_mgr = CheckpointManager(checkpoint_dir, logger)
    
    logger.info("="*70)
    logger.info("PHASE 2: H100 OPTIMIZED TRAINING & XAI")
    logger.info("="*70)
    
    # Config GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"✅ H100 GPU Detected: {len(gpus)}")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("   └─ Mixed Precision (FP16) Enabled")
    else:
        logger.warning("⚠️ No GPU detected - Performance will be degraded")

    # ═══════════════════════════════════════════════════════════════════
    # 1. LOAD DATA (Reuse from Phase 1)
    # ═══════════════════════════════════════════════════════════════════
    
    preprocessed_file = checkpoint_dir / "preprocessed_data.npz"
    if not preprocessed_file.exists():
        logger.error("❌ Preprocessed data not found! Please run Phase 1 first.")
        # Fallback: We could re-run connection here, but for now we assume Phase 1 is done
        return {"error": "Missing preprocessed data"}
        
    logger.info("Loading cached data from Phase 1...")
    data = np.load(preprocessed_file, allow_pickle=True)
    X_train_seq = data['X_train_seq']
    X_val_seq = data['X_val_seq']
    X_test_seq = data['X_test_seq']
    y_train_arr = data['y_train_arr']
    y_val_arr = data['y_val_arr']
    y_test_arr = data['y_test_arr']
    logger.info(f"✅ Data Loaded. Train: {len(X_train_seq)} samples")

    # Constants
    MAX_WORDS = 5000
    MAX_LEN = 50

    # ═══════════════════════════════════════════════════════════════════
    # 2. MODEL DEFINITION
    # ═══════════════════════════════════════════════════════════════════

    def build_bilstm(config: HyperparameterConfig) -> tf.keras.Model:
        model = Sequential([
            Embedding(MAX_WORDS, 64, input_length=MAX_LEN, name="embedding"),
            Bidirectional(LSTM(config.lstm_units, return_sequences=False), name="bilstm"),
            Dropout(config.dropout_rate, name="dropout1"),
            Dense(32, activation='relu', name="dense1"),
            Dropout(config.dropout_rate / 2, name="dropout2"),
            Dense(1, activation='sigmoid', name="output")
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            metrics=['accuracy'] # 'accuracy' is supported in mixed_precision
        )
        return model

    def fitness_function(config: HyperparameterConfig, epochs=2) -> float:
        try:
            tf.keras.backend.clear_session()
            model = build_bilstm(config)
            # Use smaller batch size for better generalization or larger for speed on H100
            # H100 can handle large batches.
            model.fit(
                X_train_seq, y_train_arr,
                validation_data=(X_val_seq, y_val_arr),
                epochs=epochs,
                batch_size=256, # Increased for H100
                verbose=0
            )
            y_pred = (model.predict(X_val_seq, verbose=0) > 0.5).astype(int)
            return accuracy_score(y_val_arr, y_pred)
        except Exception as e:
            logger.error(f"Fit error: {e}")
            return 0.0

    results_list: List[AlgorithmResult] = []

    # ═══════════════════════════════════════════════════════════════════
    # 3. CUCKOO SEARCH OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "─"*70)
    logger.info("PHASE 2.1: CUCKOO SEARCH (CS)")
    logger.info("─"*70)
    
    cs_checkpoint = checkpoint_mgr.load_checkpoint("CuckooSearch")
    best_config_global = None
    
    if cs_checkpoint:
        logger.info("✅ Loaded CS results from checkpoint")
        res = AlgorithmResult(**cs_checkpoint)
        results_list.append(res)
        best_config_global = HyperparameterConfig(res.best_lstm_units, res.best_dropout, res.best_learning_rate)
    else:
        logger.info("Running Cuckoo Search...")
        start_time = time.time()
        
        # CS Parameters
        n_nests = 10
        pa = 0.25 # Probability of discovery/abandoning
        n_iterations = 5 
        
        # Bounds: LSTM (32-128), Dropout (0.2-0.5), LR (0.001-0.01)
        bounds = [(32, 128), (0.2, 0.5), (0.001, 0.01)]
        
        # Initialize Nests
        nests = [] # List of [lstm, drop, lr]
        fitness = []
        
        for _ in range(n_nests):
            lstm = int(np.random.uniform(bounds[0][0], bounds[0][1]))
            drop = np.random.uniform(bounds[1][0], bounds[1][1])
            lr = np.random.uniform(bounds[2][0], bounds[2][1])
            nests.append([lstm, drop, lr])
            
        logger.info("  Evaluating initial nests...")
        for i in range(n_nests):
            cfg = HyperparameterConfig(int(nests[i][0]), nests[i][1], nests[i][2])
            acc = fitness_function(cfg)
            fitness.append(acc)
            logger.info(f"    Nest {i+1}: {cfg} -> {acc:.4f}")
            
        # Get current best
        best_idx = np.argmax(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Levy Flight Helper
        def get_levy_step(lambda_val=1.5):
            sigma = (math.gamma(1 + lambda_val) * math.sin(math.pi * lambda_val / 2) / 
                    (math.gamma((1 + lambda_val) / 2) * lambda_val * 2 ** ((lambda_val - 1) / 2))) ** (1 / lambda_val)
            u = np.random.normal(0, sigma)
            v = np.random.normal(0, 1)
            step = u / abs(v) ** (1 / lambda_val)
            return step

        for iter_num in range(n_iterations):
            logger.info(f"  CS Iteration {iter_num + 1}/{n_iterations}")
            
            # 1. Generate new solutions via Levy Flights
            new_nests = []
            for i in range(n_nests):
                old_nest = nests[i]
                step_size = 0.01 * get_levy_step() * (np.array(old_nest) - np.array(best_nest))
                
                new_sol = np.array(old_nest) + step_size * np.random.randn(3) # Simple random walk component
                
                # Clip
                new_sol[0] = np.clip(new_sol[0], bounds[0][0], bounds[0][1])
                new_sol[1] = np.clip(new_sol[1], bounds[1][0], bounds[1][1])
                new_sol[2] = np.clip(new_sol[2], bounds[2][0], bounds[2][1])
                new_nests.append(list(new_sol))
            
            # Evaluate new solutions (Selection)
            for i in range(n_nests):
                cfg = HyperparameterConfig(int(new_nests[i][0]), new_nests[i][1], new_nests[i][2])
                f_new = fitness_function(cfg)
                
                if f_new > fitness[i]:
                    nests[i] = new_nests[i]
                    fitness[i] = f_new
            
            # 2. Abandon worst nests (Discovery)
            # Create mask
            k = int(n_nests * pa)
            # Sort by fitness
            sorted_indices = np.argsort(fitness)
            worst_indices = sorted_indices[:k] # ascending
            
            for idx in worst_indices:
                # Replace with random
                lstm = int(np.random.uniform(bounds[0][0], bounds[0][1]))
                drop = np.random.uniform(bounds[1][0], bounds[1][1])
                lr = np.random.uniform(bounds[2][0], bounds[2][1])
                nests[idx] = [lstm, drop, lr]
                # Re-eval
                cfg = HyperparameterConfig(lstm, drop, lr)
                fitness[idx] = fitness_function(cfg)
            
            # Update Best
            curr_best_idx = np.argmax(fitness)
            if fitness[curr_best_idx] > best_fitness:
                best_fitness = fitness[curr_best_idx]
                best_nest = nests[curr_best_idx].copy()
                logger.info(f"    └─ New Best: {best_fitness:.4f}")

        exec_time = time.time() - start_time
        best_config_global = HyperparameterConfig(int(best_nest[0]), best_nest[1], best_nest[2])
        
        result = AlgorithmResult(
            algorithm_name="CuckooSearch",
            best_accuracy=float(best_fitness),
            best_lstm_units=best_config_global.lstm_units,
            best_dropout=best_config_global.dropout_rate,
            best_learning_rate=best_config_global.learning_rate,
            execution_time_seconds=float(exec_time),
            iterations_completed=n_iterations,
            timestamp=datetime.now().isoformat()
        )
        checkpoint_mgr.save_checkpoint("CuckooSearch", result.to_dict())
        results_list.append(result)
        logger.info(f"✅ CS Complete. Best: {best_fitness:.4f} @ {best_config_global}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. EXPLAINABLE AI (XAI)
    # ═══════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "─"*70)
    logger.info("PHASE 2.2: XAI ANALYSIS")
    logger.info("─"*70)
    
    if best_config_global:
        logger.info(f"Training best model for XAI: {best_config_global}")
        # Retrain on full train data, for more epochs
        full_model = build_bilstm(best_config_global)
        full_model.fit(X_train_seq, y_train_arr, epochs=5, batch_size=64, verbose=0)
        
        # Save Model Artifact
        full_model.save("/tmp/best_bilstm_model.keras")
        logger.info("✅ Best model retrained and saved.")

        # --- A. SHAP ---
        logger.info("Running SHAP Analysis...")
        try:
            # DeepExplainer is ideal for DL, but can be slow. Use background sample.
            background = X_train_seq[:100]
            explainer = shap.GradientExplainer(full_model, background)
            
            test_sample = X_test_seq[:5]
            shap_values = explainer.shap_values(test_sample)
            
            # Since SHAP visualizers often need matplotlib, we save plot
            import matplotlib.pyplot as plt
            # We can't easily display it, but we can verify it ran
            shap_summary_path = checkpoint_dir / "shap_summary_test.npy"
            np.save(shap_summary_path, shap_values)
            logger.info(f"✅ SHAP values calculated and saved to {shap_summary_path}")
        except Exception as e:
            logger.error(f"❌ SHAP Failed: {e}")

        # --- B. LIME ---
        logger.info("Running LIME Analysis...")
        try:
            explainer_lime = lime_tabular.LimeTabularExplainer(
                training_data=X_train_seq,
                mode="classification",
                feature_names=[f"word_{i}" for i in range(MAX_LEN)]
            )
            # Explain first test instance
            exp = explainer_lime.explain_instance(
                X_test_seq[0], 
                full_model.predict, 
                num_features=10
            )
            lime_path = checkpoint_dir / "lime_explanation.html"
            exp.save_to_file(str(lime_path))
            logger.info(f"✅ LIME explanation saved to {lime_path}")
        except Exception as e:
            logger.error(f"❌ LIME Failed: {e}")
            
    else:
        logger.warning("Skipping XAI as no best config found.")

    return {"status": "success", "results": [r.to_dict() for r in results_list]}

@app.local_entrypoint()
def main():
    print("🚀 Starting Phase 2 on Modal (H100)...")
    try:
        res = run_phase2_training.remote()
        print("\n✅ Execution Complete!")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")