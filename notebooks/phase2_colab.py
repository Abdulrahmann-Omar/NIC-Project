# -*- coding: utf-8 -*-
"""
================================================================================
NATURE INSPIRED COMPUTATION - PHASE 2 (Google Colab Version)
Cuckoo Search Optimization + Explainable AI (XAI)
GPU Accelerated (T4/V100/A100)
================================================================================

This script is designed to run in Google Colab with GPU acceleration.
It implements:
1. Cuckoo Search (CS) for hyperparameter optimization
2. XAI Analysis: SHAP, LIME, and Saliency Maps

Usage in Colab:
  1. Upload this file or clone the repo
  2. Run each section in order
  3. Results are saved to Google Drive (optional)

================================================================================
"""

# ==============================================================================
# SECTION 0: SETUP & INSTALLATION (Run First in Colab)
# ==============================================================================

# Uncomment and run in Colab:
# !pip install -q tensorflow keras shap lime nltk scikit-learn matplotlib seaborn tqdm

import os
import json
import time
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm.auto import tqdm

# ==============================================================================
# SECTION 1: GPU CONFIGURATION
# ==============================================================================

def setup_gpu():
    """Configure GPU for optimal performance."""
    import tensorflow as tf
    
    print("=" * 60)
    print("  GPU Configuration")
    print("=" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] GPU Detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"     - {gpu.name}")
        
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Enable mixed precision for faster training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[OK] Mixed Precision (FP16) Enabled")
    else:
        print("[WARNING] No GPU detected - using CPU (slower)")
    
    return len(gpus) > 0

# ==============================================================================
# SECTION 2: DATA LOADING
# ==============================================================================

def download_and_preprocess_data(max_words=5000, max_len=50):
    """Download IMDB dataset and preprocess for training."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    
    print("\n" + "=" * 60)
    print("  Data Loading & Preprocessing")
    print("=" * 60)
    
    # Load IMDB dataset
    print("[...] Loading IMDB dataset...")
    (X_train_raw, y_train), (X_test_raw, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_words)
    
    # Decode back to text for tokenization
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    def decode_review(encoded):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded])
    
    # Pad sequences
    print("[...] Padding sequences...")
    X_train_seq = pad_sequences(X_train_raw, maxlen=max_len, padding='post', truncating='post')
    X_test_seq = pad_sequences(X_test_raw, maxlen=max_len, padding='post', truncating='post')
    
    # Split train into train/val
    X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_train_seq, y_train, test_size=0.2, random_state=42
    )
    
    print(f"[OK] Data loaded:")
    print(f"     Train: {len(X_train_seq)} samples")
    print(f"     Val:   {len(X_val_seq)} samples")
    print(f"     Test:  {len(X_test_seq)} samples")
    
    return {
        'X_train': X_train_seq,
        'X_val': X_val_seq,
        'X_test': X_test_seq,
        'y_train': np.array(y_train),
        'y_val': np.array(y_val),
        'y_test': np.array(y_test),
        'max_words': max_words,
        'max_len': max_len
    }

# ==============================================================================
# SECTION 3: MODEL DEFINITION
# ==============================================================================

@dataclass
class HyperparameterConfig:
    """BiLSTM model hyperparameters."""
    lstm_units: int
    dropout_rate: float
    learning_rate: float
    
    def __repr__(self) -> str:
        return f"LSTM={self.lstm_units}, Drop={self.dropout_rate:.3f}, LR={self.learning_rate:.6f}"

def build_bilstm(config: HyperparameterConfig, max_words: int, max_len: int):
    """Build BiLSTM model with given hyperparameters."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
    
    model = Sequential([
        Embedding(max_words, 64, input_length=max_len, name="embedding"),
        Bidirectional(LSTM(config.lstm_units, return_sequences=False), name="bilstm"),
        Dropout(config.dropout_rate, name="dropout1"),
        Dense(32, activation='relu', name="dense1"),
        Dropout(config.dropout_rate / 2, name="dropout2"),
        Dense(1, activation='sigmoid', dtype='float32', name="output")  # float32 for mixed precision
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        metrics=['accuracy']
    )
    
    return model

# ==============================================================================
# SECTION 4: CUCKOO SEARCH OPTIMIZATION
# ==============================================================================

@dataclass
class AlgorithmResult:
    """Structured result from optimization algorithm."""
    algorithm_name: str
    best_accuracy: float
    best_lstm_units: int
    best_dropout: float
    best_learning_rate: float
    execution_time_seconds: float
    iterations_completed: int
    convergence_history: List[float]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def levy_flight(lambda_val=1.5):
    """Generate Levy flight step using Mantegna's algorithm."""
    sigma = (math.gamma(1 + lambda_val) * math.sin(math.pi * lambda_val / 2) / 
            (math.gamma((1 + lambda_val) / 2) * lambda_val * 2 ** ((lambda_val - 1) / 2))) ** (1 / lambda_val)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / lambda_val)
    return step

def cuckoo_search(data: dict, n_nests=10, pa=0.25, n_iterations=10, epochs_per_eval=2):
    """
    Cuckoo Search optimization for BiLSTM hyperparameters.
    
    Args:
        data: Dictionary with train/val/test data
        n_nests: Number of host nests (population size)
        pa: Probability of discovery (abandonment rate)
        n_iterations: Number of CS iterations
        epochs_per_eval: Training epochs per fitness evaluation
    
    Returns:
        AlgorithmResult with optimization results
    """
    from sklearn.metrics import accuracy_score
    import tensorflow as tf
    
    print("\n" + "=" * 60)
    print("  CUCKOO SEARCH OPTIMIZATION")
    print("=" * 60)
    print(f"  Parameters: nests={n_nests}, pa={pa}, iterations={n_iterations}")
    print("=" * 60)
    
    # Search bounds: [LSTM units, Dropout, Learning Rate]
    bounds = [(32, 128), (0.2, 0.5), (0.001, 0.01)]
    
    def fitness_function(config: HyperparameterConfig) -> float:
        """Train model and return validation accuracy."""
        try:
            tf.keras.backend.clear_session()
            model = build_bilstm(config, data['max_words'], data['max_len'])
            model.fit(
                data['X_train'], data['y_train'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=epochs_per_eval,
                batch_size=256,
                verbose=0
            )
            y_pred = (model.predict(data['X_val'], verbose=0) > 0.5).astype(int)
            return accuracy_score(data['y_val'], y_pred)
        except Exception as e:
            print(f"[ERROR] Fitness evaluation failed: {e}")
            return 0.0
    
    start_time = time.time()
    convergence_history = []
    
    # Initialize nests randomly
    print("\n[Phase 1] Initializing nests...")
    nests = []
    fitness = []
    
    for i in tqdm(range(n_nests), desc="Initial evaluation"):
        lstm = int(np.random.uniform(bounds[0][0], bounds[0][1]))
        drop = np.random.uniform(bounds[1][0], bounds[1][1])
        lr = np.random.uniform(bounds[2][0], bounds[2][1])
        nests.append([lstm, drop, lr])
        
        config = HyperparameterConfig(lstm, drop, lr)
        acc = fitness_function(config)
        fitness.append(acc)
        print(f"  Nest {i+1}: {config} -> Acc: {acc:.4f}")
    
    # Find initial best
    best_idx = np.argmax(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    convergence_history.append(best_fitness)
    print(f"\n[OK] Initial best: {best_fitness:.4f}")
    
    # Main CS loop
    print("\n[Phase 2] Cuckoo Search iterations...")
    for iter_num in range(n_iterations):
        print(f"\n--- Iteration {iter_num + 1}/{n_iterations} ---")
        
        # Step 1: Generate new solutions via Levy flights
        new_nests = []
        for i in range(n_nests):
            old_nest = nests[i]
            step_size = 0.01 * levy_flight() * (np.array(old_nest) - np.array(best_nest))
            new_sol = np.array(old_nest) + step_size * np.random.randn(3)
            
            # Clip to bounds
            new_sol[0] = np.clip(new_sol[0], bounds[0][0], bounds[0][1])
            new_sol[1] = np.clip(new_sol[1], bounds[1][0], bounds[1][1])
            new_sol[2] = np.clip(new_sol[2], bounds[2][0], bounds[2][1])
            new_nests.append(list(new_sol))
        
        # Evaluate and update
        for i in tqdm(range(n_nests), desc="Levy flight eval"):
            config = HyperparameterConfig(int(new_nests[i][0]), new_nests[i][1], new_nests[i][2])
            f_new = fitness_function(config)
            
            if f_new > fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = f_new
        
        # Step 2: Abandon worst nests (discovery)
        k = int(n_nests * pa)
        sorted_indices = np.argsort(fitness)
        worst_indices = sorted_indices[:k]
        
        for idx in worst_indices:
            lstm = int(np.random.uniform(bounds[0][0], bounds[0][1]))
            drop = np.random.uniform(bounds[1][0], bounds[1][1])
            lr = np.random.uniform(bounds[2][0], bounds[2][1])
            nests[idx] = [lstm, drop, lr]
            
            config = HyperparameterConfig(lstm, drop, lr)
            fitness[idx] = fitness_function(config)
        
        # Update global best
        curr_best_idx = np.argmax(fitness)
        if fitness[curr_best_idx] > best_fitness:
            best_fitness = fitness[curr_best_idx]
            best_nest = nests[curr_best_idx].copy()
            print(f"  [NEW BEST] Accuracy: {best_fitness:.4f}")
        
        convergence_history.append(best_fitness)
        print(f"  Current best: {best_fitness:.4f}")
    
    exec_time = time.time() - start_time
    best_config = HyperparameterConfig(int(best_nest[0]), best_nest[1], best_nest[2])
    
    print("\n" + "=" * 60)
    print(f"  CUCKOO SEARCH COMPLETE")
    print(f"  Best Accuracy: {best_fitness:.4f}")
    print(f"  Best Config: {best_config}")
    print(f"  Runtime: {exec_time:.1f} seconds")
    print("=" * 60)
    
    return AlgorithmResult(
        algorithm_name="CuckooSearch",
        best_accuracy=float(best_fitness),
        best_lstm_units=best_config.lstm_units,
        best_dropout=best_config.dropout_rate,
        best_learning_rate=best_config.learning_rate,
        execution_time_seconds=float(exec_time),
        iterations_completed=n_iterations,
        convergence_history=convergence_history,
        timestamp=datetime.now().isoformat()
    ), best_config

# ==============================================================================
# SECTION 5: EXPLAINABLE AI (XAI) ANALYSIS
# ==============================================================================

def run_xai_analysis(model, data: dict, save_dir: str = "./xai_outputs"):
    """Run SHAP and LIME analysis on the trained model."""
    import shap
    from lime import lime_tabular
    
    print("\n" + "=" * 60)
    print("  EXPLAINABLE AI ANALYSIS")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    # --- SHAP Analysis ---
    print("\n[1/2] Running SHAP Analysis...")
    try:
        # Use a small background sample for efficiency
        background = data['X_train'][:100]
        explainer = shap.GradientExplainer(model, background)
        
        test_sample = data['X_test'][:10]
        shap_values = explainer.shap_values(test_sample)
        
        # Save SHAP values
        shap_path = os.path.join(save_dir, "shap_values.npy")
        np.save(shap_path, shap_values)
        
        # Create summary plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values[0], test_sample, show=False, max_display=20)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=150)
        plt.close()
        
        results['shap'] = {'status': 'success', 'path': shap_path}
        print(f"  [OK] SHAP values saved to {shap_path}")
        
    except Exception as e:
        results['shap'] = {'status': 'error', 'message': str(e)}
        print(f"  [ERROR] SHAP failed: {e}")
    
    # --- LIME Analysis ---
    print("\n[2/2] Running LIME Analysis...")
    try:
        explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=data['X_train'],
            mode="classification",
            feature_names=[f"word_{i}" for i in range(data['max_len'])]
        )
        
        # Explain first test instance
        exp = explainer_lime.explain_instance(
            data['X_test'][0],
            model.predict,
            num_features=10
        )
        
        lime_path = os.path.join(save_dir, "lime_explanation.html")
        exp.save_to_file(lime_path)
        
        # Create feature importance plot
        fig = exp.as_pyplot_figure()
        fig.savefig(os.path.join(save_dir, "lime_importance.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        results['lime'] = {'status': 'success', 'path': lime_path}
        print(f"  [OK] LIME explanation saved to {lime_path}")
        
    except Exception as e:
        results['lime'] = {'status': 'error', 'message': str(e)}
        print(f"  [ERROR] LIME failed: {e}")
    
    return results

# ==============================================================================
# SECTION 6: VISUALIZATION
# ==============================================================================

def plot_convergence(result: AlgorithmResult, save_path: str = None):
    """Plot convergence curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(result.convergence_history, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.title(f'{result.algorithm_name} Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add best accuracy annotation
    best_iter = np.argmax(result.convergence_history)
    plt.annotate(f'Best: {result.best_accuracy:.4f}',
                xy=(best_iter, result.best_accuracy),
                xytext=(best_iter + 0.5, result.best_accuracy + 0.01),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Convergence plot saved to {save_path}")
    
    plt.show()

def create_results_summary(cs_result: AlgorithmResult, xai_results: dict, save_dir: str = "."):
    """Create comprehensive results summary."""
    
    # Create results dataframe
    summary = {
        'Metric': [
            'Algorithm',
            'Best Accuracy',
            'LSTM Units',
            'Dropout Rate',
            'Learning Rate',
            'Runtime (seconds)',
            'Iterations',
            'SHAP Status',
            'LIME Status'
        ],
        'Value': [
            cs_result.algorithm_name,
            f"{cs_result.best_accuracy:.4f}",
            cs_result.best_lstm_units,
            f"{cs_result.best_dropout:.4f}",
            f"{cs_result.best_learning_rate:.6f}",
            f"{cs_result.execution_time_seconds:.1f}",
            cs_result.iterations_completed,
            xai_results.get('shap', {}).get('status', 'N/A'),
            xai_results.get('lime', {}).get('status', 'N/A')
        ]
    }
    
    df = pd.DataFrame(summary)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, "phase2_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results summary saved to {csv_path}")
    
    # Display
    print("\n" + "=" * 60)
    print("  PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    return df

# ==============================================================================
# SECTION 7: MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution pipeline."""
    print("\n")
    print("=" * 70)
    print("  NATURE INSPIRED COMPUTATION - PHASE 2")
    print("  Cuckoo Search + Explainable AI")
    print("=" * 70)
    
    # Setup
    has_gpu = setup_gpu()
    
    # Load data
    data = download_and_preprocess_data(max_words=5000, max_len=50)
    
    # Run Cuckoo Search
    cs_result, best_config = cuckoo_search(
        data,
        n_nests=8,        # Reduce for faster demo
        pa=0.25,
        n_iterations=5,   # Reduce for faster demo
        epochs_per_eval=2
    )
    
    # Train final model with best config
    print("\n[FINAL] Training optimized model for XAI...")
    import tensorflow as tf
    final_model = build_bilstm(best_config, data['max_words'], data['max_len'])
    final_model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=5,
        batch_size=128,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = final_model.evaluate(data['X_test'], data['y_test'], verbose=0)
    print(f"\n[OK] Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model_path = "./best_bilstm_model.keras"
    final_model.save(model_path)
    print(f"[OK] Model saved to {model_path}")
    
    # Run XAI analysis
    xai_results = run_xai_analysis(final_model, data, save_dir="./xai_outputs")
    
    # Plot convergence
    plot_convergence(cs_result, save_path="./convergence_plot.png")
    
    # Create summary
    create_results_summary(cs_result, xai_results, save_dir=".")
    
    # Save full results as JSON
    with open("phase2_full_results.json", "w") as f:
        json.dump(cs_result.to_dict(), f, indent=2)
    
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - best_bilstm_model.keras")
    print("  - phase2_results.csv")
    print("  - phase2_full_results.json")
    print("  - convergence_plot.png")
    print("  - xai_outputs/shap_summary.png")
    print("  - xai_outputs/lime_explanation.html")
    
    return cs_result, xai_results

# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    result, xai = main()
