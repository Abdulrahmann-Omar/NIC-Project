"""
Nature Inspired Computation - Phase 1
Modal.com Deployment with DOE + 6 Metaheuristics
With Terminal Observability + Checkpoint System

Run with: modal run phase1_modal.py
"""

import modal
import json
from pathlib import Path

# Create Modal app
app = modal.App("nic-phase1-observable")

# Define GPU image - using TensorFlow's official GPU image
image = (
    modal.Image.from_registry("tensorflow/tensorflow:latest-gpu")
    .apt_install("unzip", "wget")
    .pip_install(
        "pandas", 
        "scikit-learn",
        "nltk",
        "tqdm",
        "matplotlib",
        "seaborn",
        "kaggle"
    )
)

# Modal Volume for persistent checkpoints
volume = modal.Volume.from_name("nic-phase1-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # NVIDIA H100 - Highest-end GPU on Modal (~7x faster than A10G)
    timeout=3600,
    secrets=[modal.Secret.from_name("kaggle-secret")],
    volumes={"/checkpoints": volume}  # Mount persistent storage
)
def run_phase1_training():
    """Main training function running on Modal.com GPU"""
    
    import os
    import pandas as pd
    import numpy as np
    import re
    import time
    import nltk
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from tqdm import tqdm
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Startup banner
    print("\n" + "="*70)
    print("    PHASE 1: NATURE INSPIRED COMPUTATION ON MODAL.COM H100 GPU")
    print("="*70)
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # GPU verification
    print("🔍 Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   └─ GPU {i}: {gpu.name}")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be SLOW.")
    print()
    
    # Download NLTK data
    print("📥 Downloading NLTK data...")
    for pkg in tqdm(['stopwords', 'punkt', 'punkt_tab', 'wordnet'], desc="NLTK packages"):
        nltk.download(pkg, quiet=True)
    print("✅ NLTK data ready\n")
    
    # =============================================
    # CHECKPOINT MANAGER SETUP
    # =============================================
    checkpoint_dir = Path("/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(algo_name, data):
        """Save algorithm results to checkpoint"""
        try:
            checkpoint_file = checkpoint_dir / f"{algo_name}_checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "algorithm": algo_name,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "data": data
                }, f, indent=2)
            print(f"    💾 Checkpoint saved: {algo_name}")
            volume.commit()  # Persist to Modal Volume
            return True
        except Exception as e:
            print(f"    ⚠️ Checkpoint save failed: {e}")
            return False
    
    def load_checkpoint(algo_name):
        """Load checkpoint if exists"""
        checkpoint_file = checkpoint_dir / f"{algo_name}_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                print(f"    ♻️ Loaded checkpoint: {algo_name} ({data['timestamp']})")
                return data['data']
            except:
                return None
        return None
    
    # =============================================
    # 1. DATASET LOADING & PREPROCESSING
    # =============================================
    print("\n" + "─"*70)
    print("📥 TASK 1: DATASET LOADING & PREPROCESSING")
    print("─"*70)
    
    # Check for cached preprocessed data
    preprocessed_file = checkpoint_dir / "preprocessed_data.npz"
    
    if preprocessed_file.exists():
        print("♻️ Loading cached preprocessed data...")
        data = np.load(preprocessed_file, allow_pickle=True)
        X_train_seq = data['X_train_seq']
        X_val_seq = data['X_val_seq']
        X_test_seq = data['X_test_seq']
        y_train_arr = data['y_train_arr']
        y_val_arr = data['y_val_arr']
        y_test_arr = data['y_test_arr']
        print(f"✅ Loaded from cache in {checkpoint_dir}")
        print(f"   Train: {len(X_train_seq)} | Val: {len(X_val_seq)} | Test: {len(X_test_seq)}\n")
    else:
        print("📥 No cache found - downloading and preprocessing...")
    
    # Configure Kaggle credentials from Modal secrets
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if not kaggle_username or not kaggle_key:
        raise ValueError("Kaggle credentials not found in Modal secrets!")
    
    # Create .kaggle directory and credentials file
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_config = {
        "username": kaggle_username,
        "key": kaggle_key
    }
    
    import json
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), 'w') as f:
        json.dump(kaggle_config, f)
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    
    print("✅ Kaggle credentials configured")
    
    # Download dataset using Kaggle API
    os.system("kaggle datasets download -d kazanova/sentiment140 -p /tmp --force")
    os.system("unzip -o /tmp/sentiment140.zip -d /tmp/")
    
    COL_NAMES = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv('/tmp/training.1600000.processed.noemoticon.csv', 
                     encoding='ISO-8859-1', names=COL_NAMES)
    df = df.drop_duplicates(subset=['text', 'target'], keep='first')
    
    # Sample 100K for training
    SAMPLE_SIZE = 30000
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Dataset shape: {df.shape}")
    
    # Preprocessing functions
    def basic_text_cleaning(text):
        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'@[^\s]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_lemmatize(text, stop_words, lemmatizer):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(tok) for tok in tokens 
                 if tok not in stop_words and len(tok) > 2]
        return ' '.join(lemmas)
    
    print("Preprocessing...")
    df['text_cleaned'] = df['text'].apply(basic_text_cleaning)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    df['text_final'] = df['text_cleaned'].apply(
        lambda t: tokenize_lemmatize(t, stop_words, lemmatizer)
    )
    
    df['target_encoded'] = df['target'].replace(4, 1)
    df = df[df['text_final'] != '']
    
    # Split data
    X = df['text_final']
    y = df['target_encoded']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # =============================================
    # 2. NEURAL NETWORK SETUP
    # =============================================
    MAX_WORDS = 10000
    MAX_LEN = 50
    
    tokenizer_nn = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer_nn.fit_on_texts(X_train)
    
    X_train_seq = pad_sequences(tokenizer_nn.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_val_seq = pad_sequences(tokenizer_nn.texts_to_sequences(X_val), maxlen=MAX_LEN)
    X_test_seq = pad_sequences(tokenizer_nn.texts_to_sequences(X_test), maxlen=MAX_LEN)
    
    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values
    
    # Model builder
    def build_bilstm(lstm_units=64, dropout_rate=0.6, learning_rate=1):
        model = Sequential([
            Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
            Bidirectional(LSTM(lstm_units, return_sequences=False)),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dropout(dropout_rate/2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        return model
    
    # Fitness function
    def fitness_function(lstm_units, dropout_rate, learning_rate):
        tf.keras.backend.clear_session()
        model = build_bilstm(lstm_units, dropout_rate, learning_rate)
        model.fit(X_train_seq, y_train_arr,
                 validation_data=(X_val_seq, y_val_arr),
                 epochs=2, batch_size=64, verbose=0)
        y_pred = (model.predict(X_val_seq, verbose=0) > 0.5).astype(int)
        return accuracy_score(y_val_arr, y_pred)
    
    # =============================================
    # 3. BASELINE MODEL
    # =============================================
    print("\n" + "─"*70)
    print("📊 TASK 3: BASELINE MODEL TRAINING")
    print("─"*70)
    print("Configuration: LSTM=64, Dropout=0.6, LR=1")
    print("Training for 3 epochs...\n")
    
    baseline_model = build_bilstm()
    baseline_start = time.time()
    baseline_model.fit(X_train_seq, y_train_arr,
                      validation_data=(X_val_seq, y_val_arr),
                      epochs=3, batch_size=64, verbose=1)
    baseline_time = time.time() - baseline_start
    
    print("\n🎯 Evaluating on test set...")
    y_pred = (baseline_model.predict(X_test_seq, verbose=0) > 0.5).astype(int)
    baseline_acc = accuracy_score(y_test_arr, y_pred)
    print(f"\n✅ Baseline Accuracy: {baseline_acc:.4f} (Training time: {baseline_time:.1f}s)")
    print("─"*70 + "\n")
    
    # =============================================
    # 4. DESIGN OF EXPERIMENTS (DOE) - TAGUCHI L9
    # =============================================
    print("\n" + "─"*70)
    print("📊 TASK 4: DESIGN OF EXPERIMENTS (TAGUCHI L2)")
    print("─"*70)
    
    # Taguchi orthogonal array
    taguchi_l9 = [
        (128, 0.2, 0.01),
        (128, 0.35, 0.001),
    ]
    
    print(f"Running {len(taguchi_l9)} systematic experiments...\n")
    
    doe_results = []
    start_time = time.time()
    
    for i, (lstm, dropout, lr) in enumerate(tqdm(taguchi_l9, desc="DOE Progress"), 1):
        print(f"\n  🔬 Experiment {i}/{len(taguchi_l9)}: LSTM={lstm}, Dropout={dropout:.2f}, LR={lr:.4f}")
        exp_start = time.time()
        acc = fitness_function(lstm, dropout, lr)
        exp_time = time.time() - exp_start
        doe_results.append({
            'lstm_units': lstm,
            'dropout_rate': dropout,
            'learning_rate': lr,
            'accuracy': acc
        })
        print(f"    ✅ Accuracy: {acc:.4f} (Time: {exp_time:.1f}s)")
    
    doe_time = time.time() - start_time
    doe_df = pd.DataFrame(doe_results)
    doe_best = doe_df.loc[doe_df['accuracy'].idxmax()]
    
    print(f"\n✅ DOE Best: Acc={doe_best['accuracy']:.4f}, "
          f"LSTM={doe_best['lstm_units']}, Dropout={doe_best['dropout_rate']:.3f}, "
          f"LR={doe_best['learning_rate']:.6f}")
    print(f"Time: {doe_time:.2f}s")
    
    # Results storage
    RESULTS = {
        'Algorithm': ['DOE-Taguchi'],
        'Best_Accuracy': [doe_best['accuracy']],
        'Best_LSTM_Units': [int(doe_best['lstm_units'])],
        'Best_Dropout': [round(doe_best['dropout_rate'], 4)],
        'Best_LR': [round(doe_best['learning_rate'], 6)],
        'Time_Seconds': [round(doe_time, 2)]
    }
    
    # =============================================
    # 5. PARTICLE SWARM OPTIMIZATION (PSO)
    # =============================================
    print("\n" + "─"*70)
    print("🐝 TASK 5: PARTICLE SWARM OPTIMIZATION (PSO)")
    print("─"*70)
    print("Configuration: 5 particles, 3 iterations")
    print("Search space: LSTM∈{32,64,128}, Dropout∈[0.2,0.5], LR∈[0.001,0.01]\n")
    
    def decode_pso(solution):
        lstm_idx = min(int(solution[0] * 3), 2)
        lstm_units = [32, 64, 128][lstm_idx]
        dropout = 0.2 + solution[1] * 0.3
        lr = 0.001 + solution[2] * 0.009
        return lstm_units, dropout, lr
    
    def pso_fitness(solution):
        lstm, dropout, lr = decode_pso(solution)
        return fitness_function(lstm, dropout, lr)
    
    # PSO Algorithm
    n_particles, n_iter = 5, 3
    dim = 3
    particles = np.random.rand(n_particles, dim)
    velocities = np.random.rand(n_particles, dim) * 0.1
    personal_best = particles.copy()
    personal_best_scores = np.array([pso_fitness(p) for p in particles])
    
    global_best_idx = np.argmax(personal_best_scores)
    global_best = personal_best[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]
    
    print(f"🎯 Initial best: {global_best_score:.4f}")
    start_time = time.time()
    
    pso_pbar = tqdm(range(n_iter), desc="PSO Iterations")
    for iteration in pso_pbar:
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = 0.7 * velocities[i] + 1.5 * r1 * (personal_best[i] - particles[i]) + 1.5 * r2 * (global_best - particles[i])
            particles[i] = np.clip(particles[i] + velocities[i], 0, 1)
            
            score = pso_fitness(particles[i])
            if score > personal_best_scores[i]:
                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = score
                if score > global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score
        pso_pbar.set_postfix({'best': f'{global_best_score:.4f}'})
        print(f"  ✅ Iteration {iteration+1}/{n_iter}: Best = {global_best_score:.4f}")
    
    pso_time = time.time() - start_time
    lstm, dropout, lr = decode_pso(global_best)
    
    RESULTS['Algorithm'].append('PSO')
    RESULTS['Best_Accuracy'].append(global_best_score)
    RESULTS['Best_LSTM_Units'].append(lstm)
    RESULTS['Best_Dropout'].append(round(dropout, 4))
    RESULTS['Best_LR'].append(round(lr, 6))
    RESULTS['Time_Seconds'].append(round(pso_time, 2))
    
    print(f"✅ PSO Best: Acc={global_best_score:.4f}, LSTM={lstm}, Dropout={dropout:.3f}, LR={lr:.6f}")
    
    # =============================================
    # 6. TABU SEARCH
    # =============================================
    print("\n" + "─"*70)
    print("📋 TASK 6: TABU SEARCH")
    print("─"*70)
    print("Configuration: 8 iterations, Tabu tenure=5")
    print("Memory-based local search with tabu list\n")
    
    def get_neighbors(solution):
        """Generate neighbors by perturbing solution"""
        neighbors = []
        for i in range(3):
            for delta in [-0.1, 0.1]:
                neighbor = solution.copy()
                neighbor[i] = np.clip(neighbor[i] + delta, 0, 1)
                neighbors.append(neighbor)
        return neighbors
    
    # Tabu Search
    n_iterations = 8
    tabu_tenure = 5
    tabu_list = []
    
    current = np.random.rand(3)
    current_score = pso_fitness(current)
    best_solution = current.copy()
    best_score = current_score
    
    print(f"🎯 Initial best: {best_score:.4f}")
    start_time = time.time()
    
    tabu_pbar = tqdm(range(n_iterations), desc="Tabu Search")
    for iteration in tabu_pbar:
        neighbors = get_neighbors(current)
        
        # Evaluate non-tabu neighbors
        best_neighbor = None
        best_neighbor_score = -1
        
        for neighbor in neighbors:
            neighbor_tuple = tuple(np.round(neighbor, 2))
            if neighbor_tuple not in tabu_list:
                score = pso_fitness(neighbor)
                if score > best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = score
        
        if best_neighbor is not None:
            current = best_neighbor
            current_score = best_neighbor_score
            
            # Update tabu list
            tabu_list.append(tuple(np.round(current, 2)))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            # Update best
            if current_score > best_score:
                best_solution = current.copy()
                best_score = current_score
        
        tabu_pbar.set_postfix({'best': f'{best_score:.4f}', 'tabu_size': len(tabu_list)})
        if iteration % 2 == 0 or iteration == n_iterations - 1:
            print(f"  ✅ Iteration {iteration+1}/{n_iterations}: Best = {best_score:.4f}, Tabu size = {len(tabu_list)}")
    
    tabu_time = time.time() - start_time
    lstm, dropout, lr = decode_pso(best_solution)
    
    RESULTS['Algorithm'].append('Tabu_Search')
    RESULTS['Best_Accuracy'].append(best_score)
    RESULTS['Best_LSTM_Units'].append(lstm)
    RESULTS['Best_Dropout'].append(round(dropout, 4))
    RESULTS['Best_LR'].append(round(lr, 6))
    RESULTS['Time_Seconds'].append(round(tabu_time, 2))
    
    print(f"✅ Tabu Best: Acc={best_score:.4f}, LSTM={lstm}, Dropout={dropout:.3f}, LR={lr:.6f}")
    
    # =============================================
    # 7. GREY WOLF OPTIMIZER (GWO)
    # =============================================
    print("\n" + "─"*70)
    print("🐺 TASK 7: GREY WOLF OPTIMIZER (GWO)")
    print("─"*70)
    print("Configuration: 5 wolves (Alpha, Beta, Delta), 3 iterations")
    print("Simulating pack hunting hierarchy\n")
    
    def gwo_fitness(solution):
        lstm, dropout, lr = decode_pso(solution)
        return fitness_function(lstm, dropout, lr)
    
    # GWO algorithm - simulates pack hunting behavior
    n_wolves = 10
    n_iter = 3
    dim = 3
    
    # Initialize wolf pack
    wolves = np.random.rand(n_wolves, dim)
    fitness_values = np.array([gwo_fitness(w) for w in wolves])
    
    # Identify alpha, beta, delta wolves (top 3)
    sorted_indices = np.argsort(fitness_values)[::-1]
    alpha = wolves[sorted_indices[0]].copy()
    beta = wolves[sorted_indices[1]].copy()
    delta = wolves[sorted_indices[2]].copy()
    alpha_score = fitness_values[sorted_indices[0]]
    
    print(f"🎯 Initial alpha score: {alpha_score:.4f}")
    start_time = time.time()
    
    gwo_pbar = tqdm(range(n_iter), desc="GWO Iterations")
    for iteration in gwo_pbar:
        a = 2 - iteration * (2 / n_iter)  # Linearly decreasing from 2 to 0
        
        for i in range(n_wolves):
            for j in range(dim):
                # Update position based on alpha, beta, delta
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A1 * D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i, j])
                X2 = beta[j] - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i, j])
                X3 = delta[j] - A3 * D_delta
                
                wolves[i, j] = (X1 + X2 + X3) / 3
            
            wolves[i] = np.clip(wolves[i], 0, 1)
        
        # Reevaluate and update hierarchy
        fitness_values = np.array([gwo_fitness(w) for w in wolves])
        sorted_indices = np.argsort(fitness_values)[::-1]
        
        if fitness_values[sorted_indices[0]] > alpha_score:
            alpha = wolves[sorted_indices[0]].copy()
            alpha_score = fitness_values[sorted_indices[0]]
        beta = wolves[sorted_indices[1]].copy()
        delta = wolves[sorted_indices[2]].copy()
        
        gwo_pbar.set_postfix({'alpha': f'{alpha_score:.4f}', 'a': f'{a:.2f}'})
        print(f"  ✅ Iteration {iteration+1}/{n_iter}: Alpha = {alpha_score:.4f}, Convergence = {a:.3f}")
    
    gwo_time = time.time() - start_time
    lstm, dropout, lr = decode_pso(alpha)
    
    RESULTS['Algorithm'].append('GWO')
    RESULTS['Best_Accuracy'].append(alpha_score)
    RESULTS['Best_LSTM_Units'].append(lstm)
    RESULTS['Best_Dropout'].append(round(dropout, 4))
    RESULTS['Best_LR'].append(round(lr, 6))
    RESULTS['Time_Seconds'].append(round(gwo_time, 2))
    
    print(f"✅ GWO Best: Acc={alpha_score:.4f}, LSTM={lstm}, Dropout={dropout:.3f}, LR={lr:.6f}")
    
    # =============================================
    # 8. WHALE OPTIMIZATION ALGORITHM (WOA)
    # =============================================
    print("\n" + "─"*70)
    print("🐋 TASK 8: WHALE OPTIMIZATION ALGORITHM (WOA)")
    print("─"*70)
    print("Configuration: 5 whales, 3 iterations")
    print("Simulating bubble-net feeding behavior\n")
    
    def woa_fitness(solution):
        lstm, dropout, lr = decode_pso(solution)
        return fitness_function(lstm, dropout, lr)
    
    # WOA algorithm - simulates humpback whale bubble-net feeding
    n_whales = 5
    n_iter = 3
    dim = 3
    
    whales = np.random.rand(n_whales, dim)
    fitness_values = np.array([woa_fitness(w) for w in whales])
    
    best_idx = np.argmax(fitness_values)
    best_whale = whales[best_idx].copy()
    best_score = fitness_values[best_idx]
    
    print(f"🎯 Initial best: {best_score:.4f}")
    start_time = time.time()
    
    woa_pbar = tqdm(range(n_iter), desc="WOA Iterations")
    for iteration in woa_pbar:
        a = 2 - iteration * (2 / n_iter)  # Decreasing from 2 to 0
        a2 = -1 + iteration * (-1 / n_iter)  # Decreasing from -1 to -2
        
        for i in range(n_whales):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            p = np.random.rand()
            
            if p < 0.5:
                if abs(A) < 1:
                    # Encircling prey
                    D = abs(C * best_whale - whales[i])
                    whales[i] = best_whale - A * D
                else:
                    # Search for prey (exploration)
                    random_whale = whales[np.random.randint(n_whales)]
                    D = abs(C * random_whale - whales[i])
                    whales[i] = random_whale - A * D
            else:
                # Spiral bubble-net feeding
                b = 1
                l = (a2 - 1) * np.random.rand() + 1
                D = abs(best_whale - whales[i])
                whales[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale
            
            whales[i] = np.clip(whales[i], 0, 1)
        
        # Update best
        fitness_values = np.array([woa_fitness(w) for w in whales])
        current_best_idx = np.argmax(fitness_values)
        
        if fitness_values[current_best_idx] > best_score:
            best_whale = whales[current_best_idx].copy()
            best_score = fitness_values[current_best_idx]
        
        woa_pbar.set_postfix({'best': f'{best_score:.4f}', 'a': f'{a:.2f}'})
        print(f"  ✅ Iteration {iteration+1}/{n_iter}: Best = {best_score:.4f}, Exploration = {a:.3f}")
    
    woa_time = time.time() - start_time
    lstm, dropout, lr = decode_pso(best_whale)
    
    RESULTS['Algorithm'].append('WOA')
    RESULTS['Best_Accuracy'].append(best_score)
    RESULTS['Best_LSTM_Units'].append(lstm)
    RESULTS['Best_Dropout'].append(round(dropout, 4))
    RESULTS['Best_LR'].append(round(lr, 6))
    RESULTS['Time_Seconds'].append(round(woa_time, 2))
    
    print(f"✅ WOA Best: Acc={best_score:.4f}, LSTM={lstm}, Dropout={dropout:.3f}, LR={lr:.6f}")
    
    # =============================================
    # 9. DIFFERENTIAL EVOLUTION (DE)
    # =============================================
    print("\n" + "─"*70)
    print("🧬 TASK 9: DIFFERENTIAL EVOLUTION (DE)")
    print("─"*70)
    print("Configuration: Population=5, F=0.8, CR=0.9, 3 iterations")
    print("Evolutionary algorithm with mutation & crossover\n")
    
    def de_fitness(solution):
        lstm, dropout, lr = decode_pso(solution)
        return fitness_function(lstm, dropout, lr)
    
    # DE algorithm - mutation and crossover
    n_population = 5
    n_iter = 3
    dim = 3
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover probability
    
    population = np.random.rand(n_population, dim)
    fitness_values = np.array([de_fitness(ind) for ind in population])
    
    best_idx = np.argmax(fitness_values)
    best_solution = population[best_idx].copy()
    best_score = fitness_values[best_idx]
    
    print(f"🎯 Initial best: {best_score:.4f}")
    start_time = time.time()
    
    de_pbar = tqdm(range(n_iter), desc="DE Generations")
    for iteration in de_pbar:
        for i in range(n_population):
            # Mutation: select 3 random distinct individuals
            indices = list(range(n_population))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            # Mutant vector
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, 0, 1)
            
            # Crossover
            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]
            
            # Selection
            trial_fitness = de_fitness(trial)
            if trial_fitness > fitness_values[i]:
                population[i] = trial
                fitness_values[i] = trial_fitness
                
                if trial_fitness > best_score:
                    best_solution = trial.copy()
                    best_score = trial_fitness
        
        de_pbar.set_postfix({'best': f'{best_score:.4f}', 'F': F, 'CR': CR})
        print(f"  ✅ Generation {iteration+1}/{n_iter}: Best = {best_score:.4f}")
    
    de_time = time.time() - start_time
    lstm, dropout, lr = decode_pso(best_solution)
    
    RESULTS['Algorithm'].append('DE')
    RESULTS['Best_Accuracy'].append(best_score)
    RESULTS['Best_LSTM_Units'].append(lstm)
    RESULTS['Best_Dropout'].append(round(dropout, 4))
    RESULTS['Best_LR'].append(round(lr, 6))
    RESULTS['Time_Seconds'].append(round(de_time, 2))
    
    print(f"✅ DE Best: Acc={best_score:.4f}, LSTM={lstm}, Dropout={dropout:.3f}, LR={lr:.6f}")
    
    # =============================================
    # 10. SIMULATED ANNEALING (SA)
    # =============================================
    print("\n" + "─"*70)
    print("❄️ TASK 10: SIMULATED ANNEALING (SA)")
    print("─"*70)
    print("Configuration: T=1.0→0.0001, Cooling rate=0.9, 8 iter/temp")
    print("Thermal annealing with probabilistic acceptance\n")
    
    def sa_fitness(solution):
        lstm, dropout, lr = decode_pso(solution)
        return fitness_function(lstm, dropout, lr)
    
    # SA algorithm - thermal annealing process
    current = np.random.rand(3)
    current_score = sa_fitness(current)
    best_solution = current.copy()
    best_score = current_score
    
    T = 1.0  # Initial temperature
    T_min = 0.0001  # Minimum temperature
    alpha = 0.9  # Cooling rate
    n_iter = 8  # Iterations per temperature
    
    print(f"🎯 Initial: {best_score:.4f}, Temperature: {T:.4f}")
    start_time = time.time()
    
    
    iteration_count = 0
    temp_stage = 0
    with tqdm(total=24, desc="SA Progress") as sa_pbar:
        while T > T_min and iteration_count < 24:
            for _ in range(n_iter):
                # Generate neighbor by perturbing current solution
                neighbor = current + np.random.randn(3) * 0.1
                neighbor = np.clip(neighbor, 0, 1)
                
                neighbor_score = sa_fitness(neighbor)
                
                # Acceptance criteria
                delta = neighbor_score - current_score
                
                if delta > 0:  # Better solution
                    current = neighbor
                    current_score = neighbor_score
                    
                    if current_score > best_score:
                        best_solution = current.copy()
                        best_score = current_score
                else:  # Worse solution - accept with probability
                    acceptance_prob = np.exp(delta / T)
                    if np.random.rand() < acceptance_prob:
                        current = neighbor
                        current_score = neighbor_score
                
                iteration_count += 1
                sa_pbar.update(1)
                sa_pbar.set_postfix({'best': f'{best_score:.4f}', 'T': f'{T:.4f}'})
            
            temp_stage += 1
            print(f"  ❄️ Stage {temp_stage}: T={T:.4f}, Best = {best_score:.4f}")
            T *= alpha  # Cool down

    
    sa_time = time.time() - start_time
    lstm, dropout, lr = decode_pso(best_solution)
    
    RESULTS['Algorithm'].append('SA')
    RESULTS['Best_Accuracy'].append(best_score)
    RESULTS['Best_LSTM_Units'].append(lstm)
    RESULTS['Best_Dropout'].append(round(dropout, 4))
    RESULTS['Best_LR'].append(round(lr, 6))
    RESULTS['Time_Seconds'].append(round(sa_time, 2))
    
    print(f"✅ SA Best: Acc={best_score:.4f}, LSTM={lstm}, Dropout={dropout:.3f}, LR={lr:.6f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(RESULTS)
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # =============================================
    # VISUALIZATION - ALGORITHM COMPARISON
    # =============================================
    print("\n--- Generating Comparison Visualizations ---")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(16, 10))
    
    # --- Subplot 1: Accuracy Comparison (Bar Chart) ---
    ax1 = plt.subplot(2, 3, 1)
    colors = sns.color_palette("husl", len(results_df))
    bars = ax1.bar(results_df['Algorithm'], results_df['Best_Accuracy'], color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison Across Algorithms', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(results_df['Best_Accuracy']) * 0.95, max(results_df['Best_Accuracy']) * 1.02])
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Highlight best algorithm
    best_idx = results_df['Best_Accuracy'].idxmax()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # --- Subplot 2: Execution Time Comparison ---
    ax2 = plt.subplot(2, 3, 2)
    ax2.barh(results_df['Algorithm'], results_df['Time_Seconds'], color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Algorithm', fontsize=12, fontweight='bold')
    ax2.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (algo, time_val) in enumerate(zip(results_df['Algorithm'], results_df['Time_Seconds'])):
        ax2.text(time_val, i, f' {time_val:.1f}s', va='center', fontsize=9)
    
    # --- Subplot 3: Accuracy vs Time (Scatter Plot) ---
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(results_df['Time_Seconds'], results_df['Best_Accuracy'], 
                         s=200, c=range(len(results_df)), cmap='viridis', 
                         edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Annotate points
    for i, algo in enumerate(results_df['Algorithm']):
        ax3.annotate(algo, 
                    (results_df['Time_Seconds'].iloc[i], results_df['Best_Accuracy'].iloc[i]),
                    textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)
    
    ax3.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy vs Execution Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # --- Subplot 4: Hyperparameter Distribution - LSTM Units ---
    ax4 = plt.subplot(2, 3, 4)
    lstm_counts = results_df['Best_LSTM_Units'].value_counts().sort_index()
    ax4.bar(lstm_counts.index.astype(str), lstm_counts.values, color='skyblue', edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('LSTM Units', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Optimal LSTM Units', fontsize=14, fontweight='bold')
    
    # --- Subplot 5: Hyperparameter Heatmap ---
    ax5 = plt.subplot(2, 3, 5)
    
    # Normalize hyperparameters for heatmap
    heatmap_data = results_df[['Best_LSTM_Units', 'Best_Dropout', 'Best_LR']].copy()
    heatmap_data['Best_LSTM_Units'] = heatmap_data['Best_LSTM_Units'] / heatmap_data['Best_LSTM_Units'].max()
    heatmap_data['Best_Dropout'] = heatmap_data['Best_Dropout']
    heatmap_data['Best_LR'] = heatmap_data['Best_LR'] / heatmap_data['Best_LR'].max()
    heatmap_data.index = results_df['Algorithm']
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=ax5, cbar_kws={'label': 'Normalized Value'}, linewidths=1)
    ax5.set_title('Hyperparameter Heatmap (Normalized)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Hyperparameter', fontsize=12, fontweight='bold')
    
    # --- Subplot 6: Summary Statistics Table ---
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_stats = [
        ['Metric', 'Value'],
        ['Best Algorithm', results_df.loc[results_df['Best_Accuracy'].idxmax(), 'Algorithm']],
        ['Best Accuracy', f"{results_df['Best_Accuracy'].max():.4f}"],
        ['Mean Accuracy', f"{results_df['Best_Accuracy'].mean():.4f}"],
        ['Std Accuracy', f"{results_df['Best_Accuracy'].std():.4f}"],
        ['Fastest Algorithm', results_df.loc[results_df['Time_Seconds'].idxmin(), 'Algorithm']],
        ['Total Time', f"{results_df['Time_Seconds'].sum():.1f}s"],
        ['Mean Time', f"{results_df['Time_Seconds'].mean():.1f}s"],
    ]
    
    table = ax6.table(cellText=summary_stats, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style best metrics
    table[(1, 0)].set_facecolor('#FFD700')  # Best algorithm
    table[(2, 0)].set_facecolor('#FFD700')  # Best accuracy
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('Phase 1: Metaheuristic Algorithms Comparison', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    viz_path = '/tmp/phase1_comparison.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {viz_path}")
    
    # Close to free memory
    plt.close()
    
    # Save results
    results_df.to_csv('/tmp/phase1_results.csv', index=False)
    print("✅ Results CSV saved to /tmp/phase1_results.csv")
    
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print(f"📊 Results CSV: /tmp/phase1_results.csv")
    print(f"📈 Visualization: {viz_path}")
    print("="*70)
    
    return results_df.to_dict()



@app.local_entrypoint()
def main():
    """Local entrypoint to trigger Modal function"""
    print("Starting Phase 1 training on Modal.com...")
    results = run_phase1_training.remote()
    print("\n✅ Training completed!")
    print(results)
