# 📚 Tutorials

Step-by-step guides to get the most out of the NIC Project.

---

## Table of Contents

1. [Tutorial 1: Your First Optimization](#tutorial-1-your-first-optimization)
2. [Tutorial 2: Comparing Algorithms](#tutorial-2-comparing-algorithms)
3. [Tutorial 3: Meta-Optimization](#tutorial-3-meta-optimization)
4. [Tutorial 4: XAI Integration](#tutorial-4-xai-integration)
5. [Tutorial 5: 3D Visualizations](#tutorial-5-3d-visualizations)
6. [Tutorial 6: Custom Models](#tutorial-6-custom-models)
7. [Tutorial 7: Dashboard Customization](#tutorial-7-dashboard-customization)

---

## Tutorial 1: Your First Optimization

**Goal**: Run PSO to optimize a BiLSTM model in 5 minutes.

### Step 1: Setup

```bash
git clone https://github.com/Abdulrahmann-Omar/NIC-Project.git
cd NIC-Project
pip install -r requirements.txt
```

### Step 2: Define Your Search Space

```python
# config.py
SEARCH_SPACE = {
    'lstm_units': [32, 64, 128],
    'dropout': (0.2, 0.5),        # Continuous range
    'learning_rate': (0.0001, 0.01),
    'batch_size': [32, 64, 128]
}
```

### Step 3: Run PSO

```python
from algorithms.pso import PSO
from models.bilstm import create_model

# Define objective function
def objective(params):
    model = create_model(
        lstm_units=params['lstm_units'],
        dropout=params['dropout'],
        learning_rate=params['learning_rate']
    )
    history = model.fit(X_train, y_train, epochs=5, verbose=0)
    return history.history['val_accuracy'][-1]

# Initialize and run PSO
pso = PSO(
    n_particles=10,
    c1=2.0,      # Cognitive coefficient
    c2=2.0,      # Social coefficient
    w=0.7        # Inertia
)

best_params = pso.optimize(objective, SEARCH_SPACE, iterations=15)
print(f"Best parameters: {best_params}")
```

### Step 4: Evaluate Results

```python
# Train final model with best parameters
final_model = create_model(**best_params)
final_model.fit(X_train, y_train, epochs=10)

# Evaluate
accuracy = final_model.evaluate(X_test, y_test)
print(f"Final accuracy: {accuracy[1]:.4f}")
```

### Expected Output

```
Iteration 1/15: Best accuracy = 0.6823
Iteration 5/15: Best accuracy = 0.7156
Iteration 10/15: Best accuracy = 0.7234
Iteration 15/15: Best accuracy = 0.7276

Best parameters: {'lstm_units': 64, 'dropout': 0.28, 'learning_rate': 0.0053}
Final accuracy: 0.7276
```

💡 **Pro Tip**: Start with fewer particles (5-10) and more iterations (15-20) for better convergence.

---

## Tutorial 2: Comparing Algorithms

**Goal**: Run multiple algorithms and compare their performance.

### Step 1: Define Algorithms to Compare

```python
from algorithms import PSO, GWO, WOA, TabuSearch, SimulatedAnnealing

algorithms = [
    ('PSO', PSO(n_particles=10)),
    ('GWO', GWO(n_wolves=10)),
    ('WOA', WOA(n_whales=10)),
    ('Tabu', TabuSearch(tabu_size=7)),
    ('SA', SimulatedAnnealing(T_init=100))
]
```

### Step 2: Run Benchmark

```python
import time
import pandas as pd

results = []

for name, algo in algorithms:
    print(f"\n{'='*50}")
    print(f"Running {name}...")
    
    start_time = time.time()
    best_params, best_score = algo.optimize(objective, SEARCH_SPACE, iterations=15)
    runtime = time.time() - start_time
    
    results.append({
        'Algorithm': name,
        'Best_Accuracy': best_score,
        'Runtime_Seconds': runtime,
        **best_params
    })

# Create comparison table
df = pd.DataFrame(results)
df = df.sort_values('Best_Accuracy', ascending=False)
print("\n" + "="*50)
print("RESULTS:")
print(df.to_string(index=False))
```

### Step 3: Visualize Comparison

```python
import plotly.express as px

fig = px.bar(
    df,
    x='Algorithm',
    y='Best_Accuracy',
    color='Best_Accuracy',
    color_continuous_scale='Viridis',
    title='Algorithm Accuracy Comparison'
)
fig.show()
```

### Step 4: Statistical Significance

```python
from scipy import stats

# Paired t-test between best and second-best
best_results = [0.7343, 0.7341, 0.7339]  # Multiple runs of Tabu
second_results = [0.7294, 0.7290, 0.7288]  # Multiple runs of WOA

t_stat, p_value = stats.ttest_rel(best_results, second_results)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")
```

---

## Tutorial 3: Meta-Optimization

**Goal**: Use Cuckoo Search to optimize PSO's own parameters.

### Concept

Instead of manually tuning PSO's `c1`, `c2`, and `w` parameters, we use another algorithm (Cuckoo Search) to find optimal values.

### Step 1: Define PSO Parameter Space

```python
PSO_PARAMS = {
    'c1': (1.0, 3.0),      # Cognitive coefficient
    'c2': (1.0, 3.0),      # Social coefficient
    'w': (0.3, 0.9),       # Inertia weight
    'w_decay': (0.9, 0.99) # Decay rate
}
```

### Step 2: Create Meta-Objective

```python
def meta_objective(pso_params):
    """
    Run PSO with given parameters and return its best performance.
    """
    pso = PSO(
        n_particles=10,
        c1=pso_params['c1'],
        c2=pso_params['c2'],
        w=pso_params['w']
    )
    
    # Run PSO to optimize the actual model
    _, best_accuracy = pso.optimize(
        objective=model_objective,
        search_space=MODEL_SEARCH_SPACE,
        iterations=10
    )
    
    return best_accuracy
```

### Step 3: Run Cuckoo Search

```python
from algorithms.cuckoo import CuckooSearch

cs = CuckooSearch(
    n_nests=15,
    pa=0.25,  # Discovery rate
    alpha=1.0  # Step size
)

best_pso_params = cs.optimize(meta_objective, PSO_PARAMS, iterations=20)
print(f"Optimal PSO parameters: {best_pso_params}")
```

### Step 4: Compare Results

```python
# Default PSO
default_pso = PSO(c1=2.0, c2=2.0, w=0.7)
default_score = default_pso.optimize(model_objective, MODEL_SEARCH_SPACE)[1]

# Optimized PSO
optimized_pso = PSO(**best_pso_params)
optimized_score = optimized_pso.optimize(model_objective, MODEL_SEARCH_SPACE)[1]

improvement = (optimized_score - default_score) / default_score * 100
print(f"Default PSO accuracy: {default_score:.4f}")
print(f"Optimized PSO accuracy: {optimized_score:.4f}")
print(f"Improvement: {improvement:.2f}%")
```

### Expected Results

```
Optimal PSO parameters: {'c1': 1.8, 'c2': 2.1, 'w': 0.6}
Default PSO accuracy: 0.7276
Optimized PSO accuracy: 0.7480
Improvement: 2.04%
```

💡 **Key Insight**: Meta-optimization provides significant gains with minimal extra effort!

---

## Tutorial 4: XAI Integration

**Goal**: Generate explanations using optimized SHAP, LIME, and Grad-CAM.

### Step 1: Install XAI Libraries

```bash
pip install shap lime tf-explain
```

### Step 2: Generate SHAP Explanations

```python
import shap

# Create explainer with optimized parameters
explainer = shap.DeepExplainer(
    model,
    X_train[:100]  # Background samples
)

# Explain a prediction
sample = X_test[0:1]
shap_values = explainer.shap_values(sample)

# Visualize
shap.summary_plot(shap_values, feature_names=vocab)
```

### Step 3: Generate LIME Explanations

```python
from lime.lime_text import LimeTextExplainer

# Optimized LIME parameters
explainer = LimeTextExplainer(
    class_names=['Negative', 'Positive'],
    kernel_width=1.2,  # Optimized by Harmony Search
    num_features=12
)

# Explain
explanation = explainer.explain_instance(
    text,
    model.predict_proba,
    num_samples=150  # Optimized by GA
)

explanation.show_in_notebook()
```

### Step 4: Generate Grad-CAM

```python
from tf_explain.core.grad_cam import GradCAM

explainer = GradCAM()

# Optimized layer index
grid = explainer.explain(
    validation_data=(sample, label),
    model=model,
    class_index=1,
    layer_name='bidirectional_1'  # Optimized layer
)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(grid)
plt.title("Grad-CAM Attention")
plt.show()
```

---

## Tutorial 5: 3D Visualizations

**Goal**: Create interactive 3D animations of algorithm behavior.

### Step 1: Generate PSO Animation

```python
from visualizations.algorithm_3d import create_pso_animation

# Create animation
fig = create_pso_animation(
    n_particles=15,
    n_iterations=25,
    bounds=(-5, 5)
)

# Save as HTML
fig.write_html("pso_animation.html")

# Or display inline
fig.show()
```

### Step 2: Generate GWO Animation

```python
from visualizations.algorithm_3d import create_gwo_animation

fig = create_gwo_animation(
    n_wolves=12,
    n_iterations=20,
    bounds=(-5, 5)
)

fig.write_html("gwo_animation.html")
```

### Step 3: Compare Search Spaces

```python
from visualizations.algorithm_3d import create_search_space_comparison

fig = create_search_space_comparison()
fig.write_html("search_spaces.html")
```

### Step 4: Embed in Dashboard

The visualizations are automatically available in the Streamlit dashboard under "3D Visualizations" page.

---

## Tutorial 6: Custom Models

**Goal**: Use your own model with the optimization framework.

### Step 1: Define Your Model Builder

```python
def create_custom_model(lstm_units, dropout, learning_rate):
    """
    Your custom model architecture.
    """
    model = tf.keras.Sequential([
        # Your layers here
        tf.keras.layers.Embedding(10000, 128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### Step 2: Define Objective Function

```python
def custom_objective(params):
    """
    Train and evaluate model with given parameters.
    Returns validation accuracy (to maximize).
    """
    model = create_custom_model(
        lstm_units=int(params['lstm_units']),
        dropout=params['dropout'],
        learning_rate=params['learning_rate']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    return max(history.history['val_accuracy'])
```

### Step 3: Run Optimization

```python
pso = PSO(n_particles=10)
best_params = pso.optimize(custom_objective, SEARCH_SPACE, iterations=15)
```

---

## Tutorial 7: Dashboard Customization

**Goal**: Modify the Streamlit dashboard for your needs.

### Step 1: Add a New Page

Edit `dashboard/app.py`:

```python
# Add to navigation
page = st.sidebar.selectbox("Navigation", [
    "Home",
    "Live Prediction",
    "Algorithm Comparison",
    "My Custom Page"  # Add this
])

# Add page content
elif page == "My Custom Page":
    st.header("My Custom Analysis")
    
    # Your content here
    st.write("Add your custom visualizations and analysis")
```

### Step 2: Add Custom Visualization

```python
import plotly.graph_objects as go

# Create custom chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results['iteration'],
    y=results['accuracy'],
    mode='lines+markers',
    name='Convergence'
))
fig.update_layout(title='My Custom Chart')
st.plotly_chart(fig)
```

### Step 3: Deploy Changes

```bash
git add .
git commit -m "Add custom dashboard page"
git push
```

Streamlit Cloud will automatically redeploy.

---

## 🎓 Next Steps

After completing these tutorials:

1. **Read the [Algorithm Reference](ALGORITHMS.md)** for deep dives
2. **Check the [API Documentation](API.md)** for all available options
3. **Explore the [Medium Article](https://medium.com/@abdu.omar.muhammad/introduction-the-hyperparameter-tuning-nightmare-e9f41d69b5ed)** for insights
4. **Try the [Live Demo](https://nic-project-2abdu.streamlit.app)** to see everything in action

---

## 🆘 Need Help?

- **Bug?** [Open an issue](https://github.com/Abdulrahmann-Omar/NIC-Project/issues)
- **Question?** [Start a discussion](https://github.com/Abdulrahmann-Omar/NIC-Project/discussions)
- **Feature request?** [Submit a PR](CONTRIBUTING.md)
