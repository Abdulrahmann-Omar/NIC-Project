"""
3D Algorithm Visualizations for Nature-Inspired Computation
============================================================
Interactive 3D visualizations of PSO, GWO, and other metaheuristics.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# TEST FUNCTIONS (Optimization Landscapes)
# =============================================================================

def sphere_function(x, y):
    """Simple sphere function: f(x,y) = x^2 + y^2"""
    return x**2 + y**2

def rastrigin_function(x, y):
    """Rastrigin function - many local minima"""
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def rosenbrock_function(x, y):
    """Rosenbrock function - banana-shaped valley"""
    return (1 - x)**2 + 100 * (y - x**2)**2

# =============================================================================
# PSO 3D VISUALIZATION
# =============================================================================

def create_pso_animation(n_particles=20, n_iterations=30, bounds=(-5, 5)):
    """Create 3D PSO animation showing particles converging to optimum."""
    
    # Initialize particles
    np.random.seed(42)
    positions = np.random.uniform(bounds[0], bounds[1], (n_particles, 2))
    velocities = np.random.uniform(-1, 1, (n_particles, 2))
    
    # PSO parameters
    w = 0.7  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    # Personal and global best
    p_best = positions.copy()
    p_best_scores = np.array([sphere_function(p[0], p[1]) for p in positions])
    g_best = p_best[np.argmin(p_best_scores)]
    
    # Store history
    history = [positions.copy()]
    
    for _ in range(n_iterations):
        # Update velocities
        r1, r2 = np.random.rand(2)
        velocities = (w * velocities + 
                      c1 * r1 * (p_best - positions) + 
                      c2 * r2 * (g_best - positions))
        
        # Update positions
        positions = positions + velocities
        positions = np.clip(positions, bounds[0], bounds[1])
        
        # Update personal best
        scores = np.array([sphere_function(p[0], p[1]) for p in positions])
        improved = scores < p_best_scores
        p_best[improved] = positions[improved]
        p_best_scores[improved] = scores[improved]
        
        # Update global best
        if np.min(scores) < sphere_function(g_best[0], g_best[1]):
            g_best = positions[np.argmin(scores)]
        
        history.append(positions.copy())
    
    # Create 3D surface for landscape
    x_range = np.linspace(bounds[0], bounds[1], 50)
    y_range = np.linspace(bounds[0], bounds[1], 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = sphere_function(X, Y)
    
    # Create animation frames
    frames = []
    for i, pos in enumerate(history):
        z_particles = np.array([sphere_function(p[0], p[1]) for p in pos])
        frames.append(go.Frame(
            data=[
                go.Surface(x=X, y=Y, z=Z, opacity=0.6, showscale=False, 
                          colorscale='Viridis', name='Landscape'),
                go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=z_particles + 0.5,
                            mode='markers', marker=dict(size=6, color='red'),
                            name='Particles')
            ],
            name=str(i)
        ))
    
    # Initial figure
    z_init = np.array([sphere_function(p[0], p[1]) for p in history[0]])
    fig = go.Figure(
        data=[
            go.Surface(x=X, y=Y, z=Z, opacity=0.6, showscale=False, 
                      colorscale='Viridis', name='Landscape'),
            go.Scatter3d(x=history[0][:, 0], y=history[0][:, 1], z=z_init + 0.5,
                        mode='markers', marker=dict(size=6, color='red'),
                        name='Particles')
        ],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title="PSO: Particle Swarm Optimization in 3D",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y", 
            zaxis_title="f(x,y)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0,
            x=0.1,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 100}, "transition": {"duration": 50}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], {"frame": {"duration": 100}, "mode": "immediate"}],
                       label=str(i), method="animate") for i, f in enumerate(frames)],
            x=0.1, len=0.8, y=0,
            currentvalue=dict(prefix="Iteration: ")
        )]
    )
    
    return fig

# =============================================================================
# GWO 3D VISUALIZATION
# =============================================================================

def create_gwo_animation(n_wolves=15, n_iterations=25, bounds=(-5, 5)):
    """Create 3D GWO animation showing wolf pack hunting."""
    
    np.random.seed(42)
    positions = np.random.uniform(bounds[0], bounds[1], (n_wolves, 2))
    
    # GWO parameter
    a_init = 2.0
    
    history = [positions.copy()]
    alpha_history = []
    
    for t in range(n_iterations):
        # Calculate fitness
        fitness = np.array([sphere_function(p[0], p[1]) for p in positions])
        
        # Get alpha, beta, delta (3 best)
        sorted_idx = np.argsort(fitness)
        alpha = positions[sorted_idx[0]]
        beta = positions[sorted_idx[1]]
        delta = positions[sorted_idx[2]]
        
        alpha_history.append(alpha.copy())
        
        # Update a
        a = a_init - t * (a_init / n_iterations)
        
        # Update positions
        new_positions = []
        for i, pos in enumerate(positions):
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha - pos)
            X1 = alpha - A1 * D_alpha
            
            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta - pos)
            X2 = beta - A2 * D_beta
            
            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta - pos)
            X3 = delta - A3 * D_delta
            
            new_pos = (X1 + X2 + X3) / 3
            new_pos = np.clip(new_pos, bounds[0], bounds[1])
            new_positions.append(new_pos)
        
        positions = np.array(new_positions)
        history.append(positions.copy())
    
    # Create surface
    x_range = np.linspace(bounds[0], bounds[1], 50)
    y_range = np.linspace(bounds[0], bounds[1], 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = sphere_function(X, Y)
    
    # Create frames
    frames = []
    for i, pos in enumerate(history):
        z_wolves = np.array([sphere_function(p[0], p[1]) for p in pos])
        fitness = z_wolves
        sorted_idx = np.argsort(fitness)
        
        # Color wolves by rank
        colors = ['gray'] * len(pos)
        if len(sorted_idx) >= 3:
            colors[sorted_idx[0]] = 'red'    # Alpha
            colors[sorted_idx[1]] = 'orange' # Beta
            colors[sorted_idx[2]] = 'yellow' # Delta
        
        frames.append(go.Frame(
            data=[
                go.Surface(x=X, y=Y, z=Z, opacity=0.5, showscale=False,
                          colorscale='Greens', name='Prey Landscape'),
                go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=z_wolves + 0.5,
                            mode='markers', 
                            marker=dict(size=8, color=colors),
                            name='Wolves')
            ],
            name=str(i)
        ))
    
    # Initial figure
    z_init = np.array([sphere_function(p[0], p[1]) for p in history[0]])
    fig = go.Figure(
        data=[
            go.Surface(x=X, y=Y, z=Z, opacity=0.5, showscale=False,
                      colorscale='Greens', name='Prey Landscape'),
            go.Scatter3d(x=history[0][:, 0], y=history[0][:, 1], z=z_init + 0.5,
                        mode='markers', marker=dict(size=8, color='gray'),
                        name='Wolves')
        ],
        frames=frames
    )
    
    fig.update_layout(
        title="GWO: Grey Wolf Optimizer - Pack Hunting",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="f(x,y)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.1,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 150}, "transition": {"duration": 50}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], {"frame": {"duration": 150}, "mode": "immediate"}],
                       label=str(i), method="animate") for i, f in enumerate(frames)],
            x=0.1, len=0.8, y=0,
            currentvalue=dict(prefix="Iteration: ")
        )],
        annotations=[
            dict(text="Red=Alpha, Orange=Beta, Yellow=Delta", x=0.5, y=1.1,
                 xref="paper", yref="paper", showarrow=False)
        ]
    )
    
    return fig

# =============================================================================
# SEARCH SPACE COMPARISON
# =============================================================================

def create_search_space_comparison():
    """Create side-by-side 3D comparison of different test functions."""
    
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=['Sphere (Unimodal)', 'Rastrigin (Multimodal)', 'Rosenbrock (Valley)']
    )
    
    # Sphere
    Z1 = sphere_function(X, Y)
    fig.add_trace(go.Surface(z=Z1, x=X, y=Y, colorscale='Blues', showscale=False), row=1, col=1)
    
    # Rastrigin (scaled)
    x2 = np.linspace(-5.12, 5.12, 100)
    y2 = np.linspace(-5.12, 5.12, 100)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = rastrigin_function(X2, Y2)
    fig.add_trace(go.Surface(z=Z2, x=X2, y=Y2, colorscale='Reds', showscale=False), row=1, col=2)
    
    # Rosenbrock (scaled)
    x3 = np.linspace(-2, 2, 100)
    y3 = np.linspace(-1, 3, 100)
    X3, Y3 = np.meshgrid(x3, y3)
    Z3 = np.log(rosenbrock_function(X3, Y3) + 1)  # Log scale for visibility
    fig.add_trace(go.Surface(z=Z3, x=X3, y=Y3, colorscale='Greens', showscale=False), row=1, col=3)
    
    fig.update_layout(
        title="Optimization Landscape Comparison",
        height=500
    )
    
    return fig

# =============================================================================
# MAIN - Generate and save
# =============================================================================

if __name__ == "__main__":
    print("Generating 3D Algorithm Visualizations...")
    
    # Generate PSO animation
    pso_fig = create_pso_animation()
    pso_fig.write_html("pso_3d_animation.html")
    print("Created: pso_3d_animation.html")
    
    # Generate GWO animation
    gwo_fig = create_gwo_animation()
    gwo_fig.write_html("gwo_3d_animation.html")
    print("Created: gwo_3d_animation.html")
    
    # Generate search space comparison
    compare_fig = create_search_space_comparison()
    compare_fig.write_html("search_space_comparison.html")
    print("Created: search_space_comparison.html")
    
    print("\nAll visualizations generated! Open HTML files in browser.")
