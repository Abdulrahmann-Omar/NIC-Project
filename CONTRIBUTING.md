# 🤝 Contributing to NIC Project

Thank you for your interest in contributing! This document provides guidelines and steps for contributing.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Gracefully accept constructive criticism
- Focus on what is best for the community

### Unacceptable Behavior

- Trolling, insulting comments, or personal attacks
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

---

## How to Contribute

### 🐛 Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Open a new issue** using the bug report template
3. **Include**:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### 💡 Suggesting Features

1. **Open a feature request issue**
2. **Describe**:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternatives you've considered

### 📖 Improving Documentation

1. Fix typos, clarify explanations, add examples
2. No issue needed for small fixes
3. Submit PR directly

### 🧬 Adding New Algorithms

1. Open an issue first to discuss
2. Follow the algorithm template (see below)
3. Include tests and documentation

---

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- pip

### Setup Steps

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/NIC-Project.git
cd NIC-Project

# 3. Add upstream remote
git remote add upstream https://github.com/Abdulrahmann-Omar/NIC-Project.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# 6. Verify setup
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
```

### Development Tools

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Setup pre-commit hooks
pre-commit install
```

---

## Pull Request Process

### 1. Create Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout master
git merge upstream/master

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add Grey Wolf Optimizer implementation

- Implement GWO class with encircling and hunting mechanisms
- Add unit tests for convergence
- Update README with usage example

Closes #123"
```

#### Commit Message Format

```
<type>: <short description>

<body - optional>

<footer - optional>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open PR on GitHub with:
- Clear title and description
- Reference to related issue(s)
- Screenshots/GIFs if UI changes
- Test results

### 5. Review Process

- Maintainers will review within 2-3 days
- Address feedback promptly
- Once approved, PR will be merged

---

## Coding Standards

### Python Style

Follow **PEP 8** with these specifics:

```python
# Line length: 88 characters (Black default)
# Use type hints for public functions
# Use docstrings for all classes and public methods

def optimize(
    self,
    objective: Callable[[Dict[str, Any]], float],
    search_space: Dict[str, Any],
    iterations: int = 15
) -> Tuple[Dict[str, Any], float]:
    """
    Run optimization to find best parameters.
    
    Args:
        objective: Function that takes parameters and returns score.
        search_space: Dictionary defining parameter bounds.
        iterations: Number of optimization iterations.
        
    Returns:
        Tuple of (best_parameters, best_score).
        
    Raises:
        ValueError: If search_space is empty.
        
    Example:
        >>> pso = PSO(n_particles=10)
        >>> best = pso.optimize(my_func, {'x': (-5, 5)})
    """
```

### Code Formatting

```bash
# Format code with Black
black src/ dashboard/ visualizations/

# Check with flake8
flake8 src/ dashboard/ visualizations/

# Type check with mypy
mypy src/ --ignore-missing-imports
```

### Imports Order

```python
# 1. Standard library
import os
import sys
from typing import Dict, List, Tuple

# 2. Third-party
import numpy as np
import tensorflow as tf
from scipy import stats

# 3. Local
from .base import BaseOptimizer
from .utils import clip_bounds
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pso.py

# Run specific test
pytest tests/test_pso.py::test_convergence
```

### Writing Tests

```python
# tests/test_pso.py
import pytest
from algorithms.pso import PSO

class TestPSO:
    """Tests for PSO optimizer."""
    
    def test_initialization(self):
        """Test PSO initializes with correct defaults."""
        pso = PSO()
        assert pso.n_particles == 10
        assert pso.c1 == 2.0
        assert pso.c2 == 2.0
    
    def test_optimization_converges(self):
        """Test PSO finds approximate optimum."""
        def sphere(params):
            return -(params['x']**2 + params['y']**2)
        
        pso = PSO(n_particles=10)
        best_params, best_score = pso.optimize(
            sphere, 
            {'x': (-5, 5), 'y': (-5, 5)},
            iterations=20
        )
        
        assert abs(best_params['x']) < 0.5
        assert abs(best_params['y']) < 0.5
    
    @pytest.mark.parametrize("n_particles", [5, 10, 20])
    def test_different_populations(self, n_particles):
        """Test PSO works with different population sizes."""
        pso = PSO(n_particles=n_particles)
        assert pso.n_particles == n_particles
```

### Test Coverage Requirements

- New features: **80%+ coverage**
- Bug fixes: Test that reproduces bug
- Algorithms: Convergence tests required

---

## Documentation Standards

### Docstring Format (Google Style)

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """Short description of function.
    
    Longer description if needed. Can span multiple lines
    and include more context.
    
    Args:
        param1: Description of first parameter.
        param2: Description of second parameter with default.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param1 is negative.
        TypeError: When param2 is not a string.
        
    Example:
        >>> result = function_name(42, "hello")
        >>> print(result)
        True
        
    Note:
        Additional notes or warnings.
    """
```

### README Updates

When adding features, update:
1. Feature list in README
2. Algorithm table if adding optimizer
3. Quick start if changing usage

### Changelog Updates

Add entry to `CHANGELOG.md`:

```markdown
## [Unreleased]

### Added
- New Grey Wolf Optimizer implementation (#123)

### Changed
- Improved PSO convergence speed by 20%

### Fixed
- Memory leak in Tabu Search (#456)
```

---

## Algorithm Template

When adding a new algorithm, use this template:

```python
# algorithms/new_algorithm.py
"""
New Algorithm Implementation
============================

Based on: Author, Year, "Paper Title", Journal

Implementation of the New Algorithm for optimization.
"""

from typing import Any, Callable, Dict, Tuple
import numpy as np

from .base import BaseOptimizer


class NewAlgorithm(BaseOptimizer):
    """
    New Algorithm optimizer.
    
    Based on the natural behavior of [inspiration].
    Proposed by [Author] in [Year].
    
    Attributes:
        population_size: Number of agents.
        param1: Description of parameter.
        
    Example:
        >>> algo = NewAlgorithm(population_size=20)
        >>> best = algo.optimize(objective, search_space)
    """
    
    def __init__(
        self,
        population_size: int = 20,
        param1: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize New Algorithm.
        
        Args:
            population_size: Number of search agents.
            param1: Controls exploration/exploitation balance.
            seed: Random seed for reproducibility.
        """
        super().__init__(seed=seed)
        self.population_size = population_size
        self.param1 = param1
    
    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Any],
        iterations: int = 30
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run optimization.
        
        Args:
            objective: Function to maximize.
            search_space: Parameter bounds.
            iterations: Number of iterations.
            
        Returns:
            Tuple of best parameters and best score.
        """
        # Implementation here
        pass
```

---

## Questions?

- 💬 [Start a discussion](https://github.com/Abdulrahmann-Omar/NIC-Project/discussions)
- 📧 Contact: abdu.omar.muhammad@gmail.com

Thank you for contributing! 🎉
