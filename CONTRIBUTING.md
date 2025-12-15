# Contributing to Nature-Inspired Computation

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** and **description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, Python version, Modal version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case**: Why is this enhancement useful?
- **Proposed solution**: How should it work?
- **Alternatives considered**: What other approaches did you think about?

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests
3. Ensure the test suite passes
4. Make sure your code follows the existing style
5. Write a clear commit message

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/nature-inspired-computation.git
cd nature-inspired-computation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Use `black` for code formatting
- Use `flake8` for linting
- Maximum line length: 88 characters
- Use type hints where possible

```bash
# Format code
black src/

# Check style
flake8 src/

# Type checking
mypy src/
```

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Examples:
```
Add PSO convergence plot
Fix checkpoint loading bug in Phase 2
Update README with installation instructions
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_algorithms.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update relevant documentation in `docs/`

## Code Review Process

1. Maintainers will review your PR within 3-5 days
2. Address any requested changes
3. Once approved, maintainers will merge your PR

## Community

- Be respectful and inclusive
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Help others in issues and discussions

## Questions?

Feel free to open an issue with the `question` label!

---

Thank you for contributing! 🚀
