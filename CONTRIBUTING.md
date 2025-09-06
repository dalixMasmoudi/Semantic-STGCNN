# Contributing to Semantic-STGCNN

We welcome contributions to the Semantic-STGCNN project! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Semantic-STGCNN.git
   cd Semantic-STGCNN
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

4. **Run code quality checks**:
   ```bash
   black semantic_stgcnn/
   flake8 semantic_stgcnn/
   mypy semantic_stgcnn/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## 📝 Coding Standards

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

### Code Guidelines

1. **Follow PEP 8** style guidelines
2. **Use type hints** for all function parameters and return values
3. **Write docstrings** for all public functions and classes
4. **Keep functions small** and focused on a single responsibility
5. **Use meaningful variable names**

### Example Code Style

```python
from typing import Tuple, Optional
import torch
import torch.nn as nn


class ExampleLayer(nn.Module):
    """
    Example neural network layer with proper documentation.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        dropout: Dropout probability (default: 0.1)
        
    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, seq_len, output_dim)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 dropout: float = 0.1):
        super(ExampleLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        x = self.linear(x)
        x = self.dropout(x)
        return x
```

## 🧪 Testing

### Writing Tests

- Write tests for all new functionality
- Use pytest for testing framework
- Aim for high test coverage (>90%)
- Include both unit tests and integration tests

### Test Structure

```python
import pytest
import torch
from semantic_stgcnn.models import SemanticSTGCNN


class TestSemanticSTGCNN:
    """Test suite for SemanticSTGCNN model."""
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = SemanticSTGCNN()
        assert model is not None
        
    def test_forward_pass(self):
        """Test model forward pass with sample input."""
        model = SemanticSTGCNN()
        
        # Create sample input
        batch_size, seq_len, num_nodes = 2, 8, 4
        input_feat = 21
        
        V = torch.randn(batch_size, input_feat, seq_len, num_nodes)
        A = torch.randn(seq_len, num_nodes, num_nodes)
        
        # Forward pass
        output = model(V, A)
        
        # Check output shape
        expected_shape = (batch_size, 12, num_nodes, 2)  # pred_seq_len=12, output_feat=2
        assert output.shape == expected_shape
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantic_stgcnn --cov-report=html

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching pattern
pytest -k "test_model" -v
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def calculate_metrics(predictions: torch.Tensor, 
                     targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate evaluation metrics for trajectory predictions.
    
    Args:
        predictions: Predicted trajectories of shape (batch, seq_len, num_peds, 2)
        targets: Ground truth trajectories of shape (batch, seq_len, num_peds, 2)
        
    Returns:
        Dictionary containing calculated metrics:
            - 'ade': Average Displacement Error
            - 'fde': Final Displacement Error
            - 'mse': Mean Squared Error
            
    Raises:
        ValueError: If input tensors have incompatible shapes
        
    Example:
        >>> predictions = torch.randn(32, 12, 4, 2)
        >>> targets = torch.randn(32, 12, 4, 2)
        >>> metrics = calculate_metrics(predictions, targets)
        >>> print(f"ADE: {metrics['ade']:.4f}")
    """
    # Implementation here
    pass
```

### README Updates

When adding new features:
1. Update the main README.md
2. Add usage examples
3. Update the API documentation
4. Include performance benchmarks if applicable

## 🐛 Bug Reports

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the bug is already fixed
3. **Create a minimal reproduction** example

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.8.10]
- PyTorch version: [e.g. 2.0.1]
- CUDA version: [e.g. 11.8]

**Additional Context**
Add any other context about the problem here.
```

## 💡 Feature Requests

### Before Submitting

1. **Check if the feature already exists**
2. **Search existing feature requests**
3. **Consider if it fits the project scope**

### Feature Request Template

```markdown
**Feature Description**
A clear description of what you want to happen.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
A clear description of how you envision the feature working.

**Alternatives Considered**
Any alternative solutions or features you've considered.

**Additional Context**
Add any other context or screenshots about the feature request here.
```

## 🏷️ Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Examples

```bash
feat(models): add semantic feature integration to STGCNN
fix(dataset): handle missing trajectory data gracefully
docs(readme): update installation instructions
test(metrics): add unit tests for ADE/FDE calculations
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Follow code style guidelines**
5. **Write clear commit messages**

### Pull Request Template

```markdown
**Description**
Brief description of changes made.

**Type of Change**
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

**Testing**
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

**Checklist**
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Address feedback** if requested
4. **Merge** once approved

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Academic publications (for significant contributions)

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Email**: For private inquiries

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Semantic-STGCNN! 🙏
