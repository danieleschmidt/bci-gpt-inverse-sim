# Contributing to BCI-GPT

Thank you for your interest in contributing to BCI-GPT! This document provides guidelines for contributing to the Brain-Computer Interface GPT Inverse Simulator project.

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contribution Guidelines](#contribution-guidelines)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Research Contributions](#research-contributions)
9. [Community Guidelines](#community-guidelines)

## Getting Started

### Ways to Contribute

We welcome contributions in many forms:

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new functionality
- 🔧 **Code Contributions**: Bug fixes, features, optimizations
- 📚 **Documentation**: Improve docs, tutorials, examples
- 🧪 **Research**: Novel algorithms, validation studies
- 🎨 **UI/UX**: Interface improvements and accessibility
- 🌍 **Localization**: Translation to other languages
- 📊 **Data**: Datasets, benchmarks, validation data

### Before You Start

1. **Search existing issues** to see if your contribution has been discussed
2. **Read the research papers** linked in the README to understand the approach
3. **Check the roadmap** in RESEARCH_OPPORTUNITIES.md for priority areas
4. **Join our community** on Discord for discussions and questions

## Development Setup

### Prerequisites

- Python 3.9+ (recommend 3.11)
- CUDA-capable GPU (recommended for training)
- Git with LFS support
- Docker (optional, for containerized development)

### Environment Setup

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/bci-gpt-inverse-sim.git
cd bci-gpt-inverse-sim

# Add upstream remote
git remote add upstream https://github.com/danieleschmidt/bci-gpt-inverse-sim.git

# Create development environment
conda create -n bci-gpt-dev python=3.11
conda activate bci-gpt-dev

# Install in development mode
pip install -e ".[dev,neuro,gpu]"

# Install pre-commit hooks
pre-commit install

# Verify setup
python -c "import bci_gpt; print('✅ Setup successful!')"
python -m pytest tests/test_models.py -v
```

### Docker Development (Alternative)

```bash
# Use development container
docker-compose -f docker-compose.dev.yml up -d
docker exec -it bci-gpt-dev bash

# Or use VS Code Dev Containers
# Open in VS Code and select "Reopen in Container"
```

## Contribution Guidelines

### Issue Reporting

When reporting bugs or requesting features:

#### Bug Reports
```markdown
## Bug Description
A clear description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- Python version:
- PyTorch version:
- GPU (if applicable):
- Operating System:

## Additional Context
Screenshots, logs, or other relevant information.
```

#### Feature Requests
```markdown
## Feature Description
Clear description of the proposed feature.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches you've considered.

## Research Relevance
How does this relate to current BCI research?
```

### Code Contributions

#### Branching Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for new features
- `feature/feature-name`: Individual feature branches
- `bugfix/issue-number`: Bug fix branches
- `research/experiment-name`: Research experiment branches

#### Branch Naming Conventions

```bash
# Features
feature/realtime-decoding-optimization
feature/multi-language-support

# Bug fixes
bugfix/memory-leak-preprocessing
bugfix/cuda-compatibility-issue

# Research
research/transformer-attention-analysis
research/gan-stability-improvements

# Documentation
docs/api-documentation
docs/tutorial-improvements
```

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>[optional scope]: <description>

feat(decoder): add real-time confidence estimation
fix(preprocessing): resolve memory leak in artifact removal
docs(api): update documentation for new endpoints
test(integration): add comprehensive system tests
refactor(core): improve model architecture organization
perf(training): optimize GPU memory usage during training
research(gan): implement progressive growing for EEG synthesis
```

#### Commit Types

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `research`: Research-related changes
- `ci`: CI/CD changes
- `build`: Build system changes

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Line length: 88 characters (Black default)
# Use type hints for all functions
def process_eeg_data(
    eeg_data: torch.Tensor,
    sampling_rate: int = 1000,
    apply_filtering: bool = True
) -> torch.Tensor:
    """Process EEG data with optional filtering.
    
    Args:
        eeg_data: Input EEG tensor (batch, channels, samples)
        sampling_rate: Sampling rate in Hz
        apply_filtering: Whether to apply preprocessing filters
        
    Returns:
        Processed EEG tensor
        
    Raises:
        ValueError: If input dimensions are invalid
    """
    pass

# Use dataclasses for structured data
@dataclass
class EEGProcessingConfig:
    """Configuration for EEG preprocessing."""
    sampling_rate: int = 1000
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    notch_freq: float = 60.0
```

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black bci_gpt/
isort bci_gpt/

# Check formatting
black --check bci_gpt/
isort --check-only bci_gpt/

# Lint code
flake8 bci_gpt/
mypy bci_gpt/

# Run all checks (automated via pre-commit)
pre-commit run --all-files
```

### Documentation Style

#### Docstrings

Use Google-style docstrings:

```python
def train_bci_model(
    model: BCIGPTModel,
    train_data: DataLoader,
    val_data: DataLoader,
    epochs: int = 100
) -> Dict[str, float]:
    """Train BCI-GPT model with given data.
    
    This function implements the complete training loop including
    validation, checkpointing, and early stopping.
    
    Args:
        model: The BCI-GPT model to train
        train_data: Training data loader
        val_data: Validation data loader  
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics:
        - 'train_loss': Final training loss
        - 'val_loss': Final validation loss
        - 'accuracy': Validation accuracy
        
    Raises:
        RuntimeError: If CUDA is required but not available
        ValueError: If data loaders are empty
        
    Example:
        >>> model = BCIGPTModel()
        >>> metrics = train_bci_model(model, train_loader, val_loader)
        >>> print(f"Final accuracy: {metrics['accuracy']:.2%}")
    """
```

#### Comments

```python
# Use comments to explain WHY, not WHAT
# Good: Explain reasoning
# Use cross-attention to align EEG features with language model representations
# This allows the model to focus on relevant neural patterns for each token

# Bad: Explain obvious code
# Create a tensor of zeros
zeros = torch.zeros(10)
```

## Testing

### Test Organization

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for workflows
├── end_to_end/       # Full system tests
├── performance/      # Performance benchmarks
├── research/         # Research validation tests
├── fixtures/         # Test data and utilities
└── conftest.py       # Pytest configuration
```

### Writing Tests

#### Unit Tests
```python
import pytest
import torch
from bci_gpt.core import BCIGPTModel

class TestBCIGPTModel:
    """Test suite for BCI-GPT model."""
    
    @pytest.fixture
    def sample_eeg_data(self):
        """Fixture providing sample EEG data."""
        return torch.randn(2, 9, 1000)  # batch=2, channels=9, samples=1000
    
    @pytest.fixture
    def model(self):
        """Fixture providing a test model."""
        return BCIGPTModel(eeg_channels=9, sequence_length=1000)
    
    def test_model_forward_pass(self, model, sample_eeg_data):
        """Test model forward pass with valid input."""
        outputs = model(sample_eeg_data)
        
        assert 'eeg_features' in outputs
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == sample_eeg_data.shape[0]
    
    def test_model_invalid_input_shape(self, model):
        """Test model behavior with invalid input."""
        invalid_eeg = torch.randn(2, 5, 1000)  # Wrong channel count
        
        with pytest.raises(ValueError, match="Expected 9 channels"):
            model(invalid_eeg)
    
    @pytest.mark.parametrize("batch_size,channels,samples", [
        (1, 9, 1000),
        (4, 9, 1000),
        (8, 9, 2000),
    ])
    def test_model_different_batch_sizes(self, batch_size, channels, samples):
        """Test model with different input dimensions."""
        model = BCIGPTModel(eeg_channels=channels, sequence_length=samples)
        eeg_data = torch.randn(batch_size, channels, samples)
        
        outputs = model(eeg_data)
        assert outputs['logits'].shape[0] == batch_size
```

#### Integration Tests
```python
def test_complete_pipeline():
    """Test complete EEG processing to text generation pipeline."""
    # Simulate EEG data
    raw_eeg = generate_synthetic_eeg()
    
    # Preprocess
    processor = EEGProcessor()
    processed_eeg = processor.preprocess(raw_eeg)
    
    # Model inference
    model = BCIGPTModel.from_pretrained("test_model.pt")
    generated_text = model.generate_text(processed_eeg)
    
    # Validate outputs
    assert isinstance(generated_text, list)
    assert all(isinstance(text, str) for text in generated_text)
    assert all(len(text.strip()) > 0 for text in generated_text)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=bci_gpt --cov-report=html

# Run tests in parallel
pytest -n auto

# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only
```

### Performance Testing

```python
import pytest
import time

class TestPerformance:
    """Performance benchmarks for critical components."""
    
    def test_realtime_decoding_latency(self):
        """Ensure real-time decoding meets latency requirements."""
        decoder = RealtimeDecoder()
        eeg_data = torch.randn(1, 9, 100)  # 100ms of data
        
        start_time = time.time()
        result = decoder.decode(eeg_data)
        latency = (time.time() - start_time) * 1000  # ms
        
        assert latency < 50  # Must be under 50ms
        assert result.confidence > 0
    
    @pytest.mark.benchmark
    def test_model_throughput(self, benchmark):
        """Benchmark model throughput."""
        model = BCIGPTModel()
        eeg_data = torch.randn(32, 9, 1000)  # Batch of 32
        
        def inference():
            with torch.no_grad():
                return model(eeg_data)
        
        result = benchmark(inference)
        # Should process >100 samples/second
        assert benchmark.stats['min'] < 0.32  # 32 samples in <0.32s
```

## Documentation

### Types of Documentation

1. **API Documentation**: Automatically generated from docstrings
2. **Tutorials**: Step-by-step guides for common tasks
3. **How-to Guides**: Solutions for specific problems
4. **Research Documentation**: Papers, experiments, results

### Writing Documentation

#### Tutorials
```markdown
# Tutorial: Training Your First BCI-GPT Model

This tutorial walks you through training a BCI-GPT model from scratch.

## Prerequisites

- Python 3.9+
- PyTorch installed
- Sample EEG dataset

## Step 1: Prepare Your Data

First, let's load and preprocess your EEG data:

```python
from bci_gpt import EEGProcessor

processor = EEGProcessor()
# ... code example
```

Why this step is important: [explanation]

## Step 2: Configure the Model

[Continue with clear, executable examples]
```

#### API Documentation

Ensure all public functions have comprehensive docstrings that will be automatically included in the API docs.

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs/
make html

# Serve locally
python -m http.server 8000
```

## Pull Request Process

### Before Submitting

1. **Fork and branch**: Create a feature branch from the latest `develop`
2. **Write tests**: Ensure new code has appropriate test coverage
3. **Update docs**: Update documentation for new features
4. **Run quality gates**: All checks must pass

```bash
# Pre-submission checklist
pytest                           # All tests pass
black --check bci_gpt/          # Code formatted
flake8 bci_gpt/                 # No linting errors
mypy bci_gpt/                   # Type checking passes
python run_quality_gates.py     # Quality gates pass
```

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update
- [ ] Research contribution

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Research Impact
- [ ] Cite relevant papers
- [ ] Describe novel contributions
- [ ] Include performance benchmarks

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] Quality gates pass
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs all quality gates
2. **Peer review**: At least one maintainer review required
3. **Research review**: Novel research contributions need domain expert review
4. **Final approval**: Project maintainer approves and merges

#### Review Criteria

- **Code quality**: Clean, readable, well-documented
- **Testing**: Appropriate test coverage
- **Performance**: No significant performance regressions
- **Research validity**: Sound methodology and evaluation
- **Documentation**: Clear explanations and examples

## Research Contributions

### Novel Research Areas

High-priority research contributions include:

1. **Improved Architectures**: Novel neural network designs
2. **Better Training Methods**: Advanced training techniques
3. **New Datasets**: High-quality EEG datasets with labels
4. **Validation Studies**: Clinical or user studies
5. **Optimization Techniques**: Performance and efficiency improvements

### Research Contribution Guidelines

#### Experimental Design
- Use rigorous experimental methodology
- Include appropriate baselines and comparisons
- Provide statistical significance testing
- Ensure reproducibility with provided code and data

#### Code Standards for Research
```python
# Research experiments should be well-documented
class ExperimentConfig:
    """Configuration for research experiment."""
    seed: int = 42  # For reproducibility
    n_subjects: int = 10
    n_trials_per_subject: int = 100
    cross_validation_folds: int = 5

def run_experiment(config: ExperimentConfig) -> Dict[str, float]:
    """Run research experiment with given configuration.
    
    Returns:
        Dictionary of metrics including mean, std, and significance tests
    """
    torch.manual_seed(config.seed)
    # ... experiment implementation
    
    return {
        'accuracy_mean': accuracy.mean(),
        'accuracy_std': accuracy.std(),
        'p_value': statistical_test(accuracy, baseline)
    }
```

#### Publishing Research
- Submit to `research/experiments/` directory
- Include comprehensive README with methodology
- Provide reproducible code and configuration files
- Document computational requirements and runtime

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

#### Our Standards

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome diverse perspectives and experiences  
- **Be collaborative**: Work together constructively
- **Be patient**: Help others learn and grow
- **Be professional**: Maintain a professional tone in all interactions

#### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Sharing others' private information without consent
- Any other conduct that would be inappropriate in a professional setting

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, technical discussions
- **GitHub Discussions**: General questions, ideas, community chat
- **Discord**: Real-time chat, collaboration, community events
- **Email**: Private communication with maintainers

### Recognition

We recognize contributors through:

- **Contributor list**: All contributors listed in README
- **Release notes**: Significant contributions highlighted
- **Research citations**: Research contributors cited in papers
- **Community awards**: Annual recognition for outstanding contributions

## Getting Help

### Resources

1. **Documentation**: Comprehensive guides and API reference
2. **Examples**: Sample code and tutorials
3. **Community**: Active Discord community
4. **Issue tracker**: Search existing issues and solutions

### Asking for Help

When asking questions:

1. **Search first**: Check if your question has been answered
2. **Be specific**: Provide clear details about your issue
3. **Include context**: System info, error messages, code samples
4. **Be patient**: Maintainers and community members are volunteers

### Mentorship Program

New contributors can request mentorship from experienced community members:

- **Beginner-friendly issues**: Labeled for new contributors
- **Pair programming**: Virtual coding sessions
- **Code reviews**: Detailed feedback on contributions
- **Research guidance**: Help with research projects

## Thank You

Your contributions make BCI-GPT better for everyone. Whether you're fixing a typo, adding a feature, or conducting groundbreaking research, every contribution matters.

Welcome to the BCI-GPT community! 🧠🤖

---

*Contributing Guide Version 1.0*  
*Last Updated: January 2025*  
*Questions? Open an issue or join our Discord!*