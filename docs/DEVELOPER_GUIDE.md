# BCI-GPT Developer Guide

## Table of Contents

1. [Development Setup](#development-setup)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Contributing](#contributing)
5. [Testing](#testing)
6. [Deployment](#deployment)

## Development Setup

### Prerequisites
- Python 3.9+
- Git
- Docker (optional)
- CUDA-capable GPU (recommended)

### Local Development

```bash
# Clone repository
git clone https://github.com/danieleschmidt/bci-gpt-inverse-sim.git
cd bci-gpt-inverse-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Development Tools

#### Code Formatting
```bash
# Format code
black bci_gpt/
isort bci_gpt/

# Lint code
flake8 bci_gpt/
mypy bci_gpt/
```

#### Documentation
```bash
# Build docs
cd docs/
make html

# Serve docs locally
python -m http.server 8080
```

## Architecture Overview

BCI-GPT follows a modular architecture with clear separation of concerns:

```
bci_gpt/
├── core/               # Core models and algorithms
│   ├── models.py       # Neural network architectures
│   ├── inverse_gan.py  # GAN for inverse mapping
│   └── fusion_layers.py # Multi-modal fusion
├── preprocessing/      # Signal processing
│   ├── eeg_processor.py
│   ├── artifact_removal.py
│   └── feature_extraction.py
├── decoding/          # Real-time decoding
│   ├── realtime_decoder.py
│   ├── token_decoder.py
│   └── confidence_estimation.py
├── training/          # Model training
│   ├── trainer.py
│   ├── gan_trainer.py
│   └── losses.py
└── utils/             # Utilities
    ├── streaming.py
    ├── metrics.py
    └── visualization.py
```

### Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Testability**: All components are unit testable
3. **Extensibility**: Easy to add new models and features
4. **Performance**: Optimized for real-time processing
5. **Reliability**: Robust error handling and recovery

## Core Components

### EEG Processing Pipeline

```python
from bci_gpt.preprocessing import EEGProcessor

class EEGProcessor:
    def __init__(self, sampling_rate: int, channels: List[str]):
        self.sampling_rate = sampling_rate
        self.channels = channels
    
    def preprocess(self, raw_eeg: np.ndarray) -> np.ndarray:
        # Bandpass filtering
        filtered = self.bandpass_filter(raw_eeg)
        
        # Artifact removal
        clean = self.remove_artifacts(filtered)
        
        # Feature extraction
        features = self.extract_features(clean)
        
        return features
```

### Model Architecture

```python
from bci_gpt.core import BCIGPTModel

class BCIGPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.eeg_encoder = EEGEncoder(config.eeg_config)
        self.language_decoder = LanguageDecoder(config.lang_config)
        self.fusion_layer = CrossAttentionFusion(config.fusion_config)
    
    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        # Encode EEG signals
        eeg_features = self.eeg_encoder(eeg_input)
        
        # Fuse with language model
        fused_features = self.fusion_layer(eeg_features)
        
        # Decode to text
        text_output = self.language_decoder(fused_features)
        
        return text_output
```

### Training Pipeline

```python
from bci_gpt.training import BCITrainer

class BCITrainer:
    def __init__(self, model: BCIGPTModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
        
        return total_loss / len(dataloader)
```

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run the test suite: `pytest`
5. Format code: `black . && isort .`
6. Commit changes: `git commit -m "Add my feature"`
7. Push to branch: `git push origin feature/my-feature`
8. Create a Pull Request

### Code Style Guidelines

#### Python Style
- Follow PEP 8
- Use Black for formatting
- Use type hints
- Write docstrings for all public functions

```python
def process_eeg_data(
    raw_data: np.ndarray,
    sampling_rate: int,
    channels: List[str]
) -> ProcessedEEG:
    """Process raw EEG data.
    
    Args:
        raw_data: Raw EEG signal data
        sampling_rate: Sampling frequency in Hz
        channels: List of channel names
    
    Returns:
        Processed EEG data with features
    
    Raises:
        ValueError: If data format is invalid
    """
    # Implementation here
    pass
```

#### Documentation Style
- Use reStructuredText format
- Include examples in docstrings
- Maintain up-to-date API documentation

### Commit Message Format

```
type(scope): brief description

Detailed explanation of changes if needed.

Closes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_preprocessing/
│   ├── test_models/
│   └── test_training/
├── integration/       # Integration tests
│   ├── test_pipelines/
│   └── test_api/
└── fixtures/          # Test data and fixtures
    ├── sample_eeg.npy
    └── test_config.yaml
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/unit/test_models/

# Run with coverage
pytest --cov=bci_gpt --cov-report=html

# Run performance tests
pytest tests/performance/ --benchmark-only
```

### Writing Tests

```python
import pytest
from bci_gpt.preprocessing import EEGProcessor

class TestEEGProcessor:
    @pytest.fixture
    def processor(self):
        return EEGProcessor(
            sampling_rate=1000,
            channels=['Fz', 'Cz', 'Pz']
        )
    
    @pytest.fixture
    def sample_eeg(self):
        return np.random.randn(3, 1000)  # 3 channels, 1000 samples
    
    def test_preprocessing_shape(self, processor, sample_eeg):
        """Test that preprocessing maintains correct shape."""
        processed = processor.preprocess(sample_eeg)
        assert processed.shape[0] == sample_eeg.shape[0]
    
    def test_bandpass_filtering(self, processor, sample_eeg):
        """Test bandpass filtering removes out-of-band frequencies."""
        processed = processor.bandpass_filter(sample_eeg, low=1, high=40)
        # Add frequency domain assertions
        assert processed is not None
```

### Test Data

Store test data in `tests/fixtures/`:
- Small synthetic EEG datasets
- Configuration files for testing
- Expected output files

### Continuous Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- Nightly builds

CI pipeline includes:
- Unit and integration tests
- Code coverage reporting
- Security scanning
- Performance benchmarks

## Performance Optimization

### Profiling

```python
# Profile code
from bci_gpt.utils import ProfiledContext

with ProfiledContext("eeg_processing"):
    processed_data = processor.preprocess(raw_data)
```

### Memory Optimization

```python
# Use memory-efficient data structures
from bci_gpt.utils import RingBuffer

# For streaming data
buffer = RingBuffer(maxsize=1000)
buffer.append(new_sample)
```

### GPU Acceleration

```python
# Enable CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input_data)
```

## Debugging

### Logging Configuration

```python
import logging
from bci_gpt.utils import setup_logging

# Setup logging
setup_logging(level='DEBUG', format='detailed')

logger = logging.getLogger(__name__)
logger.debug("Processing EEG chunk of size %s", data.shape)
```

### Debug Tools

```python
# Visualize EEG signals
from bci_gpt.utils import plot_eeg_signals

plot_eeg_signals(eeg_data, channels=['Fz', 'Cz', 'Pz'])

# Debug model predictions
from bci_gpt.utils import debug_model_output

debug_model_output(model, input_data, layer_names=['encoder', 'decoder'])
```

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
2. **Slow training**: Enable mixed precision, use DataLoader with multiple workers
3. **Poor accuracy**: Check data quality, adjust hyperparameters, increase training data

## Deployment

### Docker Deployment

```dockerfile
# Multi-stage build for production
FROM python:3.9-slim as base
WORKDIR /app

FROM base as dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM dependencies as application
COPY . .
RUN pip install -e .

CMD ["python", "-m", "bci_gpt.server"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bci-gpt
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bci-gpt
  template:
    metadata:
      labels:
        app: bci-gpt
    spec:
      containers:
      - name: bci-gpt
        image: bci-gpt:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Monitoring

```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')

@REQUEST_LATENCY.time()
def process_request():
    REQUEST_COUNT.inc()
    # Process request
```

## API Development

### Adding New Endpoints

```python
from fastapi import APIRouter, HTTPException
from bci_gpt.api.models import EEGProcessRequest, EEGProcessResponse

router = APIRouter()

@router.post("/eeg/process", response_model=EEGProcessResponse)
async def process_eeg(request: EEGProcessRequest):
    """Process EEG data."""
    try:
        result = await eeg_processor.process(request.eeg_data)
        return EEGProcessResponse(
            processed_data=result.data,
            quality_score=result.quality
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### API Testing

```python
from fastapi.testclient import TestClient
from bci_gpt.api import app

client = TestClient(app)

def test_process_eeg():
    response = client.post("/eeg/process", json={
        "eeg_data": [[0.1, 0.2, 0.3]],
        "sampling_rate": 1000
    })
    assert response.status_code == 200
    assert "processed_data" in response.json()
```

---

For more information, see the [API Documentation](./API.md) or [User Guide](./USER_GUIDE.md).
