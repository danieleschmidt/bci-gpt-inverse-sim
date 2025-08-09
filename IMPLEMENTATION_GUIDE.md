# BCI-GPT Implementation Guide

**Complete Developer Guide for Brain-Computer Interface GPT System**  
**Version:** 1.0  
**Target Audience:** Developers, Researchers, DevOps Engineers  
**Last Updated:** January 2025

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Development Environment](#development-environment)
4. [Core Components](#core-components)
5. [Training & Fine-tuning](#training--fine-tuning)
6. [Deployment](#deployment)
7. [Testing & Validation](#testing--validation)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended)
- Docker & Docker Compose
- Kubernetes (for production)
- 16GB+ RAM, 4GB+ GPU memory

### 5-Minute Setup
```bash
# Clone repository
git clone https://github.com/danieleschmidt/bci-gpt-inverse-sim.git
cd bci-gpt-inverse-sim

# Setup environment
conda create -n bci-gpt python=3.9
conda activate bci-gpt

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt

# Verify installation
python -c "from bci_gpt import BCIGPTModel; print('✅ Installation successful!')"

# Run basic tests
python -m pytest tests/test_models.py -v
```

### Hello World Example
```python
from bci_gpt import BCIGPTModel, EEGProcessor
import numpy as np

# Create synthetic EEG data
eeg_data = np.random.randn(1, 9, 1000).astype(np.float32)

# Initialize model
model = BCIGPTModel(eeg_channels=9, sequence_length=1000)

# Process EEG and generate text
with torch.no_grad():
    outputs = model.forward(torch.tensor(eeg_data))
    generated_text = model.generate_text(torch.tensor(eeg_data))
    
print(f"Generated from synthetic EEG: {generated_text[0]}")
```

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EEG Hardware  │───▶│  Preprocessing  │───▶│   BCI-GPT Model │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐              │
│  Text Output    │◀───│ Real-time Decode│◀─────────────┘
└─────────────────┘    └─────────────────┘
                                 │
┌─────────────────┐              │
│  Inverse GAN    │◀─────────────┘
│ (Text→EEG)      │
└─────────────────┘
```

### Component Map
```python
# Core system components
components = {
    "preprocessing": {
        "path": "bci_gpt/preprocessing/",
        "key_files": ["eeg_processor.py", "artifact_removal.py", "feature_extraction.py"],
        "purpose": "Signal cleaning and feature extraction"
    },
    "core_models": {
        "path": "bci_gpt/core/", 
        "key_files": ["models.py", "inverse_gan.py", "fusion_layers.py"],
        "purpose": "Neural network architectures"
    },
    "decoding": {
        "path": "bci_gpt/decoding/",
        "key_files": ["realtime_decoder.py", "token_decoder.py", "confidence_estimation.py"],
        "purpose": "Real-time thought-to-text conversion"
    },
    "training": {
        "path": "bci_gpt/training/",
        "key_files": ["trainer.py", "losses.py", "augmentation.py"],
        "purpose": "Model training and optimization"
    },
    "deployment": {
        "path": "bci_gpt/deployment/",
        "key_files": ["server.py", "production.py", "docker_configs.py"],
        "purpose": "Production deployment and APIs"
    }
}
```

## Development Environment

### Docker Development Setup
```bash
# Development container with all dependencies
docker-compose -f docker-compose.yml up -d

# Enter development environment
docker exec -it bci-gpt-dev bash

# Or use VS Code Dev Containers
# .devcontainer/devcontainer.json is pre-configured
```

### Local Development Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Install in development mode
pip install -e ".[dev,neuro,gpu]"

# Setup pre-commit hooks
pre-commit install

# Configure environment variables
cp .env.example .env
# Edit .env with your settings
```

### GPU Setup (NVIDIA)
```bash
# Verify CUDA installation
nvidia-smi

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify GPU access in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Core Components

### 1. EEG Preprocessing Pipeline

#### Basic Usage
```python
from bci_gpt.preprocessing import EEGProcessor, SignalQuality

# Initialize processor
processor = EEGProcessor(
    sampling_rate=1000,
    channels=['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
    reference='average',
    notch_filter=60
)

# Load and process EEG
eeg_data = processor.load_data("data/subject_01.fif")  # MNE format
processed = processor.preprocess(
    eeg_data,
    bandpass=(0.5, 40),
    artifact_removal='ica',
    epoch_length=1.0
)

# Check signal quality
quality = SignalQuality.assess(processed)
print(f"Quality score: {quality.score}/100")
```

#### Advanced Preprocessing
```python
# Custom preprocessing pipeline
pipeline = processor.create_pipeline([
    ('notch_filter', {'freq': 60}),
    ('bandpass', {'l_freq': 0.5, 'h_freq': 40}),
    ('artifact_removal', {'method': 'ica', 'n_components': 20}),
    ('rereferencing', {'ref_channels': 'average'}),
    ('epoching', {'duration': 1.0, 'overlap': 0.5}),
    ('feature_extraction', {'methods': ['psd', 'temporal']})
])

processed_data = pipeline.transform(raw_eeg)
```

### 2. BCI-GPT Model Architecture

#### Model Initialization
```python
from bci_gpt.core import BCIGPTModel

# Standard configuration
model = BCIGPTModel(
    eeg_channels=9,
    eeg_sampling_rate=1000,
    sequence_length=1000,
    language_model="gpt2-medium",
    fusion_method="cross_attention",
    latent_dim=256,
    freeze_lm=False
)

# Advanced configuration
model = BCIGPTModel(
    eeg_channels=64,  # High-density EEG
    eeg_sampling_rate=2000,  # High sampling rate
    sequence_length=2000,
    language_model="microsoft/DialoGPT-medium",
    fusion_method="cross_attention",
    latent_dim=512,
    freeze_lm=True  # Only train BCI components
)
```

#### Model Training Loop
```python
from bci_gpt.training import BCIGPTTrainer

# Initialize trainer
trainer = BCIGPTTrainer(
    model=model,
    learning_rate=1e-4,
    batch_size=16,
    max_epochs=100,
    patience=10,
    device="cuda"
)

# Train model
history = trainer.fit(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    callbacks=[
        'early_stopping',
        'model_checkpoint',
        'learning_rate_scheduler'
    ]
)

# Save trained model
model.save_pretrained("models/bci_gpt_best.pt")
```

### 3. Inverse GAN for EEG Synthesis

#### Training the Inverse GAN
```python
from bci_gpt.inverse import InverseGAN, TextToEEG

# Initialize inverse GAN
inverse_gan = InverseGAN(
    text_embedding_dim=768,
    noise_dim=100,
    eeg_channels=9,
    eeg_sequence_length=1000,
    generator_layers=[512, 1024, 2048],
    discriminator_layers=[2048, 1024, 512]
)

# Training loop
for epoch in range(100):
    for batch_idx, (text_embeddings, real_eeg) in enumerate(dataloader):
        # Train discriminator
        d_loss = inverse_gan.train_discriminator(text_embeddings, real_eeg)
        
        # Train generator
        g_loss = inverse_gan.train_generator(text_embeddings)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")

# Save trained GAN
inverse_gan.save("models/inverse_gan.pt")
```

#### Text-to-EEG Generation
```python
# Initialize text-to-EEG converter
text2eeg = TextToEEG(
    inverse_model_path="models/inverse_gan.pt",
    language_model="gpt2-medium"
)

# Generate synthetic EEG
text = "Hello, this is a test sentence."
synthetic_eeg = text2eeg.generate(
    text=text,
    duration=2.0,  # seconds
    sampling_rate=1000,
    style="imagined_speech"
)

# Validate realism
validation_metrics = text2eeg.validate_synthetic(
    synthetic_eeg, 
    reference_eeg_path="data/real_eeg_reference.npy"
)
print(f"Realism score: {validation_metrics['realism_score']:.3f}")
```

### 4. Real-Time Decoding

#### Setting Up Real-Time Decoder
```python
from bci_gpt.decoding import RealtimeDecoder
from bci_gpt.utils import StreamingEEG

# Initialize decoder
decoder = RealtimeDecoder(
    model_checkpoint="models/bci_gpt_best.pt",
    device="cuda",
    buffer_size=1000,  # ms
    confidence_threshold=0.7,
    sampling_rate=1000
)

# Setup EEG stream (example with simulated data)
class MockEEGStream:
    def __init__(self):
        self.is_connected = False
    
    def start(self):
        self.is_connected = True
    
    def get_data(self):
        if self.is_connected:
            # Return simulated EEG data
            return np.random.randn(9, 100)  # 100ms of data
        return None

# Connect to EEG stream
eeg_stream = MockEEGStream()
eeg_stream.start()

# Start real-time decoding
decoder.start_decoding(eeg_stream)

# Get decoded results
while True:
    result = decoder.get_next_result()
    if result and result.confidence > 0.8:
        print(f"Decoded: {result.text} (confidence: {result.confidence:.2f})")
    time.sleep(0.1)  # 10 Hz polling
```

#### Hardware Integration Example (OpenBCI)
```python
from bci_gpt.hardware import OpenBCIStream

# Connect to OpenBCI board
openbci_stream = OpenBCIStream(
    board_type="cyton",
    serial_port="/dev/ttyUSB0",
    channels=8,
    sampling_rate=250
)

# Configure channels for imagined speech
openbci_stream.configure_channels([
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"
])

# Start streaming
openbci_stream.start()
decoder.start_decoding(openbci_stream)
```

## Training & Fine-tuning

### Dataset Preparation
```python
from bci_gpt.data import BCIDataset, create_dataloaders

# Prepare dataset
dataset = BCIDataset(
    eeg_dir="data/eeg_recordings/",
    text_dir="data/transcripts/",
    subjects=["S001", "S002", "S003"],
    task="imagined_speech"
)

# Create data loaders
train_loader, val_loader = create_dataloaders(
    dataset,
    train_split=0.8,
    batch_size=16,
    num_workers=4,
    pin_memory=True
)
```

### Training Configuration
```python
# Training configuration
training_config = {
    "model": {
        "eeg_channels": 9,
        "sequence_length": 1000,
        "language_model": "gpt2-medium",
        "fusion_method": "cross_attention",
        "latent_dim": 256
    },
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "max_epochs": 100,
        "patience": 10,
        "weight_decay": 1e-5
    },
    "data": {
        "train_subjects": ["S001", "S002", "S003", "S004"],
        "val_subjects": ["S005"],
        "test_subjects": ["S006"],
        "augmentation": True
    }
}

# Train model with configuration
trainer = BCIGPTTrainer.from_config(training_config)
trainer.fit()
```

### Advanced Training Techniques

#### Multi-GPU Training
```python
# Distributed training setup
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    
# Or use DistributedDataParallel for better performance
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

#### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Fine-tuning for New Subjects
```python
from bci_gpt.training import SubjectAdapter

# Load pre-trained model
base_model = BCIGPTModel.from_pretrained("models/bci_gpt_base.pt")

# Adapt to new subject
adapter = SubjectAdapter(
    base_model=base_model,
    adaptation_method="maml",  # Model-Agnostic Meta-Learning
    shots_per_class=5
)

# Collect calibration data
calibration_data = collect_subject_calibration(
    subject_id="new_subject_01",
    words=["yes", "no", "help", "stop", "more"],
    repetitions=5
)

# Fine-tune model
adapted_model = adapter.adapt(
    calibration_data,
    steps=50,
    learning_rate=0.001
)

# Evaluate adaptation performance
accuracy = evaluate_subject_performance(adapted_model, test_data)
print(f"Subject adaptation accuracy: {accuracy:.2%}")
```

## Deployment

### Local Deployment
```bash
# Start local server
python -m bci_gpt.deployment.server --host 0.0.0.0 --port 8000

# Or using Gunicorn for production
gunicorn bci_gpt.deployment.server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Docker Deployment
```bash
# Build production image
docker build -t bci-gpt:latest .

# Run container
docker run -d \
    --name bci-gpt-api \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -e ENVIRONMENT=production \
    bci-gpt:latest
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get deployments -n bci-gpt
kubectl get pods -n bci-gpt

# Access logs
kubectl logs -f deployment/bci-gpt-api -n bci-gpt
```

### Production API Usage
```python
import requests

# API endpoint
api_url = "https://api.bci-gpt.com/v1"

# Upload EEG data for processing
files = {"eeg_data": open("sample_eeg.npy", "rb")}
response = requests.post(f"{api_url}/decode", files=files)

result = response.json()
print(f"Decoded text: {result['text']}")
print(f"Confidence: {result['confidence']}")

# Real-time streaming endpoint
import websocket

def on_message(ws, message):
    result = json.loads(message)
    print(f"Real-time result: {result}")

ws = websocket.WebSocketApp(
    "wss://api.bci-gpt.com/v1/stream",
    on_message=on_message
)
ws.run_forever()
```

## Testing & Validation

### Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=bci_gpt --cov-report=html
```

### Integration Tests
```python
# Custom integration test
from bci_gpt.testing import SystemTester

tester = SystemTester()

# Test end-to-end pipeline
results = tester.test_pipeline(
    input_data="data/test_eeg.npy",
    expected_output="Hello world",
    tolerance=0.1
)

assert results["success"] == True
assert results["word_error_rate"] < 0.2
```

### Performance Benchmarks
```python
from bci_gpt.evaluation import BenchmarkSuite

# Run standard benchmarks
benchmark = BenchmarkSuite()
results = benchmark.run_all()

print(f"Latency: {results['latency_ms']:.1f}ms")
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Throughput: {results['samples_per_second']:.0f}")
```

### Model Validation
```python
# Cross-validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kfold.split(dataset):
    model = BCIGPTModel()
    # Train on train_idx, validate on val_idx
    score = evaluate_model(model, val_data)
    cv_scores.append(score)

print(f"Cross-validation accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

## Performance Optimization

### Model Optimization
```python
# Quantization for faster inference
from bci_gpt.optimization import ModelOptimizer

optimizer = ModelOptimizer()

# Post-training quantization
quantized_model = optimizer.quantize_model(
    model,
    method="dynamic",  # or "static", "qat"
    calibration_data=calibration_loader
)

# Model pruning
pruned_model = optimizer.prune_model(
    model,
    sparsity=0.8,
    structured=False
)

# Export to ONNX for production
optimizer.export_onnx(
    model,
    output_path="models/bci_gpt_optimized.onnx",
    input_shape=(1, 9, 1000)
)
```

### Caching & Memory Optimization
```python
from bci_gpt.optimization import CacheManager, MemoryOptimizer

# Setup caching
cache = CacheManager(
    backend="redis",
    host="localhost",
    port=6379,
    ttl=3600  # 1 hour
)

# Cache model outputs
@cache.cached(key="eeg_features_{eeg_hash}")
def extract_features(eeg_data):
    return model.encode_eeg(eeg_data)

# Memory optimization
memory_optimizer = MemoryOptimizer()
memory_optimizer.optimize_model(model)
```

### GPU Optimization
```python
# Mixed precision inference
with torch.cuda.amp.autocast():
    outputs = model(eeg_data)

# Batch processing for throughput
def batch_decode(eeg_batch, batch_size=32):
    results = []
    for i in range(0, len(eeg_batch), batch_size):
        batch = eeg_batch[i:i+batch_size]
        with torch.no_grad():
            batch_results = model(batch)
        results.extend(batch_results)
    return results
```

## Troubleshooting

### Common Issues & Solutions

#### 1. CUDA Out of Memory
```python
# Solution: Reduce batch size or use gradient accumulation
trainer = BCIGPTTrainer(
    model=model,
    batch_size=8,  # Reduced from 16
    accumulate_grad_batches=2  # Effective batch size = 16
)
```

#### 2. Poor Model Performance
```python
# Debug training
from bci_gpt.debugging import TrainingDebugger

debugger = TrainingDebugger(model)
debugger.check_gradients()  # Check for vanishing/exploding gradients
debugger.visualize_features()  # Visualize learned features
debugger.analyze_losses()  # Analyze loss components
```

#### 3. Real-time Latency Issues
```python
# Profile real-time pipeline
from bci_gpt.profiling import RealTimeProfiler

profiler = RealTimeProfiler()
with profiler:
    result = decoder.decode(eeg_data)

profiler.print_stats()  # Show bottlenecks
```

### Logging & Monitoring
```python
from bci_gpt.utils import setup_logging, get_logger

# Setup comprehensive logging
setup_logging(
    level="DEBUG",
    log_to_file=True,
    log_to_tensorboard=True
)

logger = get_logger(__name__)

# Monitor system health
from bci_gpt.monitoring import SystemMonitor

monitor = SystemMonitor()
monitor.start()

# Check metrics
metrics = monitor.get_metrics()
logger.info(f"System metrics: {metrics}")
```

### Performance Debugging
```python
# Profile model inference
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=profiler.tensorboard_trace_handler('./log/profiler')
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 10:  # Only profile first 10 steps
            break
        outputs = model(batch)
        prof.step()
```

## Contributing

### Development Workflow
```bash
# Fork and clone repository
git clone https://github.com/your-username/bci-gpt-inverse-sim.git

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

### Code Style Guidelines
```bash
# Format code
black bci_gpt/
isort bci_gpt/

# Run linting
flake8 bci_gpt/
mypy bci_gpt/

# Pre-commit hooks (automatic)
pre-commit run --all-files
```

### Testing Requirements
- All new features must include unit tests
- Integration tests for major components
- Performance benchmarks for optimization changes
- Documentation updates for user-facing changes

### Pull Request Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Performance impact assessed
- [ ] Security review completed

## Advanced Topics

### Custom Model Architectures
```python
# Extend base model
class CustomBCIGPT(BCIGPTModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom layers
        self.custom_fusion = CustomFusionLayer()
    
    def forward(self, eeg_data, **kwargs):
        # Custom forward pass
        eeg_features = self.eeg_encoder(eeg_data)
        fused_features = self.custom_fusion(eeg_features)
        return self.decode_to_text(fused_features)
```

### Research Extensions
```python
# Multi-subject meta-learning
from bci_gpt.research import MetaLearner

meta_learner = MetaLearner(
    base_model=BCIGPTModel,
    meta_learning_rate=0.001,
    adaptation_steps=5
)

# Train on multiple subjects
for subject_batch in subject_dataloader:
    meta_loss = meta_learner.meta_train(subject_batch)

# Fast adaptation to new subject
adapted_model = meta_learner.adapt(new_subject_data)
```

### Clinical Integration
```python
from bci_gpt.clinical import ClinicalValidator, SafetyMonitor

# Clinical validation
validator = ClinicalValidator()
safety_report = validator.validate_system(model, clinical_data)

# Real-time safety monitoring
safety_monitor = SafetyMonitor()
safety_monitor.start_monitoring(patient_session)
```

## Resources & References

### Documentation
- [API Reference](https://bci-gpt.readthedocs.io/en/latest/api/)
- [Tutorials](https://bci-gpt.readthedocs.io/en/latest/tutorials/)
- [Research Papers](https://github.com/danieleschmidt/bci-gpt-papers)

### Community
- [GitHub Issues](https://github.com/danieleschmidt/bci-gpt-inverse-sim/issues)
- [Discord Server](https://discord.gg/bci-gpt)
- [Research Forum](https://forum.bci-gpt.com)

### Hardware Compatibility
- OpenBCI (Cyton, Ganglion)
- g.tec (g.USBamp, g.HIamp)
- ANT Neuro (eego™, WaveGuard)
- Cognionics (Dry electrodes)
- Custom LSL-compatible devices

---
*Implementation Guide Version 1.0*  
*For BCI-GPT System v0.1.0*  
*Complete Developer Reference*