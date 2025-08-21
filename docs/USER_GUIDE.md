# BCI-GPT User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

## Getting Started

BCI-GPT is a brain-computer interface system that converts imagined speech from EEG signals into text. This guide will help you get started with using the system.

### Prerequisites

- Python 3.9 or later
- EEG recording device (OpenBCI, Emotiv, etc.)
- Basic understanding of EEG concepts

### Quick Start

1. Install BCI-GPT
2. Connect your EEG device
3. Run the calibration procedure
4. Start decoding thoughts to text

## Installation

### Option 1: pip install (Recommended)
```bash
pip install bci-gpt
```

### Option 2: From Source
```bash
git clone https://github.com/danieleschmidt/bci-gpt-inverse-sim.git
cd bci-gpt-inverse-sim
pip install -e .
```

### Option 3: Docker
```bash
docker pull bci-gpt:latest
docker run -p 8000:8000 bci-gpt:latest
```

### Hardware Setup

#### Supported EEG Devices
- OpenBCI Cyton (8-channel)
- OpenBCI Ganglion (4-channel)
- Emotiv EPOC+ (14-channel)
- g.tec g.USBamp
- Custom devices via LSL

#### Electrode Placement
For optimal performance, place electrodes at:
- **Primary**: Fz, Cz, Pz (speech motor cortex)
- **Secondary**: F3, F4, C3, C4 (language areas)
- **Reference**: Mastoids or earlobes

## Basic Usage

### 1. Initialize the System

```python
from bci_gpt import BCIGPTSystem

# Initialize system
bci = BCIGPTSystem(
    device='openbci',
    channels=['Fz', 'Cz', 'Pz', 'F3', 'F4', 'C3', 'C4'],
    sampling_rate=1000
)

# Connect to device
bci.connect()
```

### 2. Calibration

```python
# Run calibration (15-20 minutes)
calibration_words = ['yes', 'no', 'hello', 'stop', 'help']
bci.calibrate(words=calibration_words, repetitions=10)
```

### 3. Real-time Decoding

```python
# Start real-time decoding
bci.start_decoding()

# Get decoded text
while True:
    text = bci.get_decoded_text()
    if text:
        print(f"Thought: {text}")
```

### 4. Batch Processing

```python
# Process recorded EEG file
from bci_gpt import process_eeg_file

results = process_eeg_file(
    'recording.edf',
    model='bci-gpt-v1',
    output_format='text'
)

print(results.decoded_text)
```

## Advanced Features

### Custom Model Training

Train your own BCI model:

```python
from bci_gpt import BCITrainer

trainer = BCITrainer(
    model_architecture='transformer',
    training_data='path/to/data',
    epochs=100,
    batch_size=32
)

# Train model
model = trainer.train()

# Save model
model.save('my_bci_model.pt')
```

### Signal Quality Monitoring

```python
# Monitor signal quality
quality = bci.get_signal_quality()

print(f"Overall quality: {quality.overall_score}")
print(f"Noisy channels: {quality.noisy_channels}")
print(f"Impedance check: {quality.impedance_ok}")
```

### Artifact Removal

```python
from bci_gpt import ArtifactRemover

# Configure artifact removal
artifact_remover = ArtifactRemover(
    methods=['ica', 'asr'],
    muscle_artifact_threshold=50,
    eye_artifact_threshold=100
)

# Apply to EEG data
clean_eeg = artifact_remover.clean(raw_eeg_data)
```

### Multi-Language Support

```python
# Set language
bci.set_language('spanish')

# Language-specific vocabulary
bci.load_vocabulary('spanish_words.txt')
```

### Integration with Applications

#### Text Editor Integration
```python
# Send decoded text to active window
from bci_gpt import TextOutput

output = TextOutput(target='active_window')
bci.set_output_handler(output)
```

#### Voice Synthesis
```python
# Convert decoded text to speech
from bci_gpt import VoiceSynthesis

voice = VoiceSynthesis(voice='neural', speed=1.0)
bci.set_voice_output(voice)
```

## Configuration

### Configuration File

Create `~/.bci_gpt/config.yaml`:

```yaml
device:
  type: openbci
  port: /dev/ttyUSB0
  channels: [Fz, Cz, Pz, F3, F4, C3, C4]
  sampling_rate: 1000

processing:
  bandpass_filter: [0.5, 40]
  notch_filter: 60
  artifact_removal: true
  
decoding:
  model: bci-gpt-v1
  confidence_threshold: 0.7
  update_interval: 0.1

output:
  format: text
  target: console
  voice_synthesis: false
```

### Environment Variables

```bash
export BCI_GPT_MODEL_PATH="/path/to/models"
export BCI_GPT_LOG_LEVEL="INFO"
export BCI_GPT_DEVICE_PORT="/dev/ttyUSB0"
```

## Troubleshooting

### Common Issues

#### 1. Device Not Found
```
Error: EEG device not detected
```

**Solutions:**
- Check USB connection
- Verify device drivers
- Check port permissions: `sudo chmod 666 /dev/ttyUSB0`
- Try different USB port

#### 2. Poor Signal Quality
```
Warning: Signal quality below threshold
```

**Solutions:**
- Check electrode contact
- Apply conductive gel
- Clean electrode sites
- Check for loose connections

#### 3. Low Decoding Accuracy
```
Warning: Decoding confidence below 50%
```

**Solutions:**
- Run recalibration
- Improve signal quality
- Use more training data
- Adjust confidence threshold

#### 4. High Latency
```
Warning: Decoding latency > 200ms
```

**Solutions:**
- Reduce buffer size
- Use GPU acceleration
- Close unnecessary applications
- Optimize system performance

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

bci = BCIGPTSystem(debug=True)
```

### Performance Optimization

#### GPU Acceleration
```python
# Enable CUDA if available
bci = BCIGPTSystem(device='cuda')
```

#### Model Optimization
```python
# Use quantized model for faster inference
bci.load_model('bci-gpt-quantized')
```

## FAQ

### General Questions

**Q: How accurate is BCI-GPT?**
A: Accuracy varies by user and setup, typically 70-90% for trained users with good signal quality.

**Q: How long does calibration take?**
A: Initial calibration takes 15-20 minutes. Regular recalibration is recommended weekly.

**Q: Can I use my own EEG device?**
A: Yes, any device compatible with Lab Streaming Layer (LSL) is supported.

### Technical Questions

**Q: What sampling rate is required?**
A: Minimum 250 Hz, recommended 1000 Hz for best performance.

**Q: How many electrodes do I need?**
A: Minimum 3 electrodes (Fz, Cz, Pz), optimal 8+ for better accuracy.

**Q: Can I train custom vocabulary?**
A: Yes, you can train models with custom words and phrases.

### Troubleshooting

**Q: Why is my accuracy low?**
A: Check signal quality, electrode placement, and consider recalibration.

**Q: The system is slow, how can I speed it up?**
A: Use GPU acceleration, reduce buffer size, or use a quantized model.

**Q: Can I use BCI-GPT offline?**
A: Yes, once installed, BCI-GPT works completely offline.

## Getting Help

### Support Channels
- **Documentation**: https://docs.bci-gpt.com
- **GitHub Issues**: https://github.com/danieleschmidt/bci-gpt-inverse-sim/issues
- **Discord Community**: https://discord.gg/bci-gpt
- **Email Support**: support@bci-gpt.com

### Contributing
- Report bugs via GitHub Issues
- Submit feature requests
- Contribute code via Pull Requests
- Improve documentation

---

**Next Steps:**
- Try the [Quick Start Tutorial](./TUTORIAL.md)
- Read the [Developer Guide](./DEVELOPER_GUIDE.md)
- Explore [API Documentation](./API.md)
