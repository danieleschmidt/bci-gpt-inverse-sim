# BCI-GPT Inverse Simulation

A Python toolkit for simulating EEG signals and decoding them back into vocabulary words — an inverse BCI (Brain-Computer Interface) pipeline.

## Overview

This package implements an end-to-end BCI decoding pipeline:

1. **EEG Signal Simulation** — Generate synthetic multi-channel EEG signals, each with a distinct spectral fingerprint corresponding to a vocabulary word.
2. **Band Power Extraction** — Extract delta, theta, alpha, and beta band power features using FFT.
3. **Inverse Mapping** — Train a 2-layer MLP to map EEG features back to vocabulary logits.
4. **Top-k Decoding** — Decode the most likely vocabulary words from the model's output.

## Vocabulary

10 words: `move`, `stop`, `left`, `right`, `up`, `down`, `yes`, `no`, `hello`, `goodbye`

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from bci_gpt.eeg_sim import EEGSignalSim
from bci_gpt.features import BandPowerExtractor
from bci_gpt.mapper import InverseMapper
from bci_gpt.decoder import full_pipeline

# Generate synthetic EEG data
sim = EEGSignalSim(n_channels=8, sample_rate=256, duration=1.0)
signals, labels = sim.generate_dataset(n_per_word=20)

# Extract features
extractor = BandPowerExtractor(sample_rate=256)
X = extractor.extract_batch(signals)

# Train inverse mapper
mapper = InverseMapper(feature_dim=X.shape[1], vocab_size=10)
mapper.fit(X, labels, epochs=200, lr=0.01)

# Decode a signal
test_signal = sim.generate_word("left")
results = full_pipeline(test_signal, extractor, mapper)
for word, prob in results:
    print(f"{word}: {prob:.3f}")
```

## Run the Demo

```bash
python -m bci_gpt.demo
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## Architecture

```
EEGSignalSim → raw EEG (8 ch × 256 samples)
     ↓
BandPowerExtractor → 32 features (8 channels × 4 bands)
     ↓
InverseMapper (MLP) → 10 logits
     ↓
top_k_decode → [(word, probability), ...]
```

## Modules

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `eeg_sim` | `EEGSignalSim` | Synthetic EEG signal generator |
| `features` | `BandPowerExtractor` | FFT-based band power features |
| `mapper` | `InverseMapper` | 2-layer numpy MLP |
| `decoder` | `top_k_decode`, `full_pipeline` | Vocabulary decoding |
| `demo` | `demo()` | End-to-end demonstration |

## License

MIT
