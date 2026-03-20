# bci-gpt-inverse-sim

**BCI-GPT Inverse Simulation** — A lightweight Python library for simulating the inverse problem in Brain-Computer Interface (BCI) systems: mapping EEG signal features back to vocabulary (word) probabilities.

## Overview

This project demonstrates how a linear inverse model can decode synthetic EEG signals into word predictions. The pipeline simulates the full flow:

```
Synthetic EEG → Band Power Features → Inverse Linear Mapping → Top-K Word Decoding
```

This is a research/educational tool for exploring EEG-to-language decoding concepts.

## Architecture

```
bci_gpt_inverse_sim/
├── eeg_sim.py        # EEGSignalSim — synthetic EEG generation (10 words × 8 channels)
├── band_power.py     # BandPowerExtractor — delta/theta/alpha/beta/gamma band powers
├── inverse_mapper.py # InverseMapper — linear W.T @ features + b → logits
├── decoder.py        # TopKDecoder — softmax + top-k selection
└── pipeline.py       # BCIPipeline — end-to-end demo
```

## Components

### EEGSignalSim
Generates synthetic 8-channel EEG signals for 10 vocabulary words. Each word has a unique dominant frequency (1.5 Hz to 45 Hz), with slight per-channel variations and Gaussian noise.

```python
from bci_gpt_inverse_sim import EEGSignalSim

sim = EEGSignalSim(seed=42)
eeg = sim.generate(word_id=3, duration=1.0, sr=256)
# eeg.shape == (8, 256)
```

### BandPowerExtractor
Extracts per-channel power in standard EEG frequency bands using Welch's method:
- **Delta**: 0.5–4 Hz
- **Theta**: 4–8 Hz
- **Alpha**: 8–13 Hz
- **Beta**: 13–30 Hz
- **Gamma**: 30–100 Hz

```python
from bci_gpt_inverse_sim import BandPowerExtractor

extractor = BandPowerExtractor()
features = extractor.extract(eeg, sr=256)
# features.shape == (40,)  — 8 channels × 5 bands
```

### InverseMapper
Maps feature vectors to vocabulary logits via a linear model:
```
logits = W.T @ features + b
```
Weights are randomly initialized (Xavier-style). Can be replaced with trained weights.

```python
from bci_gpt_inverse_sim import InverseMapper

mapper = InverseMapper(n_features=40, n_vocab=10)
logits = mapper.forward(features)
# logits.shape == (10,)
```

### TopKDecoder
Applies softmax to logits and returns the top-k most likely words.

```python
from bci_gpt_inverse_sim import TopKDecoder

decoder = TopKDecoder()
top3 = decoder.decode(logits, k=3)
# [(word_id, probability), ...]
```

### BCIPipeline
End-to-end convenience wrapper:

```python
from bci_gpt_inverse_sim import BCIPipeline

pipeline = BCIPipeline(sr=256, duration=1.0, seed=42)
result = pipeline.run(word_id=5, k=3)
print(result["top_k"])
```

## Quick Start

```bash
pip install numpy scipy
python -m bci_gpt_inverse_sim.pipeline
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- scipy ≥ 1.10

## License

MIT
