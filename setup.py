#!/usr/bin/env python3
"""Setup script for bci-gpt-inverse-sim package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bci-gpt-inverse-sim",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    description="Brain-Computer Interface GPT Inverse Simulator for Imagined Speech",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/bci-gpt-inverse-sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.4.0",
        "transformers>=4.21.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "mne>=1.4.0",
        "pyedflib>=0.1.30",
        "pylsl>=1.16.0",
        "brainflow>=5.8.0",
        "tensorboard>=2.10.0",
        "wandb>=0.13.0",
        "hydra-core>=1.2.0",
        "omegaconf>=2.2.0",
        "rich>=12.0.0",
        "typer>=0.7.0",
        "pydantic>=1.10.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "websockets>=11.0.0",
    ],
    extras_require={
        "neuro": [
            "mne-connectivity>=0.4.0",
            "mne-realtime>=0.1.0",
            "nilearn>=0.10.0",
            "neurols>=0.2.0",
            "neurodsp>=2.2.0",
            "fooof>=1.0.0",
        ],
        "clinical": [
            "cryptography>=3.4.8",
            "pydicom>=2.3.0",
            "nibabel>=4.0.0",
            "SimpleITK>=2.2.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bci-gpt=bci_gpt.cli:main",
            "bci-gpt-train=bci_gpt.training.cli:main",
            "bci-gpt-decode=bci_gpt.decoding.cli:main",
            "bci-gpt-inverse=bci_gpt.inverse.cli:main",
        ],
    },
)