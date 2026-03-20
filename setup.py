from setuptools import setup, find_packages

setup(
    name="bci-gpt-inverse-sim",
    version="0.1.0",
    description="BCI-GPT Inverse Simulation: EEG signal generation and vocabulary decoding",
    author="Daniel Schmidt",
    author_email="danschmidt88@gmail.com",
    url="https://github.com/danieleschmidt/bci-gpt-inverse-sim",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy>=1.20",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
