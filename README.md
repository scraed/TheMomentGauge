# Moment Gauge

Moment Gauge is a Python library for the maximal entropy moment method based on JAX. This library allows you to efficiently perform moment-based computations in Python using the power of JAX.

## Prerequisites

- You must have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

## Installation

Follow these steps to install Moment Gauge and its dependencies:

### 1. Clone the repository

First, clone the Moment Gauge repository from GitHub using the following command:

```
git clone https://github.com/scraed/TheMomentGauge.git
```

This will create a new directory called `TheMomentGauge` containing the project files.

### 2. Create a Conda environment

To install the required dependencies, create a new Conda environment using the provided `environment.yml` file. First, navigate to the `TheMomentGauge` directory:

```
cd TheMomentGauge
```

Next, create the Conda environment:

```
conda env create -f environment.yml
```

This command will create a new Conda environment with the necessary dependencies.

### 3. Activate the Conda environment

Activate the Conda environment by running:

```
conda activate moment_gauge
```

### 4. Install JAX

JAX is a required dependency for Moment Gauge. Follow the instructions below to install JAX with or without GPU support. Please refer to the [official JAX installation guide](https://github.com/google/jax#installation) for more instructions on installing JAX. You could use the latest version of JAX, but we have only tested the code on JAX 0.4.1. Please report to us if the code does not campatible with the latest version of JAX.

#### CPU Installation

To install a CPU-only version of JAX, run the following commands:

```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

#### GPU Installation (CUDA, installed via pip)

First, install the NVIDIA driver. Next, run the following commands:

```
pip install --upgrade pip

# CUDA 12 installation (wheels only available on Linux)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation (wheels only available on Linux)
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 5. Add the package to PYTHONPATH

In order to use the Moment Gauge library in your Python projects, you need to add the package to your PYTHONPATH. For example, if the directory is located at `/home/user/TheMomentGauge`, the command would be:

```
export PYTHONPATH=$PYTHONPATH:/home/user/TheMomentGauge
```

To make this change permanent, add the above command to your shell's configuration file (e.g., `.bashrc` or `.zshrc`).

## Usage

With Moment Gauge installed, you can now use the library in your Python projects by importing it as follows:

```python
import MomentGauge
```

For more information on using Moment Gauge, refer to the project documentation and examples provided in the repository.