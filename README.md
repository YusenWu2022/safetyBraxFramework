# Safety Brax
A safety sub-project to OmniDrone.

## Installation

To install Brax from pypi, install it with:

```
git clone https://github.com/PKU-MARL/safety-brax.git
cd safety-brax
conda create -n safety-brax python=3.8
conda activate safety-brax
pip install -e .

# You need to install pytorch and JAX[GPU] manual.
# Training on NVidia GPU is supported, but you must first install
[CUDA, CuDNN, and JAX with GPU support](https://github.com/google/jax#installation).

Fox Jax with GPU support, you can install from pypi, install it with:
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
