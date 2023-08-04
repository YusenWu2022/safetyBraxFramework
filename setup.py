"""setup.py for Brax.

Install for development:

  pip intall -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="safety_brax",
    version="0.0.1",
    description=("Safety Brax is a benchmark for differentiable safe RL."),
    author="PKU-MARL",
    author_email="yaodong.yang@pku.edu.cn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PKU-MARL/safety-brax/",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    # scripts=["bin/learn"],
    install_requires=[
        "absl-py",
        "dataclasses",
        "dm_env",
        "etils",
        "flask",
        "flask_cors",
        "flax",
        "grpcio",
        "gym",
        "jax",
        "jaxlib",
        "jaxopt",
        "jinja2",
        "mujoco",
        "numpy",
        "optax",
        "Pillow",
        "pytinyrenderer",
        "scipy",
        "tensorboardX",
        "trimesh==3.9.35",
        "typing-extensions",
        # test
        "pytest>=7.0.0",
        "pre-commit>=2.17.0",
        "isort>=5.10.0",
        "black>=22.1.0",
    ],
    extras_require={
        "develop": ["pytest", "transforms3d"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="JAX reinforcement learning rigidbody physics"
)
