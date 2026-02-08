"""
Setup script for pinn-sgm-finance package.

This package implements Physics-Informed Neural Networks (PINNs) for solving
the Fokker-Planck equation and integrating theoretical scores with Score-Based
Generative Models (SGMs) for financial applications.
"""

from setuptools import setup, find_packages

setup(
    name="pinn-sgm-finance",
    version="1.0.0",
    author="Bilal Saleh Husain",
    author_email="bhusain@uwo.ca",
    description="Theory-Constrained Score Estimation via PINNs and SGMs for Finance",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    url="https://github.com/bilalsalehhusain/pinn-sgm-finance",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "jupyter>=1.0.0",
            "ipython>=8.12.0",
        ],
    },
)
