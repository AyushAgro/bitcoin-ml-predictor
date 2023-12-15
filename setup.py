#!/usr/bin/env python3
"""
Setup script for Bitcoin Price Prediction System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bitcoin-ml-predictor",
    version="1.0.0",
    author="Ayush Agrawal",
    author_email="ayushagrwal031220@gmail.com",
    description="A machine learning system for Bitcoin price prediction using GRU neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bitcoin-ml-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "bitcoin-predictor=cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="bitcoin, cryptocurrency, machine learning, neural network, gru, prediction, trading",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bitcoin-ml-predictor/issues",
        "Source": "https://github.com/yourusername/bitcoin-ml-predictor",
        "Documentation": "https://github.com/yourusername/bitcoin-ml-predictor#readme",
    },
) 