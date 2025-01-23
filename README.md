# Medical Nodule Classification with Custom ResNet

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1+-orange.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Details](#model-details)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This repository contains a comprehensive Python script for training a classification layer on a custom ResNet model tailored for medical nodule classification. The pipeline includes:

- **Data Processing**: Loading and preprocessing NIfTI medical images, applying lung windowing, and extracting nodules.
- **Model Training**: Training a classification layer on top of a pretrained ResNet model.
- **Cross-Validation**: Performing stratified k-fold cross-validation using SVM and XGBoost classifiers.
- **Evaluation**: Assessing model performance on test data with detailed metrics.

## Features

- Custom ResNet architecture integration (`resnet.py` and `model.py`).
- Support for ResNet-34 and ResNet-50 with pretrained weights.
- Comprehensive data preprocessing for 3D medical images.
- Training pipeline using PyTorch with GPU support.
- Cross-validation with SVM and XGBoost, including ensemble methods.
- Detailed evaluation metrics and confusion matrix generation.
- Saving and loading of model weights and preprocessing scalers.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (optional but recommended for faster training)

### Clone the Repository

git clone https://github.com/yourusername/medical-nodule-classification.git
cd medical-nodule-classificatio

