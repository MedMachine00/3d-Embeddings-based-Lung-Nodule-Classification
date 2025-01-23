# Medical Nodule Classification with Custom ResNet

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1+-orange.svg)
weights available at https://drive.google.com/drive/folders/1kj-0pYeo_BVQYa8Xy3Na79ii5ZcANDCT?usp=drive_link
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
- [License](#license)

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

## Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install Dependencies

pip install numpy nibabel torch torchvision scikit-learn xgboost scikit-image tqdm joblib

## Quick Start

Prepare Your Data: Ensure your dataset is organized with separate directories for each class, containing NIfTI files for masks and CT images.

Download Pretrained Weights: Place the pretrained ResNet-34 and ResNet-50 weights in the pretrained_weights/ directory.

Configure Paths and Parameters: Update the data_root, pretrain_path, and other parameters in the script as needed.

Run the Training Script:

python train_classifier.py

## Usage
The main script train_classifier.py orchestrates the entire pipeline from data loading to model evaluation. Below is an overview of its key components:

## Command-Line Arguments
You can modify the script to accept command-line arguments using argparse for greater flexibility. Currently, paths and parameters are hardcoded for simplicity.

## Data Loading and Preprocessing

Loading NIfTI Files: Utilizes nibabel to load .nii or .nii.gz files.
Lung Windowing: Applies window width and level adjustments to enhance nodule visibility.
Nodule Extraction: Identifies and extracts nodules from the CT images based on mask files.
Resizing: Resizes 3D nodules to a uniform size suitable for the ResNet model.

## Model Training

Custom ResNet Integration: Loads a custom ResNet model (resnet.py) and appends a classification layer (model.py).
Pretrained Weights: Supports loading pretrained weights for ResNet-34 and ResNet-50.
Training Loop: Trains only the classification layer while freezing the base ResNet model.

## Cross-Validation

Stratified K-Fold: Ensures balanced class distribution across folds.
SVM and XGBoost: Trains these classifiers on the extracted embeddings.
Ensemble Methods: Combines predictions from multiple classifiers for improved performance.


## Data Preparation
Ensure your dataset is organized as follows:

Class Directories: 0-filtered and 1-filtered represent different classes.
Patient Directories: Each PETCT_* directory corresponds to a unique patient.
Files:
filtered_components.nii.gz: Binary mask files indicating nodule locations.
CTres.nii.gz: CT scan images in Hounsfield Units (HU).


## Model Details

Custom ResNet (resnet.py and model.py)
ResNet Variants: Supports ResNet-34 and ResNet-50 architectures.
Classification Layer: Added a fully connected layer with dropout for classification.
Embedding Extraction: Retrieves feature embeddings from the base ResNet model.

## Pretrained Weights

Available Weights:
resnet34_pretrained.pth
resnet50_pretrained.pth
Loading Weights: Ensure the correct path is specified in the script to load these weights.

## Training
To train the classification layer:

Set Parameters: Adjust input dimensions, number of classes, and other hyperparameters in the script.

python train_classifier.py

Monitor Training: The script will output training and validation losses and accuracies for each epoch.

Best Model Saving: The model with the lowest validation loss is saved to the results/ directory.


## Evaluation

After training, the script performs evaluation on the test set and outputs detailed metrics:

Neural Network Evaluation: Provides accuracy, precision, recall, F1-score, confusion matrix, and classification report.
Cross-Validation Metrics: Aggregates performance across all folds for SVM, XGBoost, and ensemble methods.
Final Test Evaluation: Assesses the performance of the trained classifiers on unseen test data.
Results.

## Results 

All results, including model weights, metrics, and evaluation reports, are saved in the results/ directory:

results/
├── best_resnet_with_classifier.pth
├── training_metrics.csv
├── all_nodule_embeddings.pkl
├── all_nodule_labels.pkl
├── all_nodule_patient_ids.pkl
├── scaler.joblib
├── cross_validation_metrics.pkl
├── nn_test_evaluation.txt



## License
This project is licensed under the MIT License. See the LICENSE file for details.
