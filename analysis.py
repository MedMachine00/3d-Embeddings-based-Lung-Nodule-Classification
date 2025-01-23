#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Script for Training a Classification Layer on Custom ResNet Model,
Performing Train-Test-Val Split, and Cross-Validation with SVM and XGBoost.

Created on [Date]

Author: biomedialab
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skimage.transform import resize
from tqdm import tqdm
from collections import Counter
import joblib
import csv
import confusion_matrix

# Import your custom ResNet model
from resnet import ResNet, Bottleneck, BasicBlock  # Ensure resnet.py is in the same directory or adjust the path accordingly

# Define a modified model with classification layer
class ResNetWithClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2, dropout_p=0.2):
        super(ResNetWithClassifier, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(base_model.embedding_fc.out_features, num_classes)  # Corrected line

    def forward(self, x):
        embeddings = self.base_model(x)  # Get embeddings from base model
        embeddings = self.dropout(embeddings)
        logits = self.fc(embeddings)
        return embeddings, logits  # Ensure two outputs are returned

# Function to load NIfTI files
def load_nii(file_path):
    """Load a NIfTI file and return its data as a NumPy array."""
    nii_img = nib.load(file_path)
    return nii_img.get_fdata(), nii_img.affine

# Function to apply lung windowing
def apply_lung_window(ct_image, window_width, window_level):
    """
    Apply lung windowing using window width and window level.

    Args:
        ct_image (np.ndarray): CT scan data in HU.
        window_width (int): Width of the window.
        window_level (int): Level of the window.

    Returns:
        np.ndarray: Normalized CT image in the range [0, 1].
    """
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)

    # Apply windowing and normalization
    ct_image = np.clip(ct_image, lower_bound, upper_bound)
    ct_image = (ct_image - lower_bound) / (upper_bound - lower_bound)
    return ct_image

# Function to compute bounding box around the nodule
def get_bounding_box(nodule_mask):
    """
    Compute the tightest bounding box around the nodule and expand it by 30% in every dimension.

    Args:
        nodule_mask (np.ndarray): Binary mask of the nodule.

    Returns:
        tuple: Slices for depth, height, and width.
    """
    coords = np.array(np.nonzero(nodule_mask))
    if coords.size == 0:
        return None
    
    # Compute tightest bounding box
    min_d, min_h, min_w = coords.min(axis=1)
    max_d, max_h, max_w = coords.max(axis=1) + 1  # +1 to include the max index

    # Compute the margins (30% of the size in each dimension)
    depth_margin = int(0.3 * (max_d - min_d))
    height_margin = int(0.3 * (max_h - min_h))
    width_margin = int(0.3 * (max_w - min_w))

    # Expand the bounding box
    min_d = max(0, min_d - depth_margin)
    min_h = max(0, min_h - height_margin)
    min_w = max(0, min_w - width_margin)
    max_d += depth_margin
    max_h += height_margin
    max_w += width_margin

    return (slice(min_d, max_d), slice(min_h, max_h), slice(min_w, max_w))


# Function to resize nodule to target size
def resize_to_target(nodule, target_size):
    """
    Resize a 3D nodule to the target size using cubic interpolation.

    Args:
        nodule (np.ndarray): 3D nodule array to resize.
        target_size (tuple): Target size (depth, height, width).

    Returns:
        np.ndarray: Resized 3D nodule.
    """
    # Ensure input and target dimensions match (D, H, W)
    assert len(nodule.shape) == 3, f"Expected 3D input, got shape {nodule.shape}"
    assert len(target_size) == 3, f"Expected target size as (D, H, W), got {target_size}"

    # Use order=3 for cubic interpolation
    resized_nodule = resize(
        nodule,
        output_shape=target_size,
        order=3,  # Cubic interpolation
        mode="constant",
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized_nodule

# Function to process nodules
def process_nodules(mask, ct_image, target_size, window_width, window_level):
    """
    Extract individual nodules based on the mask and resize them.

    This function assumes that each nodule in the mask has a unique label.
    """
    nodules = []
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # Assuming 0 is background

    for label in unique_labels:
        nodule_mask = (mask == label).astype(np.uint8)
        if np.sum(nodule_mask) == 0:
            continue

        # Compute the bounding box
        bbox = get_bounding_box(nodule_mask)
        if bbox is None:
            continue

        # Extract the nodule region from the CT image
        nodule_ct = ct_image[bbox] * nodule_mask[bbox]

        # Apply lung windowing
        nodule_ct = apply_lung_window(nodule_ct, window_width, window_level)

        # Optional: Normalize the nodule region before resizing
        nodule_ct = (nodule_ct - np.mean(nodule_ct)) / np.std(nodule_ct) if np.std(nodule_ct) != 0 else nodule_ct

        # Resize the nodule to match the input size
        resized_nodule = resize_to_target(nodule_ct, target_size)

        # Ensure the resized nodule has the exact target shape
        assert resized_nodule.shape == target_size, f"Resized nodule shape mismatch: {resized_nodule.shape} vs {target_size}"

        nodules.append(resized_nodule)

    return nodules

# Function to traverse directories and collect file paths
def traverse_directories(data_root, class_dirs):
    """
    Traverse the directory structure and return paired mask and CT image paths along with patient IDs.

    Args:
        data_root (str): Root directory containing class directories.
        class_dirs (list): List of class directory names (e.g., ['0-filtered', '1-filtered']).

    Returns:
        list: List of tuples (mask_path, ct_path, class_label, patient_id).
    """
    paired_files = []

    for class_label, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_root, class_dir)
        patient_dirs = glob.glob(os.path.join(class_path, "PETCT_*"))

        for pdir in patient_dirs:
            mask_path = glob.glob(os.path.join(pdir, "**", "filtered_components.nii.gz"), recursive=True)
            ct_path = glob.glob(os.path.join(pdir, "**", "CTres.nii.gz"), recursive=True)

            if mask_path and ct_path:
                # Extract patient ID (assuming directory name contains unique patient identifier)
                patient_id = os.path.basename(pdir)
                paired_files.append((mask_path[0], ct_path[0], class_label, patient_id))
            else:
                print(f"Missing files in {pdir}. Skipping.")

    return paired_files

# Function to extract embeddings
def extract_embeddings(model, nodules, device):
    """
    Extract embeddings for a batch of nodules using the model.

    Args:
        model (torch.nn.Module): The ResNet model.
        nodules (list): List of 3D nodule NumPy arrays.
        device (torch.device): The device (CPU or GPU) to use.

    Returns:
        list: List of embeddings for each nodule.
    """
    model.eval()  # Set the model to evaluation mode
    embeddings = []

    with torch.no_grad():  # Disable gradient computation
        for nodule in nodules:
            # Convert NumPy array to PyTorch tensor
            nodule_tensor = torch.from_numpy(nodule).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
            nodule_tensor = nodule_tensor.to(device)

            # Pass through the model and extract embeddings
            embeddings_out, _ = model(nodule_tensor)  # Ignore logits
            embeddings.append(embeddings_out.cpu().numpy().flatten())  # Convert to NumPy and flatten

    return embeddings




# Main function



def main():
    # Define paths and parameters
    data_root = "/mnt/a5a06f50-755f-4873-a813-c52f55bcaa88/TCIA-PET/filter"  # Adjust as needed
    pretrain_path = "/home/biomedialab/Downloads/MedicalNet_pytorch_files2/pretrain/resnet_50_23dataset.pth"  # Adjust as needed
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    input_W, input_H, input_D = 32, 32, 32
    model_depth = 50  # Adjust based on your ResNet variant
    gpu_id = [0]
    class_dirs = ['0-filtered', '1-filtered']
    window_width = 1500
    window_level = -600
    num_classes = 2  # Adjust based on your dataset
    embedding_dim = 128  # As per your ResNet definition

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate base ResNet model
    base_model = ResNet(
        Bottleneck, [3, 4, 6, 3],
        sample_input_D=input_D,
        sample_input_H=input_H,
        sample_input_W=input_W,
        shortcut_type='A',
        no_cuda=False,
        embedding_dim=embedding_dim
    )
    base_model.to(device)

    # Wrap base_model with classification layer
    model = ResNetWithClassifier(base_model, num_classes=num_classes, dropout_p=0.2)
    model.to(device)

    # Load pretrained weights into base_model
    if os.path.exists(pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location=device)
        # Assuming the checkpoint contains 'state_dict'
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Load state_dict into base_model
        base_model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded successfully.")
    else:
        print(f"Pretrained weights file not found at {pretrain_path}. Proceeding without pretrained weights.")

    # Freeze base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Define optimizer to train only the classification layer
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Optionally, define a scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Traverse dataset directory
    paired_files = traverse_directories(data_root, class_dirs)
    print(f"Total patient directories found: {len(paired_files)}")

    # Initialize lists to store data
    nodules_list = []      # List to store raw nodules
    labels_list = []       # Corresponding labels

    # Process paired files
    for mask_path, ct_path, patient_label, patient_id in tqdm(paired_files, desc="Processing Files"):
        mask, _ = load_nii(mask_path)
        ct_image, _ = load_nii(ct_path)

        # Process nodules directly from the mask
        nodules = process_nodules(
            mask, ct_image,
            target_size=(input_D, input_H, input_W),
            window_width=window_width,
            window_level=window_level
        )

        if not nodules:
            print(f"No nodules found in {os.path.dirname(mask_path)}. Skipping.")
            continue

        nodules_list.extend(nodules)
        labels_list.extend([patient_label] * len(nodules))

    print(f"Total nodules extracted: {len(nodules_list)}")

    # Convert lists to numpy arrays
    nodules_array = np.array(nodules_list)  # Shape: [num_nodules, D, H, W]
    labels_array = np.array(labels_list)    # Shape: [num_nodules]

    # Verify label distribution
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    label_distribution = dict(zip(unique_labels, counts))
    print(f"Overall label distribution: {label_distribution}")

    # Check if there are at least two classes
    if len(unique_labels) < 2:
        print("Error: The dataset contains only one class. Cannot proceed with training.")
        return

    # Split data into Train, Val, Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        nodules_array,
        labels_array,
        test_size=0.1,
        random_state=42,
        stratify=labels_array
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1111,  # 0.1111 * 0.9 ≈ 0.1 of total data
        random_state=42,
        stratify=y_train_val
    )

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # **Training Loop for Classification Layer**
    # Convert numpy arrays to PyTorch tensors for training the classification layer
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Shape: [batch, 1, D, H, W]
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # **[CORRECTION STARTS HERE]**
    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Shape: [num_samples, 1, D, H, W]
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    # **[CORRECTION ENDS HERE]**

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # Newly added

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Adjust batch size based on GPU memory
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # Newly added

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    num_epochs = 10  # Increased from 5 to 10 for better training

    # Initialize variables for tracking the best model
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = os.path.join(output_dir, "best_resnet_with_classifier.pth")

    print("\n--- Starting Training of Classification Layer ---\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings, logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                embeddings, logits = model(inputs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"\nEpoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}\n")

        # Step the scheduler
        scheduler.step()

        # Save the best model based on validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated at Epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

    print(f"\nTraining completed. Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}\n")

    # Save training and validation metrics to CSV
    metrics_data = []
    for epoch in range(num_epochs):
        metrics_data.append({
            'epoch': epoch + 1,
            'train_loss': train_losses[epoch],
            'train_accuracy': train_accuracies[epoch],
            'val_loss': val_losses[epoch],
            'val_accuracy': val_accuracies[epoch]
        })

    # Define the CSV file path
    metrics_csv_file = os.path.join(output_dir, "training_metrics.csv")

    # Write metrics to CSV
    with open(metrics_csv_file, mode='w', newline='') as csv_file:
        fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for data in metrics_data:
            writer.writerow(data)

    print(f"Saved training metrics to {metrics_csv_file}\n")

    # **Load the Best Model for Evaluation**
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path} for evaluation.\n")
    else:
        print("Best model not found. Using the current trained model for evaluation.\n")

    # **Extract embeddings for all nodules (Train + Val + Test)**
    print("\nExtracting embeddings for all nodules...")
    all_nodules = np.concatenate([X_train, X_val, X_test], axis=0)
    all_labels = np.concatenate([y_train, y_val, y_test], axis=0)
    all_patient_ids = np.concatenate([np.array(['train']*X_train.shape[0]),
                                      np.array(['val']*X_val.shape[0]),
                                      np.array(['test']*X_test.shape[0])], axis=0)

    # Create a DataLoader for all nodules
    all_nodules_tensor = torch.tensor(all_nodules, dtype=torch.float32).unsqueeze(1)  # Shape: [num_nodules, 1, D, H, W]
    all_dataset = TensorDataset(all_nodules_tensor, torch.tensor(all_labels, dtype=torch.long))
    all_loader = DataLoader(all_dataset, batch_size=8, shuffle=False)

    # Extract embeddings
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for inputs, _ in tqdm(all_loader, desc="Extracting Embeddings"):
            inputs = inputs.to(device)
            embeddings, _ = model(inputs)
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # Shape: [num_nodules, embedding_dim]
    print(f"All embeddings shape: {all_embeddings.shape}")

    # Save embeddings and labels
    embeddings_file = os.path.join(output_dir, "all_nodule_embeddings.pkl")
    labels_file = os.path.join(output_dir, "all_nodule_labels.pkl")
    patient_ids_file = os.path.join(output_dir, "all_nodule_patient_ids.pkl")
    with open(embeddings_file, "wb") as f:
        pickle.dump(all_embeddings, f)
    with open(labels_file, "wb") as f:
        pickle.dump(all_labels, f)
    with open(patient_ids_file, "wb") as f:
        pickle.dump(all_patient_ids, f)
    print("Saved all embeddings and labels.\n")

    # Prepare data for cross-validation with SVM and XGBoost
    # We'll use the training set (X_train and y_train) for cross-validation

    # Normalize embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(all_embeddings[:X_train.shape[0]])
    X_val_scaled = scaler.transform(all_embeddings[X_train.shape[0]:X_train.shape[0] + X_val.shape[0]])
    X_test_scaled = scaler.transform(all_embeddings[X_train.shape[0] + X_val.shape[0]:])

    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Save scaler
    scaler_file = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_file)
    print("Saved scaler.\n")

    # Initialize Stratified K-Fold
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize metrics storage
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
    }

    fold_number = 1

    for train_index, test_index in skf.split(X_train_scaled, y_train):
        print(f"\n--- Fold {fold_number} ---")

        X_fold_train, X_fold_test = X_train_scaled[train_index], X_train_scaled[test_index]
        y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

        # Handle class imbalance by using class_weight in classifiers

        # Train SVM with Linear Kernel and class_weight='balanced'
        print("Training SVM with Linear Kernel and class_weight='balanced'...")
        svm = SVC(kernel="linear", probability=True, class_weight='balanced', random_state=42)
        svm.fit(X_fold_train, y_fold_train)

        # Train XGBoost with scale_pos_weight
        counter = Counter(y_fold_train)
        if len(counter) != 2:
            print("Warning: XGBoost expects binary classification with two classes.")
            xgb = None  # Skip XGBoost if not binary
        else:
            neg, pos = counter[0], counter[1]
            scale_pos_weight = neg / pos if pos != 0 else 1
            print(f"XGBoost scale_pos_weight: {scale_pos_weight}")
            xgb = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=scale_pos_weight
            )
            xgb.fit(X_fold_train, y_fold_train)

        # Evaluate models
        print("\nEvaluating models...")
        models_to_evaluate = [("SVM", svm)]
        if xgb is not None:
            models_to_evaluate.append(("XGBoost", xgb))

        for model_name, model_obj in models_to_evaluate:
            y_pred = model_obj.predict(X_fold_test)
            acc = accuracy_score(y_fold_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_fold_test, y_pred, average='weighted')
            print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

            # Store metrics
            metrics['accuracy'].append(acc)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)

        # Ensemble: Averaging Probabilities (if XGBoost is available)
        if xgb is not None:
            print("\nCreating Ensemble with SVM and XGBoost...")
            svm_proba = svm.predict_proba(X_fold_test)
            xgb_proba = xgb.predict_proba(X_fold_test)
            ensemble_proba = (svm_proba + xgb_proba) / 2
            ensemble_pred = np.argmax(ensemble_proba, axis=1)

            # Evaluate Ensemble
            acc = accuracy_score(y_fold_test, ensemble_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_fold_test, ensemble_pred, average='weighted')
            print(f"Ensemble - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

            # Store metrics
            metrics['accuracy'].append(acc)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)

        fold_number += 1

    # Calculate average and standard deviation of metrics
    print("\n=== Cross-Validation Results ===")
    for metric_name, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric_name.capitalize()}: {mean:.4f} ± {std:.4f}")

    # Save cross-validation metrics
    cross_val_metrics_file = os.path.join(output_dir, "cross_validation_metrics.pkl")
    with open(cross_val_metrics_file, "wb") as f:
        pickle.dump(metrics, f)
    print("Saved cross-validation metrics.\n")

    # **Final Evaluation on Test Set using trained classifiers**
    print("\n--- Final Evaluation on Test Set ---")

    # Train SVM on entire training set
    print("\nTraining SVM on entire training set...")
    svm_final = SVC(kernel="linear", probability=True, class_weight='balanced', random_state=42)
    svm_final.fit(X_train_scaled, y_train)

    # Train XGBoost on entire training set
    print("\nTraining XGBoost on entire training set...")
    counter = Counter(y_train)
    if len(counter) == 2:
        neg, pos = counter[0], counter[1]
        scale_pos_weight = neg / pos if pos != 0 else 1
        xgb_final = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight
        )
        xgb_final.fit(X_train_scaled, y_train)
    else:
        xgb_final = None
        print("Warning: XGBoost expects binary classification with two classes.")

    # Evaluate SVM on test set
    print("\nEvaluating SVM on Test Set...")
    y_test_pred_svm = svm_final.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_test_pred_svm)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred_svm, average='weighted')
    print(f"SVM - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Evaluate XGBoost on test set
    if xgb_final is not None:
        print("\nEvaluating XGBoost on Test Set...")
        y_test_pred_xgb = xgb_final.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_test_pred_xgb)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred_xgb, average='weighted')
        print(f"XGBoost - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Evaluate Ensemble on test set
    if xgb_final is not None:
        print("\nCreating Ensemble with SVM and XGBoost on Test Set...")
        svm_proba_test = svm_final.predict_proba(X_test_scaled)
        xgb_proba_test = xgb_final.predict_proba(X_test_scaled)
        ensemble_proba_test = (svm_proba_test + xgb_proba_test) / 2
        ensemble_pred_test = np.argmax(ensemble_proba_test, axis=1)

        # Evaluate Ensemble
        acc = accuracy_score(y_test, ensemble_pred_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, ensemble_pred_test, average='weighted')
        print(f"Ensemble - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # **Evaluate Neural Network on Test Set**
    print("\n--- Neural Network Evaluation on Test Set ---")

    # Create a DataLoader for the neural network evaluation
    nn_test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    nn_test_loader = DataLoader(nn_test_dataset, batch_size=8, shuffle=False)

    # Initialize lists to store predictions and true labels
    nn_all_preds = []
    nn_all_true = []

    with torch.no_grad():
        for inputs, labels in tqdm(nn_test_loader, desc="Neural Network Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            embeddings, logits = model(inputs)

            # Get predictions
            _, preds = torch.max(logits, 1)

            # Append to lists
            nn_all_preds.extend(preds.cpu().numpy())
            nn_all_true.extend(labels.cpu().numpy())

    # Calculate Metrics
    nn_accuracy = accuracy_score(nn_all_true, nn_all_preds)
    nn_precision, nn_recall, nn_f1, _ = precision_recall_fscore_support(nn_all_true, nn_all_preds, average='weighted')
    nn_conf_matrix = confusion_matrix(nn_all_true, nn_all_preds)
    nn_class_report = classification_report(nn_all_true, nn_all_preds, target_names=class_dirs)

    # Display Metrics
    print("\n--- Neural Network Evaluation on Test Set ---")
    print(f"Accuracy: {nn_accuracy:.4f}")
    print(f"Precision: {nn_precision:.4f}")
    print(f"Recall: {nn_recall:.4f}")
    print(f"F1-Score: {nn_f1:.4f}")
    print("\nConfusion Matrix:")
    print(nn_conf_matrix)
    print("\nClassification Report:")
    print(nn_class_report)

    # Save Metrics to a Text File
    nn_metrics_text_file = os.path.join(output_dir, "nn_test_evaluation.txt")
    with open(nn_metrics_text_file, "w") as f:
        f.write("--- Neural Network Evaluation on Test Set ---\n")
        f.write(f"Accuracy: {nn_accuracy:.4f}\n")
        f.write(f"Precision: {nn_precision:.4f}\n")
        f.write(f"Recall: {nn_recall:.4f}\n")
        f.write(f"F1-Score: {nn_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(nn_conf_matrix))
        f.write("\n\nClassification Report:\n")
        f.write(nn_class_report)

    print(f"Saved neural network evaluation metrics to {nn_metrics_text_file}\n")

    # **Save the trained neural network model**
    nn_model_file = os.path.join(output_dir, "resnet_with_classifier.pth")
    torch.save(model.state_dict(), nn_model_file)
    print("Saved neural network model.\n")

if __name__ == "__main__":
    main()
