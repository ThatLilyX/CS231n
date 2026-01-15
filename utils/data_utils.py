"""
Data loading and preprocessing utilities for CS221n assignments.
"""

import numpy as np
import pickle
import os
from typing import Tuple, Dict


def load_cifar10_batch(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single batch of CIFAR-10 data.

    Args:
        filename: Path to the batch file

    Returns:
        Tuple of (images, labels)
        - images: np.ndarray of shape (N, 3072) containing image data
        - labels: np.ndarray of shape (N,) containing labels
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_cifar10(root: str) -> Dict[str, np.ndarray]:
    """
    Load the entire CIFAR-10 dataset.

    Args:
        root: Root directory containing CIFAR-10 data

    Returns:
        Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    xs = []
    ys = []

    # Load training batches
    for b in range(1, 6):
        f = os.path.join(root, f'data_batch_{b}')
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)

    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    # Load test batch
    X_test, y_test = load_cifar10_batch(os.path.join(root, 'test_batch'))

    # Split training data into train and validation
    num_training = 49000
    num_validation = 1000

    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }


def preprocess_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                   subtract_mean: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess image data by normalizing and mean-centering.

    Args:
        X_train: Training images
        X_val: Validation images
        X_test: Test images
        subtract_mean: Whether to subtract the mean image

    Returns:
        Tuple of preprocessed (X_train, X_val, X_test)
    """
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train = X_train - mean_image
        X_val = X_val - mean_image
        X_test = X_test - mean_image

    return X_train, X_val, X_test


def get_data_batch(X: np.ndarray, y: np.ndarray, batch_size: int,
                   shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate batches of data for training.

    Args:
        X: Input data
        y: Labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data

    Yields:
        Batches of (X_batch, y_batch)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]
