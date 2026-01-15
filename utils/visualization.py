"""
Visualization utilities for CS221n assignments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def visualize_grid(X: np.ndarray, ubound: float = 255.0, padding: int = 1) -> np.ndarray:
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Args:
        X: Data of shape (N, H, W, C)
        ubound: Upper bound for pixel values
        padding: Amount of padding between images

    Returns:
        Grid image
    """
    N, H, W, C = X.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))

    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = X[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding

    return grid


def show_images(X: np.ndarray, titles: Optional[List[str]] = None,
                cols: int = 4, figsize: tuple = (12, 8)) -> None:
    """
    Display a grid of images.

    Args:
        X: Images of shape (N, H, W, C) or (N, H, W)
        titles: Optional list of titles for each image
        cols: Number of columns in the grid
        figsize: Figure size
    """
    N = X.shape[0]
    rows = int(np.ceil(N / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if N > 1 else [axes]

    for i in range(N):
        img = X[i]
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze()

        if img.ndim == 2:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img.astype('uint8'))

        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    # Hide any unused subplots
    for i in range(N, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, metrics: List[str] = ['loss', 'accuracy']) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary containing training history
        metrics: List of metrics to plot
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            axes[i].plot(history[train_key], label=f'Train {metric}')
        if val_key in history:
            axes[i].plot(history[val_key], label=f'Val {metric}')

        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} over Epochs')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_weights(W: np.ndarray, num_classes: int = 10,
                     class_names: Optional[List[str]] = None) -> None:
    """
    Visualize learned weights as images.

    Args:
        W: Weight matrix of shape (input_dim, num_classes)
        num_classes: Number of classes
        class_names: Optional class names
    """
    w_grid = W.T.reshape(num_classes, 32, 32, 3)

    # Normalize weights for visualization
    w_min, w_max = w_grid.min(), w_grid.max()
    w_grid = 255.0 * (w_grid - w_min) / (w_max - w_min)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_classes):
        axes[i].imshow(w_grid[i].astype('uint8'))
        axes[i].axis('off')
        title = class_names[i] if class_names else f'Class {i}'
        axes[i].set_title(title)

    plt.tight_layout()
    plt.show()
