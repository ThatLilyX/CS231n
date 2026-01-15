"""
Loss Curve Visualization Assignment

This module demonstrates how to track and visualize loss curves during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10,
                           noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for regression/classification.

    Args:
        n_samples: Number of samples
        n_features: Number of input features
        noise: Amount of noise to add

    Returns:
        Tuple of (X, y) tensors
    """
    # TODO: Implement synthetic data generation
    # Hint: Use torch.randn for random data
    pass


def train_with_loss_tracking(model: nn.Module, train_loader, val_loader,
                            criterion, optimizer, num_epochs: int = 100) -> Dict[str, List[float]]:
    """
    Train a model and track loss values.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs

    Returns:
        Dictionary with 'train_loss' and 'val_loss' lists
    """
    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        # TODO: Implement training loop
        # - Iterate through train_loader
        # - Forward pass
        # - Compute loss
        # - Backward pass
        # - Update weights
        # - Accumulate loss

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            # TODO: Implement validation loop
            # - Iterate through val_loader
            # - Forward pass
            # - Compute loss
            # - Accumulate loss
            pass

        # Record losses
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return history


def plot_loss_curves(history: Dict[str, List[float]], title: str = "Loss Curves",
                     save_path: str = None) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss'
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def compare_learning_rates(learning_rates: List[float], model_fn,
                          train_loader, val_loader, num_epochs: int = 100) -> None:
    """
    Compare loss curves for different learning rates.

    Args:
        learning_rates: List of learning rates to test
        model_fn: Function that returns a fresh model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
    """
    plt.figure(figsize=(12, 5))

    for lr in learning_rates:
        model = model_fn()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        history = train_with_loss_tracking(model, train_loader, val_loader,
                                          criterion, optimizer, num_epochs)

        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], label=f'LR={lr}', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss for Different Learning Rates', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()


def identify_training_issues(history: Dict[str, List[float]]) -> str:
    """
    Analyze loss curves to identify potential training issues.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss'

    Returns:
        String describing the training behavior
    """
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])

    final_gap = val_loss[-1] - train_loss[-1]

    # TODO: Implement logic to identify:
    # - Overfitting: validation loss increases while training loss decreases
    # - Underfitting: both losses remain high
    # - Good fit: both losses decrease and converge
    # - Unstable training: loss values oscillate wildly

    if final_gap > 0.5:
        return "Overfitting: Validation loss much higher than training loss"
    elif train_loss[-1] > 1.0:
        return "Underfitting: Both losses remain high"
    else:
        return "Good fit: Training and validation losses converge"


if __name__ == "__main__":
    print("Loss Curve Visualization Assignment")
    print("=" * 50)

    # TODO:
    # 1. Generate synthetic data
    # 2. Create train/val data loaders
    # 3. Define a simple model
    # 4. Train and track losses
    # 5. Plot loss curves
    # 6. Experiment with different learning rates
    # 7. Analyze results

    print("\nImplement the TODOs to complete this assignment!")
