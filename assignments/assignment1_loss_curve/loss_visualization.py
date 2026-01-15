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


class SimpleNet(nn.Module):
    """
    A simple neural network for regression.

    Args:
        input_dim: Number of input features
        hidden_dim: Number of neurons in hidden layer
        output_dim: Number of output dimensions
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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
    # Generate random input features
    X = torch.randn(n_samples, n_features)

    # Generate target as a linear combination of features plus noise
    true_weights = torch.randn(n_features, 1)
    y = X @ true_weights + noise * torch.randn(n_samples, 1)
    y = y.squeeze()

    return X, y


def create_data_loaders(X: torch.Tensor, y: torch.Tensor,
                       train_ratio: float = 0.8,
                       batch_size: int = 32,
                       shuffle: bool = True) -> Tuple:
    """
    Create train and validation data loaders.

    Args:
        X: Input features
        y: Target values
        train_ratio: Ratio of training data
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import TensorDataset, DataLoader

    n_samples = X.shape[0]
    train_size = int(train_ratio * n_samples)

    # Split data
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


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

        # Iterate through train_loader
        for X_batch, y_batch in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch).squeeze()

            # Compute loss
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item() * X_batch.size(0)

        # Average training loss
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            # Iterate through val_loader
            for X_batch, y_batch in val_loader:
                # Forward pass
                outputs = model(X_batch).squeeze()

                # Compute loss
                loss = criterion(outputs, y_batch)

                # Accumulate loss
                val_loss += loss.item() * X_batch.size(0)

        # Average validation loss
        val_loss /= len(val_loader.dataset)

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

    # Check if validation loss is increasing in the second half
    mid_point = len(val_loss) // 2
    val_trend = val_loss[-1] - val_loss[mid_point]
    train_trend = train_loss[-1] - train_loss[mid_point]

    # Check for unstable training (high variance)
    val_std = np.std(val_loss[-20:]) if len(val_loss) >= 20 else np.std(val_loss)

    # Identify training issues
    if val_std > 0.5 * np.mean(val_loss):
        return "Unstable training: Loss values oscillate wildly (learning rate may be too high)"
    elif val_trend > 0 and train_trend < 0 and final_gap > 0.2:
        return "Overfitting: Validation loss increases while training loss decreases"
    elif final_gap > 0.5:
        return "Overfitting: Large gap between validation and training loss"
    elif train_loss[-1] > 1.0 and train_trend > -0.1:
        return "Underfitting: Both losses remain high and not improving"
    elif train_loss[-1] > 1.0:
        return "Underfitting: Losses are high (may need more training or larger model)"
    else:
        return "Good fit: Training and validation losses converge"


if __name__ == "__main__":
    print("Loss Curve Visualization Assignment")
    print("=" * 50)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1)
    print(f"   Data shape: X={X.shape}, y={y.shape}")

    # 2. Create train/val data loaders
    print("\n2. Creating data loaders...")
    train_loader, val_loader = create_data_loaders(X, y, train_ratio=0.8, batch_size=32)
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Define a simple model
    print("\n3. Defining model...")
    model = SimpleNet(input_dim=10, hidden_dim=64, output_dim=1)
    print(f"   {model}")

    # 4. Train and track losses
    print("\n4. Training model...")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    history = train_with_loss_tracking(model, train_loader, val_loader,
                                      criterion, optimizer, num_epochs=100)

    # 5. Plot loss curves
    print("\n5. Plotting loss curves...")
    plot_loss_curves(history, title="Training and Validation Loss")

    # 6. Experiment with different learning rates
    print("\n6. Comparing different learning rates...")
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    compare_learning_rates(learning_rates,
                          lambda: SimpleNet(10, 64, 1),
                          train_loader, val_loader, num_epochs=100)

    # 7. Analyze results
    print("\n7. Analyzing training behavior...")
    diagnosis = identify_training_issues(history)
    print(f"   {diagnosis}")

    print("\n" + "=" * 50)
    print("Assignment complete!")
