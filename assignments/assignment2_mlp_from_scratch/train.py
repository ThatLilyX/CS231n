"""
Training script for 2-layer MLP on MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Dict, List
import time

from mlp import TwoLayerMLP


def get_mnist_loaders(batch_size: int = 64, data_dir: str = './datasets'):
    """
    Load MNIST dataset and create data loaders.

    Args:
        batch_size: Batch size for training
        data_dir: Directory to store downloaded data

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader,
                criterion, optimizer, device: str) -> tuple:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # 1. Zero gradients
        optimizer.zero_grad()
        # 2. Forward pass
        output = model(data)
        # 3. Compute loss
        loss = criterion(output, target)
        # 4. Backward pass
        loss.backward()
        # 5. Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total += target.size(0)
        # correct += ...  # Calculate correct predictions
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model: nn.Module, test_loader: DataLoader,
            criterion, device: str) -> tuple:
    """
    Evaluate model on test set.

    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 1. Forward pass
            output = model(data)
            # 2. Compute loss
            loss = criterion(output, target)
            test_loss += loss.item()
            # 3. Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def plot_training_curves(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training and validation curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def train_model(num_epochs: int = 10, batch_size: int = 64,
                learning_rate: float = 0.01, hidden_dim: int = 256):
    """
    Main training function.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Create model
    model = TwoLayerMLP(
        input_dim=28*28,
        hidden_dim=hidden_dim,
        output_dim=10
    ).to(device)

    print(f"Model created with {model.get_num_parameters():,} parameters\n")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Training loop
    print("Starting training...")
    print("=" * 70)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s) - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print("=" * 70)
    print(f"\nTraining complete! Final validation accuracy: {history['val_acc'][-1]:.2f}%\n")

    # Plot results
    plot_training_curves(history)

    # Save model to the models directory
    torch.save(model.state_dict(), 'models/mlp_mnist.pth')
    print("Model saved to 'models/mlp_mnist.pth'")

    return model, history


if __name__ == "__main__":
    # TODO: Adjust hyperparameters as needed
    train_model(
        num_epochs=10,
        batch_size=64,
        learning_rate=0.01,
        hidden_dim=256
    )
