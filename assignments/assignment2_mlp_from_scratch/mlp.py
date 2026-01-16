"""
2-Layer MLP Implementation From Scratch (No nn.Sequential)

This module implements a 2-layer Multi-Layer Perceptron without using nn.Sequential.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):
    """
    A 2-layer Multi-Layer Perceptron implemented from scratch.

    Architecture:
        Input -> Linear -> ReLU -> Linear -> Output

    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Number of neurons in hidden layer
        output_dim (int): Number of output classes
        dropout_prob (float): Dropout probability (default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout_prob: float = 0.0):
        super(TwoLayerMLP, self).__init__()

        # Layer 1: Input -> Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Activation function
        self.relu = nn.ReLU()

        # Optional: Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

        # Layer 2: Hidden -> Output
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/He initialization.
        """
        # Hint: Use nn.init.xavier_uniform_ or nn.init.kaiming_uniform_
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Step 1: Flatten input if needed (for images)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, input_dim)
        # Step 2: First layer + activation
        x = self.fc1(x)
        x = self.relu(x)
        # Step 3: Dropout (if using)
        if self.dropout is not None:
            x = self.dropout(x)
        # Step 4: Second layer
        x = self.fc2(x)
        # Step 5: Return output (no softmax here - use CrossEntropyLoss)
        return x

    def get_num_parameters(self) -> int:
        """
        Calculate total number of trainable parameters.

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ThreeLayerMLP(nn.Module):
    """
    A 3-layer MLP for comparison (optional challenge).

    Architecture:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
    """

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int,
                 output_dim: int):
        super(ThreeLayerMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def test_model_architecture():
    """
    Test that the model is implemented correctly without nn.Sequential.
    """
    print("Testing 2-Layer MLP Architecture...")

    # Create a small model
    model = TwoLayerMLP(input_dim=784, hidden_dim=128, output_dim=10)

    # Check that nn.Sequential is not used
    has_sequential = any(isinstance(m, nn.Sequential) for m in model.modules())
    if has_sequential:
        print("❌ FAIL: Model uses nn.Sequential - not allowed!")
        return False

    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, 784)

    try:
        output = model(dummy_input)
        assert output.shape == (batch_size, 10), f"Wrong output shape: {output.shape}"
        print(f"✓ Forward pass works! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ FAIL: Forward pass error: {e}")
        return False

    # Check number of parameters
    num_params = model.get_num_parameters()
    expected_params = 784 * 128 + 128 + 128 * 10 + 10  # weights + biases
    print(f"✓ Total parameters: {num_params:,}")
    print(f"  Expected: {expected_params:,}")

    # Print model architecture
    print("\n✓ Model Architecture:")
    print(model)

    print("\n✅ All tests passed! Model is correctly implemented.")
    return True


if __name__ == "__main__":
    test_model_architecture()
