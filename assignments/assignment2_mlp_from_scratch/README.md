# Assignment 2: 2-Layer MLP From Scratch (No nn.Sequential)

## Objective
Implement a 2-layer Multi-Layer Perceptron (MLP) from scratch using PyTorch, without using `nn.Sequential`. This will help you understand the internal workings of neural networks.

## Tasks
1. Implement a 2-layer MLP class manually defining all layers
2. Implement forward pass
3. Train the network on a dataset (MNIST or CIFAR-10)
4. Implement custom weight initialization
5. Add regularization techniques (optional)
6. Visualize learned features

## Architecture
```
Input Layer (784 for MNIST or 3072 for CIFAR-10)
    ↓
Hidden Layer 1 (e.g., 256 neurons) + ReLU activation
    ↓
Output Layer (10 classes) + Softmax
```

## Requirements
- **No `nn.Sequential`** - Define layers explicitly in `__init__`
- Use `nn.Linear`, `nn.ReLU`, etc., but manually connect them
- Implement forward pass explicitly
- Use proper weight initialization (Xavier/He)

## Files
- `mlp.py` - 2-layer MLP implementation
- `train.py` - Training script
- `evaluate.py` - Evaluation and testing
- `mlp_notebook.ipynb` - Interactive notebook
- `utils.py` - Helper functions

## Learning Goals
- Understand the structure of neural networks
- Learn proper layer initialization
- Practice implementing forward pass manually
- Understand the difference between explicit and sequential architectures
- Learn debugging techniques for custom models

## Expected Outputs
- Trained 2-layer MLP model
- Training/validation accuracy plots
- Confusion matrix
- Visualized weight matrices

## Grading Criteria
- Correct implementation without nn.Sequential ✓
- Proper forward pass implementation ✓
- Achieves reasonable accuracy (>85% on MNIST) ✓
- Clean, well-documented code ✓

## Resources
- [PyTorch nn.Module Tutorial](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Weight Initialization](http://cs231n.github.io/neural-networks-2/#init)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [What is torch.nn really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html)
