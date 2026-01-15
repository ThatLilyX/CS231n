# CS221n: Deep Learning for Computer Vision

Stanford University CS221n coursework and projects repository.

## Course Overview

CS221n (also known as CS231n) is Stanford's course on Convolutional Neural Networks for Visual Recognition. This repository contains assignments, projects, and personal notes from the course.

## Repository Structure

```
CS221n/
├── assignments/          # Course assignments
│   ├── assignment1/     # Image Classification, kNN, SVM, Softmax
│   ├── assignment2/     # Fully Connected Nets, Batch Normalization, Dropout, CNNs
│   └── assignment3/     # RNNs, LSTMs, Network Visualization, Style Transfer, GANs
├── projects/            # Course projects and experiments
├── notes/               # Lecture notes and summaries
├── datasets/            # Downloaded datasets (gitignored)
├── models/              # Trained models (gitignored)
└── utils/               # Shared utility functions

```

## Setup Instructions

### 1. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Datasets will be automatically downloaded when running assignments, or you can manually download them to the `datasets/` directory.

## Dependencies

Core libraries:
- PyTorch / TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook
- Pillow (PIL)
- scikit-learn

See [requirements.txt](requirements.txt) for complete list.

## Assignments

### Assignment 1: Image Classification
- k-Nearest Neighbor classifier
- Support Vector Machine (SVM)
- Softmax classifier
- Two-layer neural network

### Assignment 2: Neural Networks
- Fully-connected neural networks
- Batch normalization
- Dropout
- Convolutional neural networks

### Assignment 3: Advanced Topics
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM)
- Network visualization
- Style transfer
- Generative Adversarial Networks (GANs)

## Resources

- [Course Website](http://cs231n.stanford.edu/)
- [Lecture Videos (YouTube)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- [Course Notes](http://cs231n.github.io/)
- [Assignment Instructions](http://cs231n.github.io/assignments/)

## Development Workflow

1. Create a new branch for each assignment:
   ```bash
   git checkout -b assignment1
   ```

2. Work on the assignment in the respective directory

3. Commit your changes regularly:
   ```bash
   git add .
   git commit -m "feat: implement kNN classifier"
   ```

4. Merge to main when complete:
   ```bash
   git checkout main
   git merge assignment1
   ```

## Notes

- The `datasets/` and `models/` directories are gitignored to avoid committing large files
- Jupyter notebooks (`.ipynb`) are included but output cells should be cleared before committing
- Use clear commit messages following conventional commits format

## License

This repository is for educational purposes. Course materials belong to Stanford University.

## Author

Lily Xiong
