# CS231n: Deep Learning for Computer Vision

Stanford University CS231n coursework and projects repository.

## Course Overview

CS231n is Stanford's course Deep Learning for Computer Vision. This repository contains assignments from the course.

## Repository Structure

```
CS231n/
├── assignments/          # Course assignments
│   ├── assignment1/     # Plot loss curves
│   ├── assignment2/     # Fully Connected Nets, Batch Normalization, Dropout, CNNs
│   └── assignment3/     # RNNs, LSTMs, Network Visualization, Style Transfer, GANs
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

## Resources

- [Course Website](http://cs231n.stanford.edu/)
- [Lecture Videos (YouTube)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
- [Course Notes](http://cs231n.github.io/)
- [Assignment Instructions](http://cs231n.github.io/assignments/)

## License

This repository is for educational purposes. Course materials belong to Stanford University.

## Author

Lily Xiong
