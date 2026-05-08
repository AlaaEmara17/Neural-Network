# Machine Learning Analysis - Digit Recognition

## Overview
This project implements and analyzes neural network models for digit recognition using the MNIST dataset. The analysis covers various aspects of training neural networks, including hyperparameter tuning, architecture experimentation, and comparison between fully connected networks and convolutional neural networks (CNNs).

## Dataset
- **Source**: MNIST training dataset (`mnist_train.csv`)
- **Features**: 784 pixel values (28x28 grayscale images)
- **Labels**: Digits 0-9
- **Split**: 60% training, 20% validation, 20% testing (with stratification)

## Models Implemented

### 1. Fully Connected Neural Network (`digit_recognition`)
- **Architecture**: 784 → 128 → 64 → 32 → 10
- **Activation**: ReLU
- **Initialization**: Kaiming normal for weights, zeros for biases
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: SGD

### 2. Convolutional Neural Network (`CNN_Digit_Recognition`)
- **Convolutional Layers**:
  - Conv2d(1, 16, 3x3) + ReLU + MaxPool2d(2)
  - Conv2d(16, 32, 3x3) + ReLU + MaxPool2d(2)
- **Fully Connected Layers**:
  - 1568 → 128 → 64 → 10
- **Regularization**: LayerNorm, Dropout(0.3)
- **Features**: Translation invariance, parameter efficiency

## Experiments Conducted

### 1. Learning Rate Analysis
- Tested learning rates: 0.1, 0.01, 0.001, 0.0001
- **Findings**:
  - High LR (0.1): Fast convergence but oscillatory
  - Medium LR (0.01): Balanced convergence and stability
  - Low LR (0.001): Smooth but slow convergence
  - Very low LR (0.0001): Insufficient learning

### 2. Batch Size Analysis
- Tested batch sizes: 16, 32, 64, 128
- **Findings**:
  - Small batches: Noisier gradients, better generalization
  - Medium batches (64): Good balance of speed and quality
  - Large batches: Smoother loss curves, slightly worse generalization

### 3. Architecture Analysis
- **Width Experiment**: Fixed 3 layers, varying neuron counts
  - [64,32,16], [128,64,32], [256,128,64], [512,256,128]
- **Depth Experiment**: Fixed ~128 neurons, varying layer count
  - 1, 2, 3, 4 hidden layers
- **Findings**: Medium architectures perform best on MNIST

## Key Concepts Covered

### Training Dynamics
- Loss calculation and accumulation
- Accuracy computation
- Training vs validation monitoring
- Overfitting prevention

### Neural Network Fundamentals
- Forward/backward propagation
- Gradient descent optimization
- Weight initialization
- Activation functions

### Convolutional Networks
- Spatial feature extraction
- Parameter sharing
- Translation invariance
- Pooling operations

## Results
- **Best Fully Connected Model**: ~95-97% test accuracy
- **CNN Model**: Improved performance with fewer parameters
- Training curves and confusion matrices generated for analysis

## Dependencies
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Usage
1. Load and preprocess MNIST data
2. Initialize model (FC or CNN)
3. Train with `training_step()` method
4. Evaluate using `evaluate_on_test()`
5. Visualize results with `plot_history()`

## Files
- `lab2-Analysis.ipynb`: Complete analysis notebook
- `mnist_train.csv`: MNIST training dataset
- `training_curves.png`: Training visualization (generated)
- `CNN_training_curves.png`: CNN training visualization (generated)