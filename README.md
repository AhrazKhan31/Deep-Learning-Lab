# Deep Learning Lab
# EXPERIMENT 1:
# Neural Network from Scratch with NumPy (MNIST Classification)

## Objective

This project demonstrates how to implement a neural network from scratch using Python and NumPy only for digit classification using the MNIST dataset.
It includes a Multi-Layer Neural Network which shows implementation of forward and backward propagation using NumPy.
Activation functions used here are ReLU and Softmax.

# EXPERIMENT 2:
# Neural Network from Scratch with NumPy

## Objective

This project demonstrates how to implement a neural network from scratch using only Python and NumPy on the make_moons dataset from sklearn. We explore two cases:

- **Without Hidden Layers (Single-Layer Perceptron)** – A simple model for binary classification.
- **With Hidden Layers (Multi-Layer Neural Network)** – A model using backpropagation with activation functions (ReLU and Sigmoid).

# EXPERIMENT 3:
# CNN Implementation for Image Classification

## Objective

The objective of this experiment is to implement Convolutional Neural Networks (CNNs) for image classification using the Cats vs. Dogs dataset and the CIFAR-10 dataset. The experiment includes exploring different configurations by varying:

- **Activation Functions**: ReLU, Tanh, and Leaky ReLU
- **Weight Initialization Techniques**: Xavier Initialization, Kaiming Initialization, and Random Initialization
- **Optimizers**: SGD, Adam, and RMSprop

Additionally, we compare the performance of our best CNN models with a pretrained ResNet-18 model.

## Steps to Complete the Task

### 1. CNN Implementation

- Define and experiment with different CNN architectures:
  - Vary the number of convolutional layers and filter sizes.
  - Include pooling layers and fully connected layers.
  - Use dropout and batch normalization for regularization and stability.
- Train the CNN models using different combinations of activation functions, weight initializations, and optimizers.
- Evaluate model performance using accuracy and loss metrics.

### 2. Training and Evaluation

- Train the CNN models on the Cats vs. Dogs and CIFAR-10 datasets.
- Save the best-performing model for each dataset and the model weights.

### 3. Transfer Learning with ResNet-18

- Fine-tune a pretrained ResNet-18 model on both datasets.
- Compare the performance of ResNet-18 with the best-performing CNN models.

## Results

- The models are evaluated based on accuracy and loss.
- The best-performing models are saved.






