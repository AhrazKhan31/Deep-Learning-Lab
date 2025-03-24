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

# Experiment 4 
# Text Generation using RNN and LSTM

## Objective

The aim of this project is to explore text generation using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models while understanding the impact of different word representation techniques:

**1. One-Hot Encoding**
**2. Trainable Word Embeddings**

The models are trained on a dataset of 100 poems to generate coherent text sequences and compare the performance of both encoding methods.

### Dataset

**Data Source:** A collection of 100 poems stored in poems-100.csv.
**Structure:** Each line contains one line of poetry.

The dataset is preprocessed to tokenize the text into words and prepare input-output pairs for training.

## Part 1: One-Hot Encoding Approach

### 1. Preprocessing

**Tokenization:** Clean the text by removing punctuation and splitting it into words.
**One-Hot Encoding:** Convert each word into a one-hot vector of size equal to the vocabulary.

### 2. Model Architecture

**Input:** One-hot encoded word sequences.
**RNN / LSTM:** The model uses either an RNN or LSTM with multiple layers to predict the next word in a sequence.

### 3. Implementation Steps

**Tokenize and Create Vocabulary:** Tokenize the dataset and create a vocabulary of unique words.
**Convert to One-Hot Encoding:** Transform the tokens into one-hot encoded vectors.
**Define RNN Model:** Create and define an RNN/LSTM model in PyTorch.
**Train Model:** Train the model to predict the next word in a sequence.
**Generate Text:** Use the trained model to generate new text.

## Part 2: Trainable Word Embeddings Approach

### 1. Preprocessing

**Tokenization:** Similar to Part 1, tokenize the text into words.
**Indexing:** Convert words into indexed sequences using word-to-index mapping.

### 2. Model Architecture

**Input:** Indexed word sequences.
**Embedding Layer:** Use an embedding layer to learn word representations during training.
**RNN / LSTM:** Pass the embedded input through the RNN/LSTM model to predict the next word.

### 3. Implementation Steps

**Tokenize and Create Vocabulary:** Tokenize the dataset and create a vocabulary of unique words.
**Convert to Indexed Sequences:** Convert tokens to indexed sequences.
**Define RNN Model with Embedding:** Build a PyTorch RNN/LSTM model with an embedding layer.
**Train Model:** Train the model with embedded word representations.
**Generate Text:** Use the trained model to generate text sequences.
