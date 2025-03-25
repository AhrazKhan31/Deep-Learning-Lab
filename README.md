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

## Models

Two types of recurrent neural network models were implemented:

1. **RNN:** A standard recurrent neural network. RNNs process sequential data by maintaining a hidden state that is updated at each time step based on the current input and the previous hidden state. This allows them to capture some dependencies between words in a sequence.

2. **LSTM:** A long short-term memory network, known for its ability to capture long-range dependencies in sequences. LSTMs are a type of RNN with a more complex internal structure that includes memory cells and gates. These gates control the flow of information into and out of the memory cells, enabling the network to learn and remember long-term dependencies.


## Approaches

Two approaches for representing input text were compared:

1. **One-Hot Encoding:** Each word is represented by a sparse vector with a single '1' at the index corresponding to the word in the vocabulary. While simple to implement, one-hot encoding can lead to very high-dimensional input vectors, especially for large vocabularies. It also doesn't capture any semantic relationships between words.

2. **Trainable Word Embeddings:** Each word is mapped to a dense, lower-dimensional vector that is learned during training. These embeddings capture semantic relationships between words, as words with similar meanings tend to have similar embeddings. This allows the network to learn more meaningful representations of the input text.


## Observations

### Loss

* **RNN:**
    * Surprisingly, the RNN model trained with one-hot encoding showed a *lower* loss compared to the RNN model with trainable word embeddings, contrary to the usual expectation. This could be attributed to several factors:
        * **Dataset Size:** The poetry dataset is relatively small. One-hot encoding might perform adequately in such cases, especially with simpler models like RNNs, as they might not fully utilize the advantages of word embeddings.
        * **Hyperparameters:** The specific model hyperparameters, such as hidden size and learning rate, play a crucial role in performance. The current configuration may favor one-hot encoding for the RNN.
        * **Randomness:**  Weight initialization and data shuffling introduce variability. This particular training run might have favored the one-hot model due to chance.

* **LSTM:**
    * As expected, the LSTM model trained with trainable word embeddings achieved the *lowest* overall loss among all approaches. This result is more aligned with the general benefits of using embeddings:
        * **Long-Range Dependencies:**  LSTMs are better at capturing long-range relationships in text.  Word embeddings provide a more informative input representation, enhancing the LSTM's ability to model these dependencies.
        * **Semantic Information:** Embeddings encode semantic similarities between words, which LSTMs can leverage for better understanding and generation of text.

### Training Time

* **RNN:**
    * The RNN model with one-hot encoding was observed to be the *fastest* to train. This is likely due to the simpler architecture of RNNs and the potential for smaller input size with one-hot encoding (depending on the vocabulary size and embedding dimension).
* **LSTM:**
    * The LSTM model with trainable word embeddings took the *longest* to train. This longer training time is attributed to two main factors:
        * **Complex Architecture:** LSTMs have a more complex internal structure compared to RNNs, which involves more computations per time step.
        * **Embedding Layer:** The addition of a trainable embedding layer introduces more parameters to the model, increasing the overall training time.


## Summary

| Approach | RNN Loss | RNN Training Time | LSTM Loss | LSTM Training Time |
|---|---|---|---|---|
| One-Hot Encoding | **Lower** | Faster | Higher | Moderate |
| Trainable Word Embeddings | Higher | Moderate | Lowest | Slower |

## Conclusion

The experiments demonstrated that while simpler models like RNNs might show acceptable performance with one-hot encoding on smaller datasets, the benefits of trainable word embeddings become more apparent with more complex models like LSTMs. These embeddings enable the network to learn more nuanced representations of the input text, ultimately leading to better performance, particularly in capturing long-term dependencies and semantic meaning.

## Future Work

* **Larger Datasets:** Testing these approaches on larger text corpora would likely further demonstrate the advantages of embeddings as they tend to shine when given more data to learn from.
* **Hyperparameter Tuning:** Conducting a thorough search for optimal hyperparameters for each model and encoding type would help to ensure fair comparison and could potentially improve individual model performance.
* **Validation Data:** Introducing a separate validation set to assess performance on unseen data would allow us to evaluate generalization ability and prevent overfitting to the training data.
* **Advanced Metrics:**  Considering other evaluation metrics like accuracy or perplexity, in addition to loss, would offer a more comprehensive evaluation of text generation quality.
