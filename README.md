IMDB Sentiment Classification using MLP

This project implements a Multi-Layer Perceptron (MLP) neural network from scratch to classify movie reviews as positive or negative.

Dataset:
IMDB Movie Reviews Dataset (50,000 reviews)

Feature Engineering:
Each review is converted into numerical features using:
- VADER compound polarity score
- TextBlob polarity score

Model:
The neural network is implemented manually (no sklearn or pytorch):
- Input layer: 2 neurons (VADER, TextBlob)
- Hidden layer: 8 neurons (tanh activation)
- Output layer: 1 neuron (sigmoid activation)

Training:
- Binary Cross Entropy loss
- Backpropagation
- Gradient Descent optimization
- 50 epochs

Result:
Test Accuracy ≈ 70%

Note:
Dataset is not included in the repository due to size. It can be downloaded from Kaggle and placed inside the data/ folder.
