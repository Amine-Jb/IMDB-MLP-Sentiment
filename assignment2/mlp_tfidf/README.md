# MLP with TF-IDF Features

## Model Architecture
This model uses a Multi-Layer Perceptron (MLP) classifier with:
- Input: TF-IDF feature vectors extracted from movie reviews
- Hidden layer: 128 neurons
- Activation function: ReLU
- Optimizer: Adam
- Output: Binary sentiment prediction (positive or negative)

## Techniques Applied
- Text vectorization using TF-IDF
- Maximum 5000 input features
- English stop-word removal
- Train-test split with random_state=42
- Hyperparameter tuning by increasing model complexity compared to Assignment 1

## Performance
- Accuracy: 86.64%

## Improvement over Assignment 1
Assignment 1 used only 2 sentiment features (VADER and TextBlob), which limited performance.
This model uses TF-IDF vectors, which preserve much more information from the full review text,
leading to significantly better results.