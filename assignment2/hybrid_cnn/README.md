# Hybrid CNN Sentiment Classifier

## Model Architecture
This model uses a hybrid text classification architecture:
- Embedding layer
- 1D convolution layer
- Global max pooling
- Dense hidden layer
- Dropout
- Output sigmoid-style binary prediction via BCEWithLogitsLoss

## Techniques Applied
- Text cleaning
- Vocabulary construction
- Sequence encoding and padding
- Learned word embeddings
- 1D convolution for phrase-level feature extraction
- Dense layer for higher-level representation
- Dropout regularization
- Adam optimizer

## Performance
Test Accuracy: 0.8696
Test Accuracy: 86.96%

## Why this model
This model is more advanced than a plain CNN because it combines convolutional feature extraction with a deeper dense stage and dropout regularization, making it a simple hybrid approach.