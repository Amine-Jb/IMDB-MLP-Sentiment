# 1D CNN Sentiment Classifier

## Model Architecture
- Embedding layer
- 1D convolution layer
- Global max pooling
- Dropout
- Output layer for binary classification

## Techniques Applied
- Text cleaning
- Vocabulary construction
- Sequence encoding and padding
- Learned word embeddings
- 1D convolution for phrase-level pattern extraction
- Adam optimizer
- Binary cross-entropy with logits loss

## Performance
Test Accuracy: 0.8789
Test Accuracy: 87.89%

## Notes
This model uses the full IMDB dataset and treats each review as a sequence of word indices.