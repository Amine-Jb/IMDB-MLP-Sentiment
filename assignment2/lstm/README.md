# LSTM Sentiment Classifier

## Model Architecture
- Embedding layer
- LSTM layer
- Dropout
- Output layer for binary classification

## Techniques Applied
- Text cleaning
- Vocabulary construction
- Sequence encoding and padding
- Learned word embeddings
- Recurrent neural network modeling with LSTM
- Adam optimizer
- Binary cross-entropy with logits loss

## Performance
The final test accuracy is reported in `results.txt`. (69.12%)

## Notes
This model processes each review as an ordered sequence of tokens, which helps it capture contextual dependencies better than non-sequential approaches.