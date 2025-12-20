# Chapter 1: The Problem

## Sentiment Analysis

**Goal**: Classify movie reviews as positive or negative.

The IMDB dataset contains 50,000 movie reviews split evenly:
- 25,000 for training
- 25,000 for testing

## The Model: BiLSTM

We use a Bidirectional LSTM because:

1. **Sequential data**: Reviews are sequences of words
2. **Context matters**: "not good" vs "good"
3. **Bidirectional**: Reads forward AND backward

```python
model = Sequential([
    Embedding(vocab_size, 128),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Bidirectional(LSTM(lstm_units // 2)),
    Dropout(dropout_rate),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## The Challenge: Hyperparameter Tuning

We need to find:
- **LSTM units**: 32, 64, 128?
- **Dropout rate**: 0.2, 0.3, 0.5?
- **Learning rate**: 0.001, 0.01?

**Search space**: 10^4+ combinations!

Grid search would take forever. Enter: Nature-Inspired Algorithms.
