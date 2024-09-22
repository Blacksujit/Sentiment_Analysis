 # IMDB Movie Reviews Sentiment Analysis

This project is a sentiment analysis model built to classify IMDB movie reviews as either positive or negative using the **IMDB dataset**. It uses deep learning techniques with LSTM (Long Short-Term Memory) networks to handle the sequential nature of the text data.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Overview

This project performs binary classification on the IMDB dataset, where movie reviews are classified into positive (1) and negative (0) sentiments. It uses TensorFlow and Keras to build and train an LSTM-based model.

The steps involved:
1. Preprocess the data (tokenization and padding).
2. Build an LSTM-based model.
3. Train the model.
4. Evaluate its performance.
5. Save the model for future predictions.

## Dataset

The dataset used is the **[IMDB movie reviews dataset](<IMDB Dataset.csv>)** consisting of 50,000 reviews split equally between training and testing sets.


- **Positive reviews**: 25,000
- **Negative reviews**: 25,000


# Model

The model is built using the following layers:

1.) Embedding layer: Transforms the integer-encoded words into dense word vectors.

2.) LSTM layer: Captures long-term dependencies in the review sequences.

3.) Dropout layer: Helps prevent overfitting by dropping neurons during training.

4.) Dense layer: Sigmoid-activated output layer for binary classification.


# Results:


The LSTM model typically achieves an accuracy of around **64.27%** on the test set.
