# IMDB Movie Reviews Sentiment Analysis

This project is a sentiment analysis model built to classify IMDB movie reviews as either positive or negative using the **IMDB dataset**. It uses various machine learning models and deep learning techniques to handle the text data.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Overview

This project performs binary classification on the IMDB dataset, where movie reviews are classified into positive (1) and negative (0) sentiments. It uses various machine learning algorithms and an LSTM-based deep learning model for comparison.

The steps involved:
1. Preprocess the data (tokenization, vectorization, and padding).
2. Build models using different algorithms.
3. Train the models.
4. Evaluate their performance.
5. Save the best model for future predictions.

## Dataset

The dataset used is the **[IMDB movie reviews dataset](<IMDB Dataset.csv>)** consisting of 50,000 reviews split equally between training and testing sets.

- **Positive reviews**: 25,000
- **Negative reviews**: 25,000

The dataset is split into training and validation sets for model training and evaluation.

## Models

We tested the following machine learning models on the dataset to determine their performance:

### 1. **Naive Bayes**
   - Model: Multinomial Naive Bayes with TF-IDF vectorization.
   - **Accuracy**: 85.55%

### 2. **Logistic Regression**
   - Model: Logistic Regression with TF-IDF vectorization.
   - **Accuracy**: 89.29%

### 3. **Naive Bayes (Small Dataset)**
   - Model: Multinomial Naive Bayes on a subset of the dataset.
   - **Accuracy**: 84.60%

### 4. **Gradient Boosting**
   - Model: Gradient Boosting Classifier with TF-IDF vectorization.
   - **Accuracy**: 80.35%

### 5. **LSTM (Deep Learning Model)**
   - Model: An LSTM (Long Short-Term Memory) neural network to capture sequential dependencies in text data.
   - **Accuracy**: 64.27%

### Model Architecture (for LSTM):
1. **Embedding Layer**: Converts integer-encoded words into dense vectors.
2. **LSTM Layer**: Captures long-term dependencies in review sequences.
3. **Dropout Layer**: Helps prevent overfitting.
4. **Dense Layer**: Sigmoid-activated output layer for binary classification.

## Results

The LSTM model is a simple neural network architecture, achieving a lower accuracy compared to traditional machine learning models like Naive Bayes and Logistic Regression. Below is a summary of the model performances:

- **Naive Bayes**: 85.55%
- **Logistic Regression**: 89.29%
- **Naive Bayes (Small Dataset)**: 84.60%
- **Gradient Boosting**: 80.35%
- **LSTM**: 64.27%

The best performing model in this case was **Logistic Regression** with an accuracy of **89.29%**.

---

 