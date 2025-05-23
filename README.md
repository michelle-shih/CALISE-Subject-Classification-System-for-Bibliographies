# Automatic Bibliography Subject Prediction System

A multi-label classification system for predicting library subject terms from bibliographic data using text mining and deep learning.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Data](#data)
- [Tools & Libraries](#tools--libraries)
- [Process](#process)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Discussion & Limitations](#discussion--limitations)
- [References](#references)

---

## Overview

This project was developed for the 2022 CALISE Big Data Competition. The goal is to automatically predict subject terms for books using their titles, language, and classification numbers, addressing the challenge that a book may belong to multiple subjects and the complexity of assigning subjects base on multiple decision rules. The system leverages text mining, word embeddings, and neural networks to perform multi-label classification on bibliographic data.

---

## Problem Statement

- Each book may have multiple subject terms (multi-label classification).
- Subject term distribution is imbalanced (some terms are much more frequent).
- Input data includes both Chinese and English records, requiring multilingual processing.
- Need to convert text data into a format suitable for neural network input.

---

## Data

- **Source**: National Chengchi University Library catalog (20,000+ records)
- **Fields**: Title, Language, Permanent Call Number, Subject Terms, Resource Type, Publication Date
- **Label**: 30 subject terms (multi-label, e.g., 'United States', 'History', '哲學', etc.)
- **Classification Systems**: Chinese Classification, Dewey Decimal Classification (DDC), and custom call numbers

---

## Tools & Libraries

- Python 3.7
- Pandas (data processing)
- NumPy (matrix operations)
- NLTK (English tokenization & lemmatization)
- CKIPtagger (Chinese tokenization)
- Gensim (Word2Vec embeddings)
- TensorFlow 2.0 / Keras 2.1.5 (neural network modeling)

---

## Process

1. **Exploratory Analysis**
   - Analyzed subject term distribution and call number patterns
   - Identified data imbalance and preprocessing needs

2. **Preprocessing**
   - Cleaned and tokenized titles (NLTK for English, CKIPtagger for Chinese)
   - Removed stopwords and performed lemmatization
   - Extracted classification numbers (first 3 digits) using regex
   - Encoded language (0: Chinese, 1: English)
   - Built one-hot vectors for subject term labels

3. **Feature Engineering**
   - Used Word2Vec (CBOW) to generate embeddings for titles and subject terms
   - Averaged token embeddings for each title and subject term
   - Combined features: title embeddings, language code, classification embeddings

4. **Modeling**
   - Built a Keras Sequential neural network for multi-label classification
   - Input: concatenated feature vectors
   - Output: 30-dimensional probability vector (one per subject term)
   - Loss: Binary cross-entropy (per-label)

5. **Training & Evaluation**
   - Early stopping to prevent overfitting
   - Evaluated using precision, recall, and F1-score (all ~0.75)
   - Tuned probability threshold for label assignment

---

## Model Architecture

- **Input Layer**: 41 features (title embeddings + language + classification)
- **Dense Layers**: [600, 800] units, ReLU activation, dropout for regularization
- **Output Layer**: 30 units, softmax activation for multi-label output
- **Optimizer**: Adam

Example:

model = keras.Sequential()
model.add(Dense(600, input_dim=41, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
---

## Results

- **Precision**: ~0.74
- **Recall**: ~0.76
- **F1 Score**: ~0.75
- The model effectively predicts multiple subject terms for each book, despite data imbalance.
- Word embeddings improved the model's ability to capture semantic relationships in both Chinese and English titles.

---

## Discussion & Limitations

- **Data imbalance**: Most books have only a few subject terms; common terms dominate.
- **Label independence**: Used sigmoid for independent probabilities, but softmax was also explored.
- **Feature limitations**: Titles, classification numbers, and language provide limited context; including book outlines or summaries could further improve accuracy.
- **Multilingual challenges**: Separate tokenization and embedding strategies were required for Chinese and English.

---

## References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/)
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [CKIPtagger](https://github.com/ckiplab/ckiptagger)
- See full reference list in the project report.

---

**Developed by:**  
Yu-Liang Peng, Xue-Jun Shih, Ping-Yun Huang (NTU LIS)  
Mentor: Sung-Chian Lin  
CALISE Big Data Competition, Registration No. CDA111004
