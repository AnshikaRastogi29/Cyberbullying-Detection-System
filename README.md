# Explainable Emotion-Aware Cyberbullying Detection System

## Overview

Cyberbullying has become a major concern on social media platforms, affecting the mental health and well-being of users. Traditional cyberbullying detection systems mainly focus on classifying text as bullying or non-bullying but often fail to explain the reason behind their predictions.

This project aims to develop an Explainable Emotion-Aware Cyberbullying Detection System that combines Machine Learning, Natural Language Processing (NLP), Sentiment Analysis, Emotion Detection, Severity Analysis, and Explainable AI (SHAP) to provide transparent and accurate predictions.

---

## Problem Statement

Existing cyberbullying detection models face several limitations:

* Lack of explainability in predictions.
* Limited use of emotion-aware analysis.
* Absence of severity-level classification.
* Difficulty in understanding why a comment is classified as cyberbullying.

This project addresses these challenges by developing an interpretable and emotion-aware cyberbullying detection framework.

---

## Objectives

* Detect cyberbullying content from social media text.
* Perform sentiment analysis on user comments.
* Identify emotions such as anger, sadness, fear, and joy.
* Classify cyberbullying severity levels.
* Provide explainable predictions using SHAP.
* Compare multiple machine learning models and evaluate performance.

---

## Research Gap

Based on the literature review, most existing studies:

* Focus only on cyberbullying classification.
* Do not provide interpretable predictions.
* Rarely combine sentiment analysis and emotion detection.
* Ignore severity-level assessment.

The proposed framework integrates these components into a single explainable system.

---

## Proposed Methodology

1. Data Collection
2. Data Preprocessing
3. Feature Extraction using TF-IDF
4. Sentiment Analysis
5. Emotion Detection
6. Machine Learning Model Training
7. Severity Classification
8. SHAP-based Explainability
9. Performance Evaluation
10. Web-based Deployment using Streamlit

---

## Tech Stack

### Programming Language

* Python

### Natural Language Processing (NLP)

* NLTK
* TextBlob
* Transformers
* TF-IDF Vectorizer

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* XGBoost

### Explainable AI

* SHAP

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Plotly

### Version Control

* Git
* GitHub

---

## Project Structure

```text
cyberbullying_project

├── README.md
├── requirements.txt
├── app.py
├── ROADMAP.md

├── dataset/

├── docs/
│   └── project_proposal.md

├── models/

├── research/
│   ├── literature_review.md
│   └── research_gap.md

├── screenshots/

└── static/
    └── css/
```

---

## Expected Features

* Cyberbullying Detection
* Sentiment Analysis
* Emotion Detection
* Severity Prediction
* SHAP Explainability
* Interactive Streamlit Dashboard
* Research-Oriented Framework

---

## Current Status

* Literature Review Completed
* Research Gap Identified
* Project Planning Completed
* Dataset Collection in Progress
* Streamlit Development Started

---

## Future Work

* Multi-language Cyberbullying Detection
* Real-time Social Media Integration
* Transformer-Based Deep Learning Models
* Advanced Explainability Visualizations
* Enhanced Severity Prediction

---

## Expected Outcome

The final system will be capable of identifying cyberbullying comments, analyzing emotions and sentiments, predicting severity levels, and explaining model decisions through SHAP-based Explainable AI. The project aims to contribute to safer and more transparent social media environments.

---

## Author

ANSHIKA RASTOGI

B.Tech Student | Machine Learning | NLP | Explainable AI
