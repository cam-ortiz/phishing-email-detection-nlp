# Phishing Email Detection using NLP

## Overview
This project focuses on detecting phishing and social engineering emails using Natural Language Processing (NLP) techniques. The goal is to classify emails as either **phishing** or **legitimate** based on textual content alone.

The project compares classical machine learning approaches with modern transformer-based models and analyzes linguistic patterns commonly used in phishing attacks.

---

## Dataset
We use publicly available labeled email datasets:
- SpamAssassin  
- Enron Spam Dataset  
- Public phishing datasets  

### Preprocessing
- Combine email subject and body  
- Minimal preprocessing to preserve linguistic patterns  
- Lowercasing text  

---

## Modeling Approaches

### Baseline Models
- TF-IDF + Logistic Regression  
- TF-IDF + Naive Bayes  
- TF-IDF + Support Vector Machine (SVM)  

### Transformer Model
- Fine-tuned DistilBERT for email classification  

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score (primary metric)  

---

## Project Structure
    phishing-email-detection-nlp/
    │
    ├── data/
    │   ├── raw/          # Original datasets (SpamAssassin, Enron, etc.)
    │   ├── processed/    # Cleaned and merged data
    │   └── samples/      # Small test datasets for development
    │
    ├── notebooks/        # Jupyter notebooks (EDA, experiments, prototyping)
    │
    ├── src/
    │   ├── data/         # Data loading and preprocessing
    │   ├── features/     # Feature engineering (TF-IDF, linguistic features)
    │   ├── models/       # ML models (baseline + transformer)
    │   ├── pipeline/     # Training and evaluation pipeline
    │   └── utils/        # Config and helper functions
    │
    ├── reports/
    │   └── figures/      # Plots, charts, and evaluation visuals
    │
    ├── requirements.txt
    ├── README.md
    └── main.py           # Entry point for running the project

---

## Installation
    git clone https://github.com/cam-ortiz/phishing-email-detection-nlp.git
    cd phishing-email-detection-nlp
    pip install -r requirements.txt

---

## Usage

Run the main training pipeline:

    python main.py

---

## Team
- Cameron Ortiz  
- Lmar Oria  
- Julian Garcia  
- Yifan Li  
