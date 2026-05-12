# Phishing Email Detection using NLP

## Overview

This project builds a Natural Language Processing (NLP) pipeline for detecting phishing and spam emails using both traditional machine learning models and transformer-based deep learning models.

The system supports:

- Spam classification using the Enron spam dataset
- Phishing detection using legitimate Enron emails and phishing emails from the Nazario phishing corpus
- Linguistic feature analysis of phishing language patterns
- Comparison between TF-IDF baseline models and a fine-tuned DistilBERT transformer model

The project evaluates multiple approaches and compares their performance using standard classification metrics.

---

# Features

## Baseline Machine Learning Models
- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes
- TF-IDF + Support Vector Machine (LinearSVC)

## Transformer Model
- Fine-tuned DistilBERT using Hugging Face Transformers

## Linguistic Feature Analysis
The project extracts and summarizes phishing-related linguistic patterns such as:
- Urgency language
- Threat/fear language
- Authority references
- Sensitive information requests
- Action-oriented language

---

# Datasets

## Enron Spam Dataset
Used for traditional ham vs spam classification.

Labels:
- `0 = Ham`
- `1 = Spam`

## Nazario Phishing Corpus
Used for phishing email examples.

For phishing experiments:
- Enron ham emails become the legitimate class
- Nazario emails become the phishing class

Labels:
- `0 = Legitimate`
- `1 = Phishing`

---

# Data Availability

The datasets used in this project are publicly available.

## Dataset Sources

### Enron Spam Dataset
- https://www2.aueb.gr/users/ion/data/enron-spam/

### Nazario Phishing Corpus
- https://monkey.org/~jose/phishing/

The GitHub repository does not include the full dataset files because they are large and are excluded by `.gitignore`.

For instructor submission, the ZIP file includes processed CSV files under:

```text
data/processed/
```

These processed datasets allow the default experiments to run without downloading or preprocessing the raw datasets.

The transformer pipeline downloads pretrained DistilBERT weights from Hugging Face on first use.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/cam-ortiz/phishing-email-detection-nlp.git
cd phishing-email-detection-nlp
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Linux / macOS
```bash
source .venv/bin/activate
```

### Windows
```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Tested with Python 3.13

---

# Running the Project

The main entry point is:

```bash
python main.py
```

---

# Command Line Options

## Dataset Options

```bash
--dataset enron
```
Run ham vs spam classification using the Enron dataset.

```bash
--dataset phishing
```
Run phishing detection using Enron ham + Nazario phishing emails.

---

## Custom CSV Input

A custom CSV can be provided with:

```bash
python main.py --dataset csv --csv path/to/file.csv
```

The CSV must contain:

```text
text,label
```

Note: if `--csv` is provided, the program loads that CSV directly.

---

## Model Options

```bash
--model baseline
```
Run TF-IDF baseline models only.

```bash
--model transformer
```
Run the DistilBERT transformer model only.

```bash
--model all
```
Run both baseline and transformer models.

---

## Additional Options

### Apply Class Balancing

```bash
--balanced
```

### Transformer Hyperparameters

```bash
--epochs 2
--batch-size 16
--max-length 128
--learning-rate 2e-5
--train-subsample 8000
```

Use:

```bash
--train-subsample 0
```

to train on the full training split.

---

# Example Commands

## Run baseline spam classification

```bash
python main.py --dataset enron --model baseline
```

## Run phishing detection with balanced classes

```bash
python main.py --dataset phishing --balanced --model baseline
```

## Run DistilBERT phishing classification

```bash
python main.py --dataset phishing --balanced --model transformer
```

## Run transformer on the full training split

```bash
python main.py --dataset phishing --balanced --model transformer --train-subsample 0
```

## Run all models

```bash
python main.py --dataset phishing --balanced --model all
```

---

# Outputs

Generated outputs are saved under:

```text
reports/results/
reports/figures/
```

The pipeline generates:
- Model comparison CSVs
- Classification reports
- Confusion matrix images
- Linguistic feature summaries for baseline model runs

---

# Project Structure

```text
phishing-email-detection-nlp/
│
├── data/
│   ├── raw/              # Original raw datasets
│   ├── processed/        # Cleaned datasets used for training
│   └── samples/          # Small development datasets
│
├── notebooks/            # Data loading and exploratory notebooks
│
├── reports/
│   ├── figures/          # Confusion matrices and plots
│   └── results/          # CSV metrics and reports
│
├── scripts/
│   └── run_transformer.py
│       # Legacy standalone transformer runner
│       # (main.py is now the preferred entry point)
│
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # TF-IDF and linguistic features
│   ├── models/           # Baseline and transformer models
│   └── pipeline/         # Dataset-building pipeline helpers
│
├── requirements.txt
├── README.md
└── main.py               # Main command-line entry point
```

---

# Team

- Cameron Ortiz
- Lmar Oria
- Julian Garcia
- Yifan Li
