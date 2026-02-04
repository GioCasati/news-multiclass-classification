# News Multi-class Classification — Python

> A reproducible **Python + scikit-learn** pipeline for large-scale news topic classification using TF-IDF features and classical machine learning ensembles.

## Overview

This project tackles automatic categorization of online news articles into **7 thematic classes** using a fully classical NLP approach.
The goal is to design an efficient, interpretable, and scalable pipeline that performs well on **high-dimensional sparse text data** without relying on deep learning.

The work emphasizes:

* robust preprocessing of noisy HTML text
* feature engineering from both content and metadata
* strong linear models for sparse spaces
* ensemble learning for improved generalization

**Final performance:**
**Macro F1-score = 0.732**

---

## Dataset

~100,000 news articles:

* 80k labeled - development set
* 20k unlabeled - evaluation set

Classes:

* International
* Business
* Technology
* Entertainment
* Sports
* General
* Health

### Main challenges

* severe class imbalance (≈ 7.6 : 1)
* noisy HTML and URLs
* high lexical overlap between categories (especially 0-5)
* very high dimensional feature space (~50k+ features after TF-IDF)

---

## Methodology

### 1. Text preprocessing

* HTML stripping with attribute preservation
* URL tokenization
* Unicode normalization
* noise removal & lowercasing
* deduplication via majority voting

### 2. Feature engineering

* TF-IDF (unigrams and bigrams)
* document length statistics
* digit density
* source encoding (publisher bias)
* temporal features (year, weekday)
* page rank metadata

### 3. Models (scikit-learn)

Hard Voting Ensemble:

| Model                   | Motivation                                       |
|-------------------------|--------------------------------------------------|
| Linear SVM              | strong margins in high-dimensional sparse spaces |
| Logistic Regression     | probabilistic, well-calibrated decisions         |
| Multinomial Naive Bayes | efficient, acts as regularizer                   |

The ensemble reduces variance and improves robustness through complementary error patterns.

---

## Results

| Model                    | Macro F1  |
|--------------------------|-----------|
| Linear SVM               | ~0.72     |
| Logistic Regression      | ~0.71     |
| Naive Bayes              | ~0.69     |
| **Hard Voting Ensemble** | **0.732** |

The ensemble's performance was evaluated on the prediction of the evaluation data, not available externally. The scores for the single classifiers are obtained with internal cross-evaluation.

---

## Repository structure

- [`src/`](src) — Python pipeline
- [`notebooks/`](notebooks) — experiments and exploratory analysis  
- [`reports/`](reports) — project report
- [`results/`](results) — plots and evaluation outputs  
- [`data/`](data) — dataset

---

## Tech stack

* Python 3
* scikit-learn
* pandas
* numpy
* matplotlib

---

## Report

A detailed methodological discussion, experiments, and analysis are available in: [Project report (PDF)](reports/News_classification_report.pdf)

---

## Author

Giovanni Casati\
MSc Data Science & Engineering — Politecnico di Torino\
Winter 2026

---

## License

Released under the [`MIT License`](LICENSE)
