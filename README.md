# Credit Card Fraud Detection

This project implements a machine learning model for detecting fraudulent credit card transactions using the `Credit Card Fraud Detection` dataset. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Acknowledgments](#acknowledgments)

---

## Dataset Overview

The dataset contains credit card transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with only 492 frauds out of 284,807 transactions (~0.17%).

Key aspects:
- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset.
- **Amount**: The transaction amount.
- **Class**: Target variable, where `1` represents fraud and `0` represents a legitimate transaction.
- **V1 - V28**: Principal Component Analysis (PCA) transformed features (anonymized).

You can find the dataset on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## Features

- **Data Preprocessing**:
  - Handling missing values (if any).
  - Scaling numerical features (e.g., `Amount`).
  - Balancing the dataset using undersampling/oversampling techniques.

- **Exploratory Data Analysis (EDA)**:
  - Class distribution visualization.
  - Correlation heatmaps.
  - Time vs. Amount analysis.
  - Distribution analysis for fraudulent and non-fraudulent transactions.

- **Model Training**:
  - Logistic Regression, Random Forest, or Neural Network-based models.
  - Evaluation using metrics like Precision, Recall, F1-score, and ROC-AUC.

- **Visualization**:
  - Fraud vs. Non-Fraud distribution plots.
  - Feature importance visualization.
  - Training and validation loss/accuracy curves.

---

## Project Structure

```
credit_card_fraud_detection/
├── data/                   # Dataset (CSV files)
├── src/                    # Source code
│   ├── models/             # Model definitions
│   ├── preprocessing/      # Data preprocessing scripts
│   ├── utils.py            # Helper functions for EDA and visualization
├── notebooks/              # Jupyter Notebooks for experimentation
├── results/                # Model outputs and visualizations
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit_card_fraud_detection.git
   cd credit_card_fraud_detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Data Preprocessing**:
   Run the preprocessing script to prepare the data for training.

   ```bash
   python src/preprocessing/preprocess_data.py
   ```

2. **Model Training**:
   Train the fraud detection model:

   ```bash
   python src/models/train_model.py
   ```

3. **Visualization**:
   Generate EDA plots and performance metrics:

   ```bash
   python src/utils/visualize_results.py
   ```

4. **Jupyter Notebooks**:
   Explore the analysis and model training steps interactively:

   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```

---

## Results

The trained model achieves the following metrics on the test set:

| Metric           | Score   |
|------------------|---------|
| Precision        | 0.98    |
| Recall           | 0.93    |
| F1-score         | 0.95    |
| ROC-AUC          | 0.99    |

Visualizations:
- Fraud vs. Non-Fraud distribution across features.
- Training and validation loss/accuracy curves.
