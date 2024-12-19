# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using a neural network model. The dataset used for this project is highly imbalanced, making it a great challenge to accurately detect fraudulent activities while minimizing false positives.

## Project Structure

```
credit_card_fraud_detection/
├── data/
│   └── creditcard.csv       # Dataset containing transaction data
├── notebooks/
│   └── fraud_detection.ipynb # Jupyter Notebook for exploratory data analysis and testing
├── src/
│   ├── data_preprocessing.py # Scripts for cleaning and preprocessing the data
│   ├── evaluation.py         # Functions to evaluate the performance of the model
│   ├── neural_network.py     # Definition of the neural network architecture
│   └── utils.py              # Utility functions for visualization and analysis
├── results/                  # Directory to save results, plots, and trained models
├── config.py                 # Configuration file for hyperparameters and file paths
├── requirements.txt          # Python dependencies for the project
└── README.md                 # Project documentation
```

## Dataset
The dataset used in this project is "creditcard.csv", which contains the following:
- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: Principal components obtained from PCA transformation.
- **Amount**: Transaction amount.
- **Class**: Label for fraud (1) and non-fraud (0).

## Requirements
To install the required Python packages, run:
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Data Preprocessing
The script `src/data_preprocessing.py` handles data loading, scaling, and splitting into training and testing sets. The key steps include:
- Scaling the "Amount" and "Time" features.
- Splitting the data into training and testing sets while maintaining class balance.

### 2. Exploratory Data Analysis
Use the Jupyter Notebook in `notebooks/fraud_detection.ipynb` to:
- Visualize the class distribution.
- Plot correlation matrices.
- Analyze fraudulent vs. non-fraudulent transactions.

### 3. Model Development
The `src/neural_network.py` script defines the neural network architecture, which includes:
- Input layer based on the preprocessed features.
- Hidden layers with ReLU activations.
- Output layer using a sigmoid activation for binary classification.

Training is handled using:
```python
model = FraudDetectionModel(input_dim=X_train_scaled.shape[1])
model.train(X_train_scaled, y_train, X_test_scaled, y_test, epochs=10, batch_size=32)
```

### 4. Evaluation
The `src/evaluation.py` script evaluates the trained model using:
- Confusion matrix
- Precision, recall, F1-score
- AUC-ROC curve

### 5. Visualization
The `src/utils.py` script provides visualization tools, including:
- Class distribution (`plot_class_distribution`)
- Correlation matrix (`plot_correlation_matrix`)
- Fraud vs. non-fraud feature comparison (`plot_fraud_vs_nonfraud`)

## Results
The trained model achieves the following metrics:
- **Accuracy**: X.X%
- **Precision**: X.X%
- **Recall**: X.X%
- **F1-Score**: X.X%
- **AUC-ROC**: X.X%

(Note: Replace `X.X%` with your actual results.)

## How to Run
1. Preprocess the data:
   ```bash
   python src/data_preprocessing.py
   ```
2. Train the model:
   ```bash
   python src/neural_network.py
   ```
3. Evaluate the model:
   ```bash
   python src/evaluation.py
   ```
