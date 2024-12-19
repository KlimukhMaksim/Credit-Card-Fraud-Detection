import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_class_distribution(data, column):
    """
    Візуалізує розподіл класів у вибраному стовпці.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=data, palette="viridis", hue=None, legend=False)
    plt.title(f"Distribution of values in a column '{column}'")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

def plot_correlation_matrix(data):
    """
    Візуалізує кореляційну матрицю між усіма числовими змінними.
    """
    plt.figure(figsize=(12, 10))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_time_vs_amount(data):
    fraud = data[data['Class'] == 1]
    genuine = data[data['Class'] == 0]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Шахрайські транзакції
    axs[0].scatter(fraud['Time'], fraud['Amount'], alpha=0.5)
    axs[0].set_title('Fraud')
    axs[0].set_ylabel('Amount')

    # Не шахрайські транзакції
    axs[1].scatter(genuine['Time'], genuine['Amount'], alpha=0.5)
    axs[1].set_title('Genuine')
    axs[1].set_xlabel('Time (in Seconds)')
    axs[1].set_ylabel('Amount')

    plt.tight_layout()
    plt.show()

def plot_fraud_vs_nonfraud(data, feature):
    """
    Порівнює розподіл значень ознаки для шахрайських і не шахрайських транзакцій.
    """
    fraud = data[data["Class"] == 1]  # Шахрайські транзакції
    non_fraud = data[data["Class"] == 0]  # Не шахрайські транзакції

    plt.figure(figsize=(10, 6))
    sns.kdeplot(fraud[feature], label="Fraud", fill=True, color="red")
    sns.kdeplot(non_fraud[feature], label="Non-Fraud", fill=True, color="blue")
    plt.title(f"Distribution {feature} for Fraud vs Non-Fraud")
    plt.legend()
    plt.show()
