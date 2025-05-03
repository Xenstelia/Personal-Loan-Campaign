import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_personal_loan_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Personal_Loan', data=data)
    plt.title('Distribution of Personal Loan Acceptance')
    plt.xlabel('Personal Loan Accepted (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()

def plot_income_vs_loan(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Personal_Loan', y='Income', data=data)
    plt.title('Income vs Personal Loan Acceptance')
    plt.xlabel('Personal Loan Accepted (0 = No, 1 = Yes)')
    plt.ylabel('Income')
    plt.show()

def plot_correlation_heatmap(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()