import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def handle_missing_values(data):
    """Handle missing values in the dataset."""
    # Example: Fill missing values with the median for numerical columns
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column].fillna(data[column].median(), inplace=True)
    
    # Example: Fill missing values with the mode for categorical columns
    for column in data.select_dtypes(include=[object]).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

def detect_and_treat_outliers(data):
    """Detect and treat outliers in the dataset."""
    # Example: Using Z-score method to identify outliers
    from scipy import stats
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    data = data[(z_scores < 3).all(axis=1)]  # Keep rows where z-score is less than 3
    return data

def feature_engineering(data):
    """Perform feature engineering on the dataset."""
    # Example: Create a new feature 'Income_to_Loan_Ratio'
    data['Income_to_Loan_Ratio'] = data['Income'] / (data['Loan_Amount'] + 1e-5)  # Avoid division by zero
    return data

def preprocess_data(filepath):
    """Load and preprocess the data."""
    data = load_data(filepath)
    data = handle_missing_values(data)
    data = detect_and_treat_outliers(data)
    data = feature_engineering(data)
    return data
