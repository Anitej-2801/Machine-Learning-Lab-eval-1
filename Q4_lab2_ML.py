# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = r"C:\Users\anite\Downloads\Lab Session Data(thyroid0387_UCI) (1).csv"
df = pd.read_csv(file_path)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Get basic info about data types and missing values
print("\nDataset Info:")
print(df.info())

# Describe numeric attributes
print("\nSummary statistics for numeric attributes:")
print(df.describe())

# Check for missing values
print("\nMissing values in each attribute:")
print(df.isnull().sum())

# Check data types and suggest encoding for categorical attributes
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical columns and suggested encoding:")
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"{col}: {unique_vals}")
    # Suggest encoding
    if len(unique_vals) <= 5:
        print(f"  Suggestion: Label Encoding (likely ordinal)")
    else:
        print(f"  Suggestion: One-Hot Encoding (likely nominal)")

# Study numeric attributes for outliers using boxplot
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nStudying numeric attributes for outliers:")
for col in numeric_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Calculate mean and variance for numeric attributes
print("\nMean and Variance of numeric attributes:")
for col in numeric_cols:
    mean_val = df[col].mean()
    var_val = df[col].var()
    std_val = df[col].std()
    print(f"{col}: Mean = {mean_val:.3f}, Variance = {var_val:.3f}, Std Dev = {std_val:.3f}")

# Study range for numeric variables
print("\nRange for numeric variables:")
for col in numeric_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"{col}: Min = {min_val}, Max = {max_val}")
