import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ---------- Functions ----------
def split_dataset(X, y, test_ratio=0.3, random_state=42):
    """
    Split dataset into train and test sets.
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Labels
        test_ratio (float): Fraction of data to be used as test set
        random_state (int): Random seed for reproducibility
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state)


# ---------- Main Program ----------
if __name__ == "__main__":
    # ✅ Use your dataset (CSV, XLSX, or ODS)
    file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv.xlsx"

    # ✅ Auto-detect file type
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".ods":
        df = pd.read_excel(file_path, engine="odf")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # ✅ Separate features (X) and labels (y)
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1].values  # only numeric features
    y = df.iloc[:, -1].values  # last column as labels (can be categorical)

    # ✅ Split into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3)

    # ✅ Print results
    print("✅ Dataset loaded successfully")
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)
