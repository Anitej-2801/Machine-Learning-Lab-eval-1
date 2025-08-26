import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ---------- Functions ----------
def split_dataset(X, y, test_ratio=0.3, random_state=42):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state)

def train_knn(X_train, y_train, k=3):
    """
    Train a kNN classifier.
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels (categorical)
        k (int): Number of neighbors
    Returns:
        KNeighborsClassifier: Trained kNN model
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# ---------- Main Program ----------
if __name__ == "__main__":
    # ✅ Use schizophrenia dataset
    file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv.xlsx"
    df = pd.read_excel(file_path)

    # ✅ Keep numeric columns for features
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1].values

    # ✅ Convert last column (labels) into categorical codes
    y = df.iloc[:, -1].astype("category").cat.codes.values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3)

    # Train kNN classifier with k=3
    knn_model = train_knn(X_train, y_train, k=3)

    # Print confirmation
    print("✅ kNN model trained successfully with k=3")
    print("Unique classes in labels:", np.unique(y))

