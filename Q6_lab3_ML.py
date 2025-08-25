import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ---------- Functions ----------
def split_dataset(X, y, test_ratio=0.3, random_state=42):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state)

def train_knn(X_train, y_train, k=3):
    """Train a kNN classifier."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def evaluate_accuracy(model, X_test, y_test):
    """Evaluate model accuracy on the test set."""
    return model.score(X_test, y_test)


# ---------- Main Program ----------
if __name__ == "__main__":
    # Load dataset (.ods file)
    file_path = r"C:\Users\anite\Downloads\proj_dataset (1) (2).ods"
    df = pd.read_excel(file_path, engine="odf")

    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1].values   # all columns except last
    y = df.iloc[:, -1].values    # last column = class labels

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3)

    # Train kNN classifier with k=3
    knn_model = train_knn(X_train, y_train, k=3)

    # Evaluate accuracy
    accuracy = evaluate_accuracy(knn_model, X_test, y_test)

    # Print results
    print(f"Accuracy of kNN (k=3) on test set: {accuracy:.4f}")
