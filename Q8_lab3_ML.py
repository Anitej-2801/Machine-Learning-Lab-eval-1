import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ---------- Functions ----------
def split_dataset(X, y, test_ratio=0.3, random_state=42):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state)

def train_knn(X_train, y_train, k):
    """Train a kNN classifier with given k."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def evaluate_accuracy(model, X_test, y_test):
    """Evaluate model accuracy on the test set."""
    return model.score(X_test, y_test)

def compare_k_values(X_train, X_test, y_train, y_test, k_range=range(1, 12)):
    """Train kNN for k in k_range and return list of accuracies."""
    accuracies = []
    for k in k_range:
        model = train_knn(X_train, y_train, k)
        acc = evaluate_accuracy(model, X_test, y_test)
        accuracies.append(acc)
    return accuracies

def plot_accuracy(k_range, accuracies):
    """Plot accuracy vs k."""
    plt.plot(k_range, accuracies, marker="o", linestyle="-")
    plt.title("kNN Accuracy vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# ---------- Main Program ----------
if __name__ == "__main__":
    # Load dataset (.ods file)
    file_path = r"C:\Users\anite\Downloads\proj_dataset (1) (2).ods"
    df = pd.read_excel(file_path, engine="odf")

    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1].values   # features
    y = df.iloc[:, -1].values    # labels

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3)

    # Compare k from 1 to 11
    k_range = range(1, 12)
    accuracies = compare_k_values(X_train, X_test, y_train, y_test, k_range)

    # Print comparison
    for k, acc in zip(k_range, accuracies):
        print(f"k={k} → Accuracy = {acc:.4f}")

    # Plot accuracy vs k
    plot_accuracy(k_range, accuracies)
