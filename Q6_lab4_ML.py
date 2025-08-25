# Lab04 - A6
# Repeat A3–A5 using schizophrenia dataset (two features + classes)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Plot training data (like A3)
def plot_training_data(X, y):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolor='k', s=80)
    plt.xlabel("f.mean")
    plt.ylabel("f.sd")
    plt.title("Training Data Scatter Plot (Project Data)")
    plt.show()

# Generate test grid (like A4)
def generate_test_data(step=0.1):
    x_values = np.arange(0, 10.1, step)
    y_values = np.arange(0, 10.1, step)
    xx, yy = np.meshgrid(x_values, y_values)
    X_test = np.c_[xx.ravel(), yy.ravel()]
    return X_test, xx, yy

# Plot classification regions (like A4/A5)
def plot_classification_result(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Generate grid test points
    X_test, xx, yy = generate_test_data()

    # Predict class labels for test points
    y_pred = knn.predict(X_test)

    plt.figure(figsize=(6, 6))
    # Plot decision boundary regions
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.bwr, alpha=0.2, s=10)
    # Overlay training data points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.bwr, edgecolor='k', s=80)
    plt.xlabel("f.mean")
    plt.ylabel("f.sd")
    plt.title(f"kNN Classification with Project Data (k={k})")
    plt.show()

# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # Load your dataset
    data = pd.read_excel(r"C:\Users\anite\Downloads\schizophrenia-features.csv.xlsx")

    # Select two features and labels
    X = data[["f.mean", "f.sd"]].values
    y = data["class"].values

    # Split into train/test (optional, for fairness)
    X_train, X_test_real, y_train, y_test_real = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step A3: Scatter plot of training data
    plot_training_data(X_train, y_train)

    # Step A4: Classification with k=3
    plot_classification_result(X_train, y_train, k=3)

    # Step A5: Repeat with different k values
    for k in [1, 3, 5, 7, 9]:
        plot_classification_result(X_train, y_train, k)
