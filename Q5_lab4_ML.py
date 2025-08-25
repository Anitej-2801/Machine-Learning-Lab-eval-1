# Lab04 - A5
# Repeat A4 for different values of k in kNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate training data (20 random points)
def generate_training_data(n_points=20, low=1, high=10):
    X = np.random.uniform(low, high, (n_points, 2))
    y = np.random.randint(0, 2, n_points)
    return X, y

# Generate test grid (~10,000 points)
def generate_test_data(step=0.1):
    x_values = np.arange(0, 10.1, step)
    y_values = np.arange(0, 10.1, step)
    xx, yy = np.meshgrid(x_values, y_values)
    X_test = np.c_[xx.ravel(), yy.ravel()]
    return X_test, xx, yy

# Plot classification result for given k
def plot_classification_result(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    X_test, xx, yy = generate_test_data()
    y_pred = knn.predict(X_test)

    plt.figure(figsize=(6, 6))
    # Plot decision regions
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.bwr, alpha=0.2, s=10)
    # Plot training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.bwr, edgecolor='k', s=80)
    plt.title(f"kNN Classification (k={k})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # Step 1: Generate training dataset
    X_train, y_train = generate_training_data()

    # Step 2: Try multiple k values
    for k in [1, 3, 5, 7, 9]:
        plot_classification_result(X_train, y_train, k)
