# Lab04 - A4
# Generate ~10,000 test points and classify them with kNN (k=3)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Function to generate training dataset (20 points, 2 features, 2 classes)
def generate_training_data(n_points=20, low=1, high=10):
    X = np.random.uniform(low, high, (n_points, 2))   # random features
    y = np.random.randint(0, 2, n_points)             # random class labels (0 or 1)
    return X, y

# Function to generate test dataset (grid of points)
def generate_test_data(step=0.1):
    x_values = np.arange(0, 10.1, step)
    y_values = np.arange(0, 10.1, step)
    xx, yy = np.meshgrid(x_values, y_values)
    X_test = np.c_[xx.ravel(), yy.ravel()]
    return X_test, xx, yy

# Function to plot classification result
def plot_classification_result(X_test, y_pred, xx, yy, X_train, y_train):
    plt.figure(figsize=(7, 7))
    # Plot decision regions (test set classification)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.bwr, alpha=0.2, s=10)
    # Overlay training data points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.bwr, edgecolor='k', s=80, marker='o')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("kNN Classification (k=3) on Test Grid")
    plt.show()

# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # Step 1: Generate synthetic training data
    X_train, y_train = generate_training_data()

    # Step 2: Train kNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Step 3: Generate test set (grid of ~10,000 points)
    X_test, xx, yy = generate_test_data()

    # Step 4: Predict class labels for test set
    y_pred = knn.predict(X_test)

    # Step 5: Plot results
    plot_classification_result(X_test, y_pred, xx, yy, X_train, y_train)
