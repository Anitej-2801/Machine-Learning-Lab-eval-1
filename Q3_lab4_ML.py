# Lab04 - A3
# Generate 20 random data points and plot them with class colors

import numpy as np
import matplotlib.pyplot as plt

# Function to generate random dataset
def generate_training_data(n_points=20, low=1, high=10):
    X = np.random.uniform(low, high, (n_points, 2))  # random values for X and Y
    y = np.random.randint(0, 2, n_points)            # random class assignment (0 or 1)
    return X, y

# Function to plot the dataset
def plot_training_data(X, y):
    plt.figure(figsize=(6, 6))
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(X[i, 0], X[i, 1], color='blue', label="Class 0" if i == 0 else "")
        else:
            plt.scatter(X[i, 0], X[i, 1], color='red', label="Class 1" if i == 1 else "")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Training Data Scatter Plot (20 Points)")
    plt.legend()
    plt.grid(True)
    plt.show()

# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # Generate 20 random training points
    X, y = generate_training_data()

    # Plot them
    plot_training_data(X, y)
