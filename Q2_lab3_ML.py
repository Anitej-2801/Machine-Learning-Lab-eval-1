import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Functions ----------
def calculate_mean_variance(feature_data):
    """Return mean and variance of a feature column."""
    mean_val = np.mean(feature_data)
    var_val = np.var(feature_data)
    return mean_val, var_val

def plot_histogram(feature_data, feature_name):
    """Plot histogram for a given feature."""
    plt.hist(feature_data, bins=10, edgecolor="black", alpha=0.7)
    plt.title(f"Histogram of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# ---------- Main Program ----------
if __name__ == "__main__":
    # Load dataset (.ods file)
    file_path = r"C:\Users\anite\Downloads\proj_dataset (1) (2).ods"
    df = pd.read_excel(file_path, engine="odf")

    # Pick a feature (example: first column)
    feature_name = df.columns[0]
    feature_data = df[feature_name].values

    # Calculate mean & variance
    mean_val, var_val = calculate_mean_variance(feature_data)

    # Print results
    print(f"Feature Selected: {feature_name}")
    print(f"Mean: {mean_val}")
    print(f"Variance: {var_val}\n")

    # Plot histogram
    plot_histogram(feature_data, feature_name)
