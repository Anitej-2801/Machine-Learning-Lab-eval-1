import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Functions ----------
def minkowski_distance(x, y, r):
    """Compute Minkowski distance between two vectors for given r."""
    return np.sum(np.abs(x - y) ** r) ** (1 / r)

def calculate_distances(vec1, vec2, r_range=range(1, 11)):
    """Calculate Minkowski distances for r values from r_range."""
    distances = []
    for r in r_range:
        dist = minkowski_distance(vec1, vec2, r)
        distances.append(dist)
    return distances

def plot_distances(r_range, distances, vec1_id, vec2_id):
    """Plot Minkowski distances against r values."""
    plt.plot(r_range, distances, marker="o", linestyle="-")
    plt.title(f"Minkowski Distance (Vector {vec1_id} vs Vector {vec2_id})")
    plt.xlabel("r (order of Minkowski distance)")
    plt.ylabel("Distance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# ---------- Main Program ----------
if __name__ == "__main__":
    # Load dataset (.ods file)
    file_path = r"C:\Users\anite\Downloads\proj_dataset (1) (2).ods"
    df = pd.read_excel(file_path, engine="odf")

    # Assuming last column is class label, so we take only feature columns
    X = df.iloc[:, :-1].values

    # -------------------------------
    # Choose two feature vectors (rows) from the dataset
    # Change vec1_id and vec2_id to select different rows
    # Example: vec1_id=0, vec2_id=5 will compare row 0 and row 5
    # -------------------------------
    vec1_id, vec2_id = 0, 1
    vec1 = X[vec1_id]
    vec2 = X[vec2_id]

    # Calculate Minkowski distances for r = 1 to 10
    r_values = range(1, 11)
    distances = calculate_distances(vec1, vec2, r_values)

    # Print results
    print(f"Vector {vec1_id}:", vec1)
    print(f"Vector {vec2_id}:", vec2)
    print("Minkowski distances for r=1 to 10:")
    for r, d in zip(r_values, distances):
        print(f"r={r}: {d}")

    # Plot distances
    plot_distances(r_values, distances, vec1_id, vec2_id)
