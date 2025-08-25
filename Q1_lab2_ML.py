import pandas as pd
import numpy as np
from numpy.linalg import pinv

# ===========================
# A1: Purchase Data Analysis
# ===========================

def load_and_clean(file):
    """Load CSV and keep only the relevant numeric columns."""
    df = pd.read_csv(file)

    # Keep only useful columns
    df = df[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]]

    # Drop rows with missing Payment (e.g. last rows with product names)
    df = df.dropna(subset=["Payment (Rs)"])

    return df

def segregate_matrices(df):
    """Segregate data into matrices A and C (AX = C)."""
    A = df.iloc[:, 1:-1].values  # product quantities
    C = df.iloc[:, -1].values    # payments
    return A, C

def vector_space_dimensionality(A):
    return A.shape[1]

def num_vectors(A):
    return A.shape[0]

def rank_matrix(A):
    return np.linalg.matrix_rank(A)

def compute_costs(A, C):
    A_pinv = pinv(A)
    return A_pinv.dot(C)

# ===========================
# Main Program
# ===========================
if __name__ == "__main__":
    # ✅ Your dataset file
    file = r"C:\Users\anite\Downloads\Lab Session Data(Purchase data).csv"

    # Load & clean
    purchase_df = load_and_clean(file)

    # Segregate into A and C
    A, C = segregate_matrices(purchase_df)

    # Perform computations
    print("Dimensionality of vector space:", vector_space_dimensionality(A))
    print("Number of vectors:", num_vectors(A))
    print("Rank of Matrix A:", rank_matrix(A))

    costs = compute_costs(A, C)
    product_names = purchase_df.columns[1:-1]  # exclude Customer & Payment
    print("\nEstimated product costs (per unit):")
    for name, cost in zip(product_names, costs):
        print(f"  {name} : {cost:.2f} Rs")


