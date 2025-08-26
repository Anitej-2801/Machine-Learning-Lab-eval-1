import numpy as np
import pandas as pd
import os

def calculate_centroid(data):
    return np.mean(data, axis=0)

def calculate_spread(data):
    return np.std(data, axis=0)

def calculate_distance(c1, c2):
    return np.linalg.norm(c1 - c2)

if __name__ == "__main__":
    # ✅ Use your dataset (CSV or XLSX)
    file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv.xlsx"
    
    # ✅ Auto-detect file type
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # ✅ Keep only numeric columns for features
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1].values  # numeric features
    y = df.iloc[:, -1].values  # last column as labels (even if categorical)
    
    # ✅ Pick first 2 classes
    class_labels = np.unique(y)[:2]
    class1_data = X[y == class_labels[0]]
    class2_data = X[y == class_labels[1]]

    # ✅ Compute centroids, spreads, and distance
    centroid1 = calculate_centroid(class1_data)
    centroid2 = calculate_centroid(class2_data)
    spread1 = calculate_spread(class1_data)
    spread2 = calculate_spread(class2_data)
    interclass_dist = calculate_distance(centroid1, centroid2)

    # ✅ Print results
    print("✅ Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Numeric features used:", df.select_dtypes(include=[np.number]).columns.tolist())
    print("\n--- Results for A1 ---")
    print(f"Centroid (Class {class_labels[0]}): {centroid1}")
    print(f"Spread (Class {class_labels[0]}): {spread1}")
    print(f"Centroid (Class {class_labels[1]}): {centroid2}")
    print(f"Spread (Class {class_labels[1]}): {spread2}")
    print(f"Interclass Distance: {interclass_dist:.4f}")


