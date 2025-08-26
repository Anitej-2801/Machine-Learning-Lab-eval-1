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

def make_predictions(model, X_test):
    """Predict labels for test set and a single test vector."""
    y_pred_all = model.predict(X_test)         # predictions for all test vectors
    y_pred_single = model.predict([X_test[0]]) # prediction for the first test vector
    return y_pred_all, y_pred_single


# ---------- Main Program ----------
if __name__ == "__main__":
    # Load schizophrenia dataset (.xlsx)
    file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv.xlsx"
    df = pd.read_excel(file_path)

    # Keep only numeric columns for features
    X = df.select_dtypes(include=[np.number]).iloc[:, :-1].values

    # Convert last column (labels) to categorical codes
    y = df.iloc[:, -1].astype("category").cat.codes.values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3)

    # Train kNN classifier with k=3
    knn_model = train_knn(X_train, y_train, k=3)

    # Evaluate accuracy
    accuracy = evaluate_accuracy(knn_model, X_test, y_test)
    print(f"✅ Accuracy of kNN (k=3) on test set: {accuracy:.4f}\n")

    # Predictions
    y_pred_all, y_pred_single = make_predictions(knn_model, X_test)

    # Print results
    print("Predictions for first 10 test vectors:")
    print(y_pred_all[:10])

    print("\nActual labels for first 10 test vectors:")
    print(y_test[:10])

    print("\nPrediction for a single test vector (X_test[0]):")
    print(f"Predicted: {y_pred_single[0]}, Actual: {y_test[0]}")
