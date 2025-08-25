# Lab04 - A1
# Confusion Matrix and Performance Metrics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Function to compute confusion matrix
def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# Function to compute precision, recall and F1-score
def compute_classification_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # 👉 Load your dataset file (use raw string r"" to avoid path errors)
    data = pd.read_excel(r"C:\Users\anite\Downloads\schizophrenia-features.csv.xlsx")

    # 👉 Select features (X) and labels (y)
    X = data[["f.mean", "f.sd", "f.propZeros"]]  # input features
    y = data["class"]  # target labels (0 = control, 1 = patient)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a kNN model (you can change n_neighbors later)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Predictions
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    # Training evaluation
    cm_train = compute_confusion_matrix(y_train, y_pred_train)
    metrics_train = compute_classification_metrics(y_train, y_pred_train)

    # Test evaluation
    cm_test = compute_confusion_matrix(y_test, y_pred_test)
    metrics_test = compute_classification_metrics(y_test, y_pred_test)

    # Print results
    print("=== Training Data Evaluation ===")
    print("Confusion Matrix:\n", cm_train)
    print("Metrics:", metrics_train)

    print("\n=== Test Data Evaluation ===")
    print("Confusion Matrix:\n", cm_test)
    print("Metrics:", metrics_test)
