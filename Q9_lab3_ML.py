import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ---------- Functions ----------
def split_dataset(X, y, test_ratio=0.3, random_state=42):
    """Split dataset into train and test sets."""
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state)

def train_knn(X_train, y_train, k=3):
    """Train a kNN classifier."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def evaluate_performance(model, X, y, dataset_name="Test"):
    """Evaluate confusion matrix & classification metrics for a dataset."""
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, digits=4)
    print(f"\nPerformance on {dataset_name} Set:")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)


# ---------- Main Program ----------
if __name__ == "__main__":
    # Load dataset (.ods file)
    file_path = r"C:\Users\anite\Downloads\proj_dataset (1) (2).ods"
    df = pd.read_excel(file_path, engine="odf")

    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1].values   # features
    y = df.iloc[:, -1].values    # labels

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_ratio=0.3)

    # Train kNN classifier with k=3
    knn_model = train_knn(X_train, y_train, k=3)

    # Evaluate on training set
    evaluate_performance(knn_model, X_train, y_train, "Training")

    # Evaluate on test set
    evaluate_performance(knn_model, X_test, y_test, "Test")
