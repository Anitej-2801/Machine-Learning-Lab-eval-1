# Lab04 - A7
# Hyperparameter tuning with GridSearchCV for kNN

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# =========================
# Main Program
# =========================
if __name__ == "__main__":
    # Load dataset
    data = pd.read_excel(r"C:\Users\anite\Downloads\schizophrenia-features.csv.xlsx")

    # Select features (using all three this time)
    X = data[["f.mean", "f.sd", "f.propZeros"]].values
    y = data["class"].values

    # Scale features to 0–1 range (important for kNN)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define kNN model
    knn = KNeighborsClassifier()

    # Define parameter grid for k values
    param_grid = {"n_neighbors": list(range(1, 21))}  # test k = 1 to 20

    # GridSearchCV setup
    grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    # Best k value
    print("Best k:", grid.best_params_)
    print("Best cross-validation accuracy:", grid.best_score_)

    # Evaluate on test set
    best_knn = grid.best_estimator_
    test_accuracy = best_knn.score(X_test, y_test)
    print("Test set accuracy with best k:", test_accuracy)
