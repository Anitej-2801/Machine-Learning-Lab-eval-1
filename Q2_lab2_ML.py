import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ===========================
# A2: Customer Classification
# ===========================

# ✅ Your dataset file
file = r"C:\Users\anite\Downloads\Lab Session Data(Purchase data).csv"
df = pd.read_csv(file)

# Keep only relevant columns
df = df[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]]

# Label customers: RICH (>200), POOR (<=200)
df["Class"] = df["Payment (Rs)"].apply(lambda x: "RICH" if x > 200 else "POOR")

# Features (X) and Target (y)
X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
y = df["Class"]

# Encode target labels (RICH=1, POOR=0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train classifier (Logistic Regression)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Show dataset with Class column
print("\nDataset with classification:")
print(df.head())

