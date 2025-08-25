import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv (1).xlsx"
df = pd.read_excel(file_path)

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Select numerical features for clustering
X = df[['f.mean', 'f.sd', 'f.propZeros']]

# List to store inertia (distortions)
distortions = []

# Compute k-means for k = 2 to 19
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    distortions.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8,5))
plt.plot(range(2, 20), distortions, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (Distortion)')
plt.xticks(range(2, 20))
plt.grid(True)
plt.show()
