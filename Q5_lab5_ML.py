import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load the dataset
file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv (1).xlsx"
df = pd.read_excel(file_path)

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Select only numerical features for clustering
X = df[['f.mean', 'f.sd', 'f.propZeros']]  # exclude target variable

# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)

# Cluster labels
labels = kmeans.labels_

# Calculate clustering evaluation metrics
sil_score = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
db_index = davies_bouldin_score(X, labels)

# Print results
print("Clustering Evaluation Metrics:")
print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_index)
