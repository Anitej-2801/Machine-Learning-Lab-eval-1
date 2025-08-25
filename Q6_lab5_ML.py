import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load the dataset
file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv (1).xlsx"
df = pd.read_excel(file_path)

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Select numerical features for clustering
X = df[['f.mean', 'f.sd', 'f.propZeros']]

# Range of k values to test
k_values = range(2, 11)  # testing k from 2 to 10

# Lists to store metrics
sil_scores = []
ch_scores = []
db_indices = []

# Perform k-means clustering for each k and calculate metrics
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    labels = kmeans.labels_
    
    sil_scores.append(silhouette_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))
    db_indices.append(davies_bouldin_score(X, labels))

# Plot the metrics against k
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, sil_scores, marker='o')
plt.title('Silhouette Score vs k')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o', color='green')
plt.title('Calinski-Harabasz Score vs k')
plt.xlabel('Number of clusters k')
plt.ylabel('CH Score')

plt.subplot(1, 3, 3)
plt.plot(k_values, db_indices, marker='o', color='red')
plt.title('Davies-Bouldin Index vs k')
plt.xlabel('Number of clusters k')
plt.ylabel('DB Index')

plt.tight_layout()
plt.show()
