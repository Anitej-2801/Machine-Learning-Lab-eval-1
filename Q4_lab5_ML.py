import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv (1).xlsx"
df = pd.read_excel(file_path)

# Remove any leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Select only numerical features for clustering (ignore target variable 'class')
X = df[['f.mean', 'f.sd', 'f.propZeros']]  # all features except 'class', 'user_id', 'class_str'

# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

# Cluster labels assigned to each sample
labels = kmeans.labels_
print("Cluster Labels:")
print(labels)

# Cluster centers
centers = kmeans.cluster_centers_
print("\nCluster Centers:")
print(centers)
