import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

# Load the CSV file
df = pd.read_csv('trees_data.csv')

# Extract relevant columns (latitude, longitude, and tree-count)
coords = df[['latitude', 'longitude']].values
tree_counts = df['tree-count'].values

# Calculate weighted coordinates by multiplying latitude and longitude with tree-count
weighted_coords = np.column_stack((coords[:, 0] * tree_counts, coords[:, 1] * tree_counts))

# Normalize the coordinates to avoid scale issues
weighted_coords = weighted_coords / np.max(weighted_coords, axis=0)

# Apply DBSCAN clustering algorithm
dbscan = DBSCAN(eps=0.05, min_samples=5).fit(weighted_coords)

# Add the cluster labels back to the DataFrame
df['cluster'] = dbscan.labels_

# Filter the clusters with the highest concentration of trees
cluster_tree_count = df.groupby('cluster')['tree-count'].sum().sort_values(ascending=False)
top_clusters = cluster_tree_count.head()

# Display the results
print("Top clusters with the highest concentration of trees:")
print(top_clusters)