import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import folium

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

# Create a map centered around the mean of the coordinates
map_center = [df['latitude'].mean(), df['longitude'].mean()]
tree_map = folium.Map(location=map_center, zoom_start=12)

# Generate colors for each cluster
clusters = df['cluster'].unique()
colors = folium.colormap.linear.Set1_09.scale(0, len(clusters))

# Plot each cluster on the map
for cluster in clusters:
    cluster_data = df[df['cluster'] == cluster]
    for _, row in cluster_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5 + row['tree-count'] / 10,  # Adjust radius based on tree count
            color=colors(cluster),
            fill=True,
            fill_opacity=0.6,
            popup=f"Cluster: {cluster}, Tree Count: {row['tree-count']}"
        ).add_to(tree_map)

# Save the map as an HTML file
tree_map.save('tree_clusters_map.html')

# Display the map inline (optional, works in Jupyter notebooks)
# tree_map