import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

# Step 1: Import the CSV file
data = pd.read_csv('crime_data.csv')

# Step 2: Filter the data for oak trees
crime_type = data[data['Type'] == 'Gunshot Wound Victims']

# Step 3: Extract latitude and longitude
coordinates = crime_type[['latitude', 'longitude']].values

# Step 4: Apply DBSCAN clustering algorithm
# Adjust the parameters as needed. eps is the maximum distance between two points to be considered in the same neighborhood.
db = DBSCAN(eps=0.01, min_samples=5).fit(coordinates)

# Step 5: Add cluster labels to the oak_trees dataframe
crime_type['cluster'] = db.labels_

# Step 6: Group by cluster and count the number of oak trees in each cluster
cluster_counts = crime_type.groupby('cluster').size().reset_index(name='count')

# Step 7: Sort clusters by the number of oak trees
sorted_clusters = cluster_counts.sort_values(by='count', ascending=False)

# Display the clusters with the highest concentration of oak trees
print(sorted_clusters.head())

# Step 8: Optionally, save the clustered data with labels
crime_type.to_csv('oak_tree_clusters.csv', index=False)