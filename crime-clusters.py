import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

# Step 1: Import the CSV file
data = pd.read_csv('crime_data.csv')

# Step 2: Filter the data for Gunshot Wound Victims using .loc to avoid the warning
crime_type = data.loc[data['Type'] == 'Gunshot Wound Victims'].copy()

# Step 3: Extract latitude and longitude
coordinates = crime_type[['Latitude', 'Longitude']].values

# Step 4: Apply DBSCAN clustering algorithm
# Adjust the parameters as needed. eps is the maximum distance between two points to be considered in the same neighborhood.
db = DBSCAN(eps=0.01, min_samples=5).fit(coordinates)

# Step 5: Add cluster labels to the crime_type dataframe
crime_type['cluster'] = db.labels_

# Step 6: Group by cluster and count the number of incidents in each cluster
cluster_counts = crime_type.groupby('cluster').size().reset_index(name='count')

# Step 7: Sort clusters by the number of incidents
sorted_clusters = cluster_counts.sort_values(by='count', ascending=False)

# Display the clusters with the highest concentration of incidents
print(sorted_clusters.head())

# Step 8: Optionally, save the clustered data with labels
crime_type.to_csv('gsw_clusters.csv', index=False)