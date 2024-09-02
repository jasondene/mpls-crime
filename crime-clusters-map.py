import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import folium

# Step 1: Import the CSV file
data = pd.read_csv('crime_data.csv')

# Step 2: Filter the data for oak trees
crime_type = data[data['Type'] == 'Gunshot Wound Victims']

# Step 3: Extract latitude and longitude
coordinates = crime_type[['latitude', 'longitude']].values

# Step 4: Apply DBSCAN clustering algorithm
db = DBSCAN(eps=0.01, min_samples=5).fit(coordinates)

# Step 5: Add cluster labels to the oak_trees dataframe
crime_type['cluster'] = db.labels_

# Step 6: Group by cluster and count the number of oak trees in each cluster
cluster_counts = crime_type.groupby('cluster').size().reset_index(name='count')

# Step 7: Sort clusters by the number of oak trees
sorted_clusters = cluster_counts.sort_values(by='count', ascending=False)

# Step 8: Create a map centered around the mean latitude and longitude
map_center = [crime_type['latitude'].mean(), crime_type['longitude'].mean()]
map_crime_type = folium.Map(location=map_center, zoom_start=13)

# Step 9: Add clusters to the map
for cluster in sorted_clusters['cluster']:
    cluster_data = crime_type[crime_type['cluster'] == cluster]

    # Compute the centroid of the cluster for the marker
    centroid = [cluster_data['latitude'].mean(), cluster_data['longitude'].mean()]

    # Create a marker for the cluster centroid
    folium.Marker(
        location=centroid,
        popup=f'Cluster {cluster}: {len(cluster_data)} oak trees',
        icon=folium.Icon(color='green')
    ).add_to(map_crime_type)

    # Optionally, add points for each tree in the cluster
    for _, row in cluster_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(map_crime_type)

# Step 10: Save the map to an HTML file
map_crime_type.save('oak_tree_clusters_map.html')