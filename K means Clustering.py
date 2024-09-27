import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess the data
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0

# Step 3: Specify the number of clusters (e.g., 10 for digits 0-9)
num_clusters = 10

# Step 4: Apply KMeans Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(x_train)

# Step 5: Evaluate Clustering
labels = kmeans.labels_

# 1. Silhouette Score
silhouette_avg = silhouette_score(x_train, labels)
print(f'Silhouette Score: {silhouette_avg:.4f}')

# 2. Calinski-Harabasz Index
calinski_harabasz = calinski_harabasz_score(x_train, labels)
print(f'Calinski-Harabasz Index: {calinski_harabasz:.4f}')

# 3. Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(x_train, labels)
print(f'Davies-Bouldin Index: {davies_bouldin:.4f}')

# 4. Adjusted Rand Index (using true labels)
ari = adjusted_rand_score(y_train, labels)
print(f'Adjusted Rand Index: {ari:.4f}')

# Step 6: Visualize the clustered images
centers = kmeans.cluster_centers_.reshape(num_clusters, 28, 28)

# Plot the cluster centers
plt.figure(figsize=(10, 4))
for i in range(num_clusters):
    plt.subplot(2, 5, i + 1)
    plt.imshow(centers[i], cmap='gray')
    plt.title(f'Cluster {i}')
    plt.axis('off')
plt.tight_layout()
plt.show()
