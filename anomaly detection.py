import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

# Step 1: Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)

# Add some anomalies
np.random.seed(42)
n_anomalies = 20
anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, 2))
X = np.vstack((X, anomalies))

# Step 2: Train One-Class SVM
model = OneClassSVM(gamma='auto', nu=0.1)  # nu is the upper bound on the fraction of outliers
model.fit(X)

# Predict anomalies
y_pred = model.predict(X)
anomalies_mask = y_pred == -1

# Step 3: Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[~anomalies_mask, 0], X[~anomalies_mask, 1], color='blue', label='Normal Data', alpha=0.5)
plt.scatter(X[anomalies_mask, 0], X[anomalies_mask, 1], color='purple', label='Anomalies', alpha=0.5)
plt.title('Anomaly Detection using One-Class SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()
