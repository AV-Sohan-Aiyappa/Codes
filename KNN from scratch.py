import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define Euclidean distance function
def euclidean_distance(p1, p2):
  """Calculates the Euclidean distance between two points."""
  return np.sqrt(np.sum((p1 - p2) ** 2))

# KNN function for prediction
def predict_knn(X_train, y_train, X_test, k):
  """Predicts class labels for test data using KNN."""
  predictions = []
  for test_instance in X_test:
    distances = [euclidean_distance(test_instance, train_instance) for train_instance in X_train]
    nearest_neighbors = np.argsort(distances)[:k]
    neighbor_labels = y_train[nearest_neighbors]
    predicted_label = np.argmax(np.bincount(neighbor_labels))
    predictions.append(predicted_label)
  return predictions

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN prediction from scratch
k = 3  # Number of neighbors
y_pred_knn = predict_knn(X_train, y_train, X_test, k)

# KNN prediction using scikit-learn
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=k)
knn_clf.fit(X_train, y_train)
y_pred_sklearn = knn_clf.predict(X_test)

# Evaluate accuracy for both methods
knn_accuracy = accuracy_score(y_test, y_pred_knn)
sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)

print(f"KNN (from scratch) Accuracy (k={k}):", knn_accuracy)
print(f"scikit-learn KNN Accuracy (k={k}):", sklearn_accuracy)


