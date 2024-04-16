from sklearn.datasets import load_iris, fetch_openml  # Load datasets
from sklearn.model_selection import train_test_split  # Split data
from sklearn.neighbors import KNeighborsClassifier  # KNN algorithm
from sklearn.preprocessing import StandardScaler  # Feature scaling (optional)
from sklearn.metrics import accuracy_score  # Evaluate performance

def iris_knn():
  # Load the Iris dataset
  iris = load_iris()
  X = iris.data  # Features
  y = iris.target  # Target labels (species)

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create the KNN classifier with k=5 neighbors
  knn = KNeighborsClassifier(n_neighbors=5)

  # Train the KNN model on the training data
  knn.fit(X_train, y_train)

  # Make predictions on the testing set
  y_pred = knn.predict(X_test)

  # Evaluate model performance (accuracy)
  accuracy = accuracy_score(y_test, y_pred)
  print("Iris KNN Accuracy:", accuracy)

iris_knn()
