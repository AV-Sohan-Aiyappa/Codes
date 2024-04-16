from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def sigmoid(z):
  """Sigmoid function to squash values between 0 and 1"""
  return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate, num_iterations):
  """Logistic regression model with gradient descent"""
  # Initialize weights with zeros (adjust as needed)
  weights = np.zeros(X.shape[1])
  bias = 0

  # Training loop using gradient descent
  for _ in range(num_iterations):
    # Calculate predicted probabilities
    predicted_prob = sigmoid(np.dot(X, weights) + bias)

    # Calculate error
    error = predicted_prob - y

    # Update weights and bias using gradients
    weights -= learning_rate * np.dot(X.T, error)
    bias -= learning_rate * np.mean(error)

  return weights, bias

# Load the Iris dataset
iris = load_iris()

# Separate features and target variable
X = iris.data
y = iris.target  # Assuming binary classification (0 or 1)

# Hyperparameters (adjust as needed)
learning_rate = 0.01
num_iterations = 1000

# Train the model
weights, bias = logistic_regression(X, y, learning_rate, num_iterations)

# Make predictions on a new data point (replace with your new data)
new_data = np.array([6, 3, 5, 2])  # Example data point with 4 features (sepal length, sepal width, petal length, petal width)

# Reshape new_data if it's a single data point
if len(new_data.shape) == 1:
  new_data = new_data.reshape(1, -1)

predicted_prob = sigmoid(np.dot(new_data, weights) + bias)

print("Predicted probability:", predicted_prob)

# Apply a threshold (e.g., 0.5) to classify (0 for lower, 1 for higher)
if predicted_prob > 0.5:
  print("Predicted class:", iris.target_names[1])  # Assuming class 1 for probability > 0.5
else:
  print("Predicted class:", iris.target_names[0])  # Assuming class 0 for probability <= 0.5

