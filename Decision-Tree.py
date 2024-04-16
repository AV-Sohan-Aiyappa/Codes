from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# Load the Iris dataset
iris = load_iris()

# Separate features (X) and target variable (y)
X = iris.data
y = iris.target

# Create a decision tree classifier object
clf = DecisionTreeClassifier()

# Train the model on the data
clf.fit(X, y)

# Make a prediction on a new flower with specific features
# (Replace these values with actual measurements)
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(new_flower)

# Print the predicted Iris species based on the index in iris.target_names
print("Predicted Iris species:", iris.target_names[prediction[0]])




