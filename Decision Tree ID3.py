import pandas as pd
import numpy as np
from collections import Counter

def entropy(y):
    counter = Counter(y)
    probabilities = [count / len(y) for count in counter.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def information_gain(X, y, attribute):
    values = set(X[attribute])
    weighted_entropy = sum(
        (len(X[X[attribute] == value]) / len(X)) * entropy(y[X[attribute] == value])
        for value in values
    )
    return entropy(y) - weighted_entropy

class DecisionTreeID3:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return next(iter(set(y)))
        
        if len(X) < self.min_samples_split or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]
        
        best_attribute = max(X.columns, key=lambda attr: information_gain(X, y, attr))
        tree = {best_attribute: {}}
        
        for value in set(X[best_attribute]):
            subset_X = X[X[best_attribute] == value]
            subset_y = y[X[best_attribute] == value]
            subtree = self._build_tree(subset_X.drop(columns=[best_attribute]), subset_y, depth + 1)
            tree[best_attribute][value] = subtree
        
        return tree
    
    def predict(self, X):
        return X.apply(self._predict_row, axis=1, tree=self.tree)
    
    def _predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        attribute = next(iter(tree))
        value = row[attribute]
        subtree = tree[attribute].get(value)
        if subtree is None:
            return Counter(tree[attribute]).most_common(1)[0][0]
        return self._predict_row(row, subtree)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)
X = df.drop(columns=['PlayTennis'])
y = df['PlayTennis']

model = DecisionTreeID3(max_depth=3)
model.fit(X, y)

# Calculate accuracy on the training data
accuracy = model.accuracy(X, y)
print(f"Accuracy on training data: {accuracy:.2f}")

