import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Data preparation
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayGolf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

le = defaultdict(LabelEncoder)
df = df.apply(lambda col: le[col.name].fit_transform(col))

X = df.drop('PlayGolf', axis=1)
y = df['PlayGolf']

# Define the Naive Bayes functions with smoothing parameter
def calculate_prior(y_train):
    classes = np.unique(y_train)
    prior = {c: np.mean(y_train == c) for c in classes}
    return prior

def calculate_likelihood(X_train, y_train, alpha):
    likelihood = {}
    n_features = X_train.shape[1]
    classes = np.unique(y_train)
    
    for c in classes:
        X_c = X_train[y_train == c]
        likelihood[c] = {}
        for feature in X_train.columns:
            feature_values = np.unique(X_train[feature])
            likelihood[c][feature] = {
                val: (np.sum(X_c[feature] == val) + alpha) / (X_c.shape[0] + alpha * len(feature_values))
                for val in feature_values
            }
    
    return likelihood

def calculate_posterior(X, prior, likelihood):
    posteriors = []
    
    for _, x in X.iterrows():
        posterior = prior.copy()
        for c in posterior:
            for feature, value in x.items():
                posterior[c] *= likelihood[c][feature].get(value, 1 / (len(X) + len(likelihood[c][feature])))
        posteriors.append(posterior)
    
    return posteriors

def predict(X, prior, likelihood):
    posteriors = calculate_posterior(X, prior, likelihood)
    return [max(p, key=p.get) for p in posteriors]

def cross_val_accuracy(X, y, k=5, alpha=1.0):
    accuracies = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        prior = calculate_prior(y_train)
        likelihood = calculate_likelihood(X_train, y_train, alpha)
        y_pred = predict(X_test, prior, likelihood)
        
        accuracies.append(np.mean(y_pred == y_test))
    
    return np.mean(accuracies)

# Hyperparameter tuning
alphas = [0.1, 0.5, 1, 1.5, 2]  # Different values for Laplace smoothing
best_alpha = None
best_accuracy = 0

for alpha in alphas:
    accuracy = cross_val_accuracy(X, y, k=5, alpha=alpha)
    print(f"Alpha: {alpha}, Cross-validated Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha

print(f"Best Alpha: {best_alpha}, Best Accuracy: {best_accuracy * 100:.2f}%")
