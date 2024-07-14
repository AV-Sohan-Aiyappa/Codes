import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = pd.read_csv(url)


def preprocess_data(data):
   
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    
    
    if 'Embarked' in data.columns:
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    
    
    if 'Embarked' in data.columns:
        data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

    
    data.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')
    return data

data = preprocess_data(data)


X = data.drop(columns=['Survived'])
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
logistic_regression = LogisticRegression(max_iter=200, random_state=42)


random_forest.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)


voting_clf = VotingClassifier(estimators=[
    ('rf', random_forest),
    ('lr', logistic_regression)
], voting='soft')


voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the ensemble model: {accuracy:.2f}")
