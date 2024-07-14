# Sohan Aiyappa's Level 2 Report 

## Task 1 : Decision Tree Based ID3 Algorithm

The ID3 algorithm is a decision tree algorithm that helps classify data by selecting the attribute that provides the highest information gain at each step. This means it splits the dataset based on the attribute that best separates the different classes, aiming to reduce uncertainty or entropy.

For this task, I built and evaluated a decision tree model using the ID3 algorithm. First, I developed functions to measure the uncertainty in the data and to determine the best attributes for splitting based on their information gain. I then created a class to encapsulate the ID3 algorithm, including methods for training the model, making predictions, and evaluating its performance.

Using the Iris dataset, which includes measurements of iris flowers to classify them into different species, I trained and tested the model. The model construction involved recursively selecting the most informative attributes and splitting the data accordingly. After training, the model achieved an accuracy of 100% on the test dataset. <br><br>
[Decision Tree-ID3](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/Decision%20Tree%20ID3.py) <br><br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/decision%20tree%20ID3.png?raw=true)

## Task 2 : Naive Bayesian Classifier 


For this task, I built and evaluated a Naive Bayes classifier to predict whether to play golf based on weather conditions. The dataset included features like Outlook, Temperature, Humidity, and Wind, with the target variable being PlayGolf.

Bayes' Theorem calculates the probability of a class based on prior knowledge of conditions related to the class. The Naive Bayes classifier assumes that the features are independent of each other.

![bayes formula](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQosTYuHJVG7RTyRqW9-9me38S5fhIezc7TXQ&s)

I encoded the categorical data into numerical values and separated the features  from the target variable . I then implemented functions to calculate prior probabilities of each class, the likelihood of feature values given the class, and posterior probabilities for predictions. Using these, I built the classifier and applied it to the test data.

After evaluating the model, I  achieved an accuracy of 50%.I am a bit disappointed with this result , I seek to improve it"s accuracy while working on the hyperparameter tuning task. <br><br>
[Naive Bayesian Classifier](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/naive-bayes%20ML.py)<br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/naive%20bayes.png?raw=true)

## Task 3 : Random Forest , GBM and Xgboost

For this task, I built and evaluated a Random Forest classifier using the Iris dataset. The dataset includes measurements of iris flowers, categorized into three species.

First, I loaded the Iris dataset and split it into training and test sets, with 70% of the data used for training and 30% for testing. I used the RandomForestClassifier from scikit-learn, setting the number of trees (estimators) to 100.

After training the model on the training set, I used it to predict the species of the flowers in the test set. The model's accuracy was evaluated using the accuracy_score metric, achieving an accuracy of approximately 97%.

This task demonstrated the effectiveness of the Random Forest algorithm in classification tasks, leveraging the ensemble of multiple decision trees to make robust and accurate predictions.
<br><br>
[Random Forest](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/Random%20Forest.py)
<br><br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/random%20forest.png?raw=true)

## Task 4 : Ensemble Techniques

I utilized a VotingClassifier from scikit-learn to predict survival on the Titanic dataset. After preprocessing to handle missing values and encode categorical variables, I split the data into 80% training and 20% test sets. The ensemble model combined predictions from RandomForestClassifier and LogisticRegression, achieving an accuracy of approximately 81% on the test set. This task highlighted the effectiveness of ensemble techniques in improving predictive performance and robustness in classification tasks. This has also been my favourite task so far , as this opened new doors by utlising multiple models to increase predictability by a significant degree. 
<br><br>
[Ensemble Model](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/Ensemble%20Techniques.py)
<br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/ensemble%20technique.png?raw=true)










