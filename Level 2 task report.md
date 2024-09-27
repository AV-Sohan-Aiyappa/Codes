# Sohan Aiyappa's Level 2 Report 

## Task 1 : Decision Tree Based ID3 Algorithm

The ID3 algorithm is a decision tree algorithm that helps classify data by selecting the attribute that provides the highest information gain at each step. This means it splits the dataset based on the attribute that best separates the different classes, aiming to reduce uncertainty or entropy.
<br>
For this task, I built and evaluated a decision tree model using the ID3 algorithm. First, I developed functions to measure the entropy in the data and to determine the best attributes for splitting based on their information gain. I then created a class to encapsulate the ID3 algorithm, including methods for training the model, making predictions, and evaluating its performance.
<br>
![image](https://i0.wp.com/sefiks.com/wp-content/uploads/2017/11/tree-v3.png?resize=804%2C506&ssl=1)
<br>
Using the PlayTennis dataset, which includes various weather conditions and other factors to classify whether tennis will be played or not, I trained and tested the model. The model construction involved recursively selecting the most informative attributes, such as outlook, temperature, humidity, and wind, and splitting the data accordingly. After training, the model achieved an accuracy of 100% on the test dataset. <br><br>
[Decision Tree-ID3](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/Decision%20Tree%20ID3.py) <br><br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/ID3.jpg?raw=true)
<br>
## Task 2 : Naive Bayesian Classifier 

Naive Bayes classifier is a simple, probabilistic machine learning algorithm based on Bayes' theorem. It assumes that features are independent given the class label, which simplifies calculations. It's commonly used for text classification, spam detection, and sentiment analysis due to its efficiency and effectiveness with large datasets.
<br>
Bayes' Theorem calculates the probability of a class based on prior knowledge of conditions related to the class. The Naive Bayes classifier assumes that the features are independent of each other.
<br>
![bayes formula](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQosTYuHJVG7RTyRqW9-9me38S5fhIezc7TXQ&s)
<br>
For this task, I built and evaluated a Multinomial Naive Bayes classifier to predict whether to play golf based on weather conditions. The dataset included features like Outlook, Temperature, Humidity, and Wind, with the target variable being PlayGolf.<br>
I encoded the categorical data into numerical values and separated the features  from the target variable . I then implemented functions to calculate prior probabilities of each class, the likelihood of feature values given the class, and posterior probabilities for predictions. Using these, I built the classifier and applied it to the test data.
<br>
After evaluating the model, I  achieved an accuracy of 50%.I am a bit disappointed with this result , I seek to improve it"s accuracy while working on the hyperparameter tuning task. <br><br>
[Naive Bayesian Classifier](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/naive-bayes%20ML.py)<br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/naive%20bayes.png?raw=true)
<br>
## Task 3 : Random Forest , GBM and Xgboost
<br>
Random Forest is an ensemble learning method that constructs multiple decision trees during training and merges their outputs for classification or regression tasks. By averaging the predictions of various trees, it reduces overfitting and improves accuracy. It's robust to noise and effective for handling large datasets with diverse features.
<br>

For this task, I built and evaluated a Random Forest classifier using the Iris dataset. The dataset includes measurements of iris flowers, categorized into three species.
First, I loaded the Iris dataset and split it into training and test sets, with 70% of the data used for training and 30% for testing. I used the RandomForestClassifier from scikit-learn, setting the number of trees (estimators) to 100.

After training the model on the training set, I used it to predict the species of the flowers in the test set. The model's accuracy was evaluated using the accuracy  metric, achieving an accuracy of 100%.

This task demonstrated the effectiveness of the Random Forest algorithm in classification tasks, leveraging the ensemble of multiple decision trees to make robust and accurate predictions.
<br><br>
[Random Forest](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/Random%20Forest.py)
<br><br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/random%20forest.png?raw=true)

## XGBoost 
XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting that is specifically designed for speed and performance.<br>
![xg](https://analyticsindiamag.com/wp-content/uploads/2020/11/xgboost.png)
<br>
I built an XGBoost model using the Wine Quality dataset to predict wine quality based on various chemical properties. First, I loaded the dataset and split it into training and testing sets. Then, I used the XGBClassifier from the XGBoost library to create my model. After training it on the data, I made predictions on the test set and calculated the accuracy to see how well it performed.
<br>
![xg](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/xgboost.png?raw=true)
<br>

## GBM 
Gradient Boosting Machine (GBM) is an ensemble learning technique used for regression and classification tasks. It builds models sequentially, where each new model corrects the errors made by the previous ones. <br>
![gbm](https://www.researchgate.net/publication/350326060/figure/fig7/AS:11431281123843737@1677817577632/Flowchart-describing-the-working-of-a-GBM-model.png)<br>

I built a GBM model using the Breast Cancer Wisconsin dataset to classify tumors as malignant or benign. After loading the dataset, I split it into training and testing sets. I used GradientBoostingClassifier from sklearn to create the model, and trained it on the data. Once trained, I made predictions on the test set and calculated the accuracy to see how well the model performed. <br>
![gbm](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/gbm%20pic.png?raw=true)
<br>
## Task 4 : Ensemble Techniques

Ensemble techniques combine multiple models to improve overall performance and robustness. They work on the principle that aggregating predictions from various models can lead to better accuracy than any single model alone. Common methods include bagging (e.g., Random Forest), boosting (e.g., AdaBoost, Gradient Boosting), and stacking. These techniques can reduce variance, bias, or improve predictions by leveraging the strengths of different algorithms.<br>
![ensemble](https://www.jcchouinard.com/wp-content/uploads/2021/11/image-10.png)<br>
I utilized a VotingClassifier from scikit-learn to predict survival on the Titanic dataset. After preprocessing to handle missing values and encode categorical variables, I split the data into 80% training and 20% test sets. The ensemble model combined predictions from RandomForestClassifier and LogisticRegression, achieving an accuracy of approximately 81% on the test set. This task highlighted the effectiveness of ensemble techniques in improving predictive performance and robustness in classification tasks. This has also been my favourite task so far , as this opened new doors by utlising multiple models to increase predictability by a significant degree. 
<br><br>
[Ensemble Model](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/Ensemble%20Techniques.py)
<br>
![image](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/ensemble.png?raw=true)
<br>


## Task 5 : Hyperparameter Tuning 

Hyperparameter tuning is the process of optimizing the parameters that govern the training of a machine learning model, known as hyperparameters. Unlike model parameters, which are learned during training, hyperparameters are set before the training process begins and can significantly influence the model's performance.<br>
![hyp](https://community.arm.com/resized-image/__size/1265x0/__key/communityserver-blogs-components-weblogfiles/00-00-00-37-98/Figure_5F00_2-hires.png)<br>

In my project, I focused on hyperparameter tuning for a Naive Bayes classifier designed to predict whether to play golf based on weather conditions. After preparing the dataset and encoding categorical variables, I implemented functions to calculate prior probabilities and likelihoods with Laplace smoothing. I utilized K-Fold cross-validation to evaluate the model's performance across different values of the smoothing parameter, exploring a range of alpha values (0.1 to 2)<br>
[Hyperparameter Tuning](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/hyp.py)<br>
![hype](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/hyperparamter.png?raw=true)
<br>

## Task 6 : K Means Clustering 
K-means clustering is an unsupervised learning algorithm used to partition a dataset into k distinct clusters based on feature similarity. The algorithm assigns data points to the nearest cluster centroid, iteratively updating the centroids until convergence. This method is useful for identifying patterns and groupings in data.<br>
![k](https://serokell.io/files/q4/q49pm3tx.K-Means_Clustering_Algorithm_pic1_(1).png)<br>

In my project, I implemented K-Means clustering on the MNIST dataset to group handwritten digits into clusters. After loading and preprocessing the data, I reshaped the images into a flat format and normalized the pixel values. I specified the number of clusters as 10, corresponding to the digits 0-9, and applied the KMeans algorithm to fit the data. I evaluated the clustering performance using several metrics. Then , I visualized the cluster centers by plotting the average images of each cluster. 
<br>
[k means clustering](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/K%20means%20Clustering.py)<br>
![k](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/K%20means%20clustering.png?raw=true)
<br>

## Task 7 : Anomaly Detection 
Anomaly detection involves identifying data points that deviate significantly from the expected pattern in a dataset. It is crucial for applications like fraud detection, network security, and fault detection, as it helps in spotting unusual behavior or outliers that could indicate errors, fraud, or other significant events.
<br>
I used Support Vector Machines (SVM) for anomaly detection in my project, employing a one-class SVM approach. <br>
![an](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/06/25729anomaly-2.10.50-PM.png)
<br>
In this task , I created a system to detect unusual data points using a One-Class Support Vector Machine (SVM). I began by generating a dataset with a main cluster of points and added some random outliers to represent anomalies. After training the model to recognize normal data, I used it to identify which points were considered anomalies. Lastly , I visualized the results, showing normal data in blue and the detected anomalies in purple.<br>
[anomaly detection](https://github.com/AV-Sohan-Aiyappa/Codes/blob/main/anomaly%20detection.py)<br>
![k means](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/anomaly%20detection.png?raw=true)
<br>


## Task 10 : Table Analysis using PaddleOCR
PaddleOCR is an open-source Optical Character Recognition (OCR) toolkit developed by PaddlePaddle. It provides a comprehensive solution for text detection and recognition in images, supporting multiple languages and text layouts. PaddleOCR includes features like text detection, character recognition, and layout analysis, making it suitable for a wide range of applications, including document digitization, data extraction from images, and scene text recognition.<br> 
![paddle](https://th.bing.com/th/id/OIP.QUMDlQ0OCy4fYlHvrFYivAHaEK?rs=1&pid=ImgDetMain)<br>
I developed a pipeline using PaddleOCR to extract and analyze tabular data from images. The process involved preprocessing images to enhance table visibility, utilizing PaddleOCR for text detection and extraction, and structuring the data into a Pandas DataFrame. Finally, I performed statistical analysis and created visualizations with Pandas and Matplotlib, demonstrating an effective approach to automate the extraction and analysis of tabular data from images.
<br>
![paddle](https://github.com/AV-Sohan-Aiyappa/Pictures/blob/main/paddleocr.png?raw=true)
<br>


> ## This level was comparitively hard , but there was so much that I learnt. Thank You! 





