# Amazon-Food-Reviews-Analysis-and-Modelling


#### Performed Exploratory Data Analysis, Data Cleaning, Data Visualization and Text Featurization(BOW, tfidf, Word2Vec). Build several ML models like KNN, Naive Bayes, Logistic Regression, SVM, Random Forest, etc.

#### Objective:
Given a text review, determine the sentiment of the review whether its positive or negative.

Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

#### About Dataset

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.<br>

Number of reviews: 568,454<br>
Number of users: 256,059<br>
Number of products: 74,258<br>
Timespan: Oct 1999 - Oct 2012<br>
Number of Attributes/Columns in data: 10 

Attribute Information:

1. Id
2. ProductId - unique identifier for the product
3. UserId - unqiue identifier for the user
4. ProfileName
5. HelpfulnessNumerator - number of users who found the review helpful
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
7. Score - rating between 1 and 5
8. Time - timestamp for the review
9. Summary - brief summary of the review
10. Text - text of the review
<hr>
#### 1 Amazon Food Reviews EDA, NLP and Text Preprocessing
1. Defined Problem Statement  
2. Performed Exploratory Data Analysis(EDA) on Amazon Fine Food Reviews Dataset plotted Word Clouds, Distplots, Histograms, etc.
3. Performed Data Cleaning & Data Preprocessing by removing unneccesary and duplicates rows and for text reviews removed html tags, punctuations, Stopwords and Stemmed the words using Porter Stemmer 
4. Documented the concepts clearly
5. Plotted TSNE plots for Different Featurization of Data viz. BOW(uni-gram,bi-gram), tfidf, Avg-Word2Vec(using Word2Vec model pretrained on Google News) and tf-idf-Word2Vec

#### 2 KNN
1. Applied K-Nearest Neighbour on Different Featurization of Data viz. BOW(uni-gram,bi-gram), tfidf, Avg-Word2Vec(using Word2Vec model pretrained on Google News) and tf-idf-Word2Vec 
2. Used both brute & kd-tree implementation of KNN 
3. Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne

#### 3 Naive Bayes

#### 4 Logistic Regression
1. Applied Logistic Regression on Different Featurization of Data viz. BOW(uni-gram,bi-gram), tfidf, Avg-Word2Vec(using Word2Vec model pretrained on Google News) and tf-idf-Word2Vec 
2. Used both Grid Search & Randomized Search Cross Validation
3. Evaluated the test data on various performance metrics like accuracy, f1-score, precision, recall,etc. also plotted Confusion matrix using seaborne
4. Showed How Sparsity increases as we increase lambda or decrease C when L1 Regularizer is used for each featurization<br>
5. Did pertubation test to check whether the features are multi-collinear or not

###### Performance Table
<img src="https://image.ibb.co/d8Ugbo/2_KNN_table.png" />

###### Conclusions:
1. Best Accuracy of 85.107% is achieved by Avg Word2Vec Featurization
2. The kd-tree and brute implementation of KNN gives relatively similar results
3. KNN is a very slow Algorithm compared to others takes alot of time to train
4. KNN did not fair in terms of precision and F1-score. Overall KNN was not that good for this dataset
<hr>
#### 5 SVM




