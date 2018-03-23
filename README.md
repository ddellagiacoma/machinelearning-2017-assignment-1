# Cross Validation

## 1. INTRODUCTION

The goal of this assignment is to experiment with classification algorithms, test the classifier using cross-validation over the training set through **Scikit-learn**, a wide-spread library for Machine Learning in use nowadays. Then, train the classifier over the full training set and, finally, use the classifier to predict the examples in the test set.

This report summarizes the methodology used and the results obtained.

## 2. DATA

The data used to train and test the predictor refers to a spam email classification. The dataset is already split into training (3.681 instances) and test (920 instances) sets. Moreover, the labels of the training examples are known but the labels of the test set are hidden.

The **train-data.csv** and the **test-data.csv** contain a series of email attributes such as the frequency of specific words or characters, and the use of capital letters in the e-mail. Finally, the **traintargets.csv** file contains the labels of the training set, i.e., whether the e-mail was considered spam (1) or not (0)

The classifier has to classify the examples in the test set with higher accuracy than the reference
baseline which is 0.63152.

## 3. LEARNING

The classification algorithm selected for the task is the Support Vector Machine (SVM). The implementation of this algorithm includes a list of parameters such as kernel, gamma and C which have higher impact on model performance. The kernel used for the algorithm is the **RBF** which is especially useful when the data-points are not linearly separable. The **RBF** kernel requires the gamma parameter which is a value that must be higher than 0 and defines how much influence a single training example has. On the other hand, the parameter **C**, which is not exclusive to **RBF** kernel, defines off between smooth decision boundary and classifying the training points correctly.

While the parameter **C** has been decided and set from the beginning to 10, the parameter **gamma** has been obtained from a cross-validation on the training set to ensure better performance to the classifier.

## 4. RESULTS

These are the results of the cross-validation using different gammas. Mean accuracy, precision, recall and F<sub>1</sub> have been noted for each different **gamma** parameter.

**gamma ->** | **0.05** | **0.01** | **0.005** | **0.001** | **0.0005**
--- | --- | --- | --- | --- | ---
**Mean accuracy** | 0.8171719220 | 0.8611796501 | 0.8679753406 | 0.8747655005 | 0.8723213232
**Mean precision** | 0.7484036796 | 0.8151098420 | 0.8309653504 | 0.8554571509 | 0.8563057956
**Mean recall** | 0.8182025906 | 0.8455184306 | 0.8421873704 | 0.8279003748 | 0.8186110608
**Mean F<sub>1</sub>** | 0.7816926146 | 0.8299486421 | 0.8364114180 | 0.8412104365 | 0.8369335319

The cross-validation highlighted that the best **gamma** value for this classifier to achieve an higher accuracy is 0.001.

The following plot shows the training score and the cross-validation score of the classifier.

![image](https://user-images.githubusercontent.com/24565161/37827220-e274f338-2e96-11e8-829c-3c96cb0f58ac.png)

Finally, the classifier has been trained over the full training set and has been used to predict the examples in the **test-data.csv** training set. However, the accuracy of the prediction cannot be tested because the labels of the test set are hidden. Nevertheless, the results of the cross-validation showed an higher accuracy than the reference baseline.
