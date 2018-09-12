#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your clfeatures_testassifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

## import learners
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


## Random Forest
clf_RF = RandomForestClassifier(n_estimators=15, criterion='gini', max_depth=None, min_samples_split=3)

## ADABoost Classifier
clf_ADA = AdaBoostClassifier(n_estimators=50)

## K-Nearest Neighbor Classifier
clf_KNN = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, metric='minkowski')

## Train Classifiers
clf_RF = clf_RF.fit(features_train,labels_train)
clf_ADA = clf_ADA.fit(features_train,labels_train)
clf_KNN = clf_KNN.fit(features_train,labels_train)

## Predicitons
pred_RF = clf_RF.predict(features_test)
pred_ADA = clf_ADA.predict(features_test)
pred_KNN = clf_KNN.predict(features_test)

## Compute Accuracy on test data
from sklearn.metrics import accuracy_score
acc_RF = accuracy_score(labels_test,pred_RF)
acc_ADA = accuracy_score(labels_test,pred_ADA)
acc_KNN = accuracy_score(labels_test,pred_KNN)

print 'accuracy of the Random Forest model = ', acc_RF
print 'accuracy of the ADABoost model = ', acc_ADA
print 'accuracy of the KNN model = ', acc_KNN


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
