#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

#clf = SVC(C=1.0, kernel='linear', gamma='auto')
clf = SVC(C=10000.0, kernel='rbf', gamma='auto')
print clf

##subset the data to train faster
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0=time()
clf.fit(features_train, labels_train)
t1=time()
time_elapsed = round((t1-t0),4)
print "Training time = ", time_elapsed, "s"

#prediction
tp0=time()
pred = clf.predict(features_test)
tp1=time()
print "Prediction time = ", round((tp1-tp0),4), "s"

#accuracy score
from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test,pred)
print "Model accuracy = ",acc

## Get predictions for specific elements
p = pred[10]
print "Prediction for element 10 = ", p

p = pred[26]
print "Prediction for element 26 = ", p

p = pred[50]
print "Prediction for element 50 = ", p

chris = pred.sum()
print "Number of class predicted for chris is", chris
#########################################################


