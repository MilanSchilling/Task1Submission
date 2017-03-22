# This routine takes a saved model and the saved descriptors of the test data and performs the prediction

import numpy as np
import time
import csv
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from scipy.stats import randint as sp_randint
np.random.seed(1)

print('Load the descriptors of the test data...')
# load test descriptors 
TestDesc = np.load("TestDesdepth_px22out100perc.npy")


# Optimize the parameters by cross-validation
parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 1, 10, 100]},
    {'kernel': ['linear'], 'C': [0.01, 1, 10, 100]}
]

print('Load the fitted model...')
# Grid search object with SVM classifier.
clf = GridSearchCV(SVC(), parameters, cv=10, n_jobs=8)
# load clf from file
clf.best_estimator_ = joblib.load("params80perc.pkl")
# predict the classes
labels_predicted = clf.predict(TestDesc)


# save the prediction to file
filename_csv = time.strftime("%Y-%m-%d-%H:%M:%S") + ".csv"
with open('./' + filename_csv, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id'] + ['Prediction'])
    for idx, label_predicted in enumerate(labels_predicted):
        writer.writerow([str(idx+1)] + [str(label_predicted)])

print('Prediction done to file: ' + str(filename_csv))