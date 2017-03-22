# This routine takes the outputfile from 'write_desc_to_file.py' and performs a SVM gridsearch. The best estimator gets wrote to file.

# Intro
import numpy as np
import pickle
import cv2
import time
from numpy import array
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, decomposition, manifold
from sklearn.model_selection import KFold, PredefinedSplit, ShuffleSplit
from sklearn.externals import joblib
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
from skimage import feature
np.random.seed(1)


def run (arr_train_des, arr_train_lab):
        print('start new run')

        # split train data
        X_train, X_test, labels_train, labels_test = train_test_split(arr_train_des, arr_train_lab,test_size=0.2, random_state=1)

        # SVM
        print('running SVM', file=f)
        start2 = time.time()

        # Optimize the parameters by cross-validation
        parameters = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 1, 10, 100]},
            {'kernel': ['linear'], 'C': [0.01, 1, 10, 100]}
        ]

        # Grid search object with SVM classifier.
        clf = GridSearchCV(SVC(), parameters, verbose=20, cv=10, n_jobs=4)
        #clf.fit(arr_train_des, arr_train_lab)
        clf.fit(X_train, labels_train)
        # save model parameters
        joblib.dump(clf.best_estimator_, 'params0_5perc.pkl', compress = 1)

        print("Best parameters set found on training set:", file=f)
        print(clf.best_params_, file=f)
        print()

        means_valid = clf.cv_results_['mean_test_score']
        stds_valid = clf.cv_results_['std_test_score']
        means_train = clf.cv_results_['mean_train_score']

        print("Grid scores:", file=f)
        for mean_valid, std_valid, mean_train, params in zip(means_valid, stds_valid, means_train, clf.cv_results_['params']):
            print("Validation: %0.3f (+/-%0.03f), Training: %0.3f  for %r" % (mean_valid, std_valid, mean_train, params), file=f)
        print()

        #labels_test, labels_predicted = arr_test_lab, clf.predict(arr_test_des)
        labels_test, labels_predicted = labels_test, clf.predict(X_test)
        print("Test Accuracy [%0.3f]" % ((labels_predicted == labels_test).mean()), file=f)

        end2 = time.time()
        print("SVM needed: " + str(round(end2-start2,3)) + " seconds.", file=f)
        print("SVM needed: " + str(round(end2-start2,3)) + " seconds.")



#load from file

#descriptors
desdepth_px22out80perc = np.load("desdepth_px22out80perc.npy")

#labels
arr_train_lab = np.load("labOut80perc.npy")

f = open('output_0_5perc_4core','w')


run(arr_train_des = desdepth_px22out80perc, arr_train_lab = arr_train_lab)

f.close()

print('finished')
