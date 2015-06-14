__author__ = 'defaultstr'

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class Ensemble(object):

    def initialize(self, valid=None):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


def accuracy_ensemble_test(test_file, base_classifiers=[]):
    #load training data
    X_test = []
    for c in base_classifiers:
        X = [float(line) for line in open('./results/%s_test.txt' % c)]
        X_test.append(X)
    X_test = np.array(zip(*X_test))

    test = pd.read_csv(test_file, header=0, delimiter='\t', quoting=3)
    y_test = np.array(map(int, list(test['sentiment'])))

    #seperate validation set
    X_valid = np.empty((10000, len(X_test[0])))
    X_valid[:5000, :] = X_test[:5000, :]
    X_valid[-5000:, :] = X_test[-5000:, :]
    X_test = X_test[5000:-5000, :]

    y_valid = np.empty((10000, ))
    y_valid[:5000] = y_test[:5000]
    y_valid[-5000:] = y_test[-5000:]
    y_test = y_test[5000:-5000]

    acc = np.array([accuracy_score(y_valid, X_valid[:, i]) for i in range(len(X_valid[0]))])
    weight = acc / np.sum(acc)

    pred_y = [1 if x > 0.5 else 0 for x in np.dot(X_test, weight)]
    print 'accuracy weighting!'
    print 'accruacy: %f' % accuracy_score(y_test, pred_y)
    print 'error rate: %f' % (1.0 - accuracy_score(y_test, pred_y))


def stacking_ensemble_test(train_file, test_file, base_classifiers=[]):
    #load training data
    X_train = []
    X_test = []
    for c in base_classifiers:
        X = [float(line) for line in open('./results/%s_train.txt' % c)]
        X_train.append(X)
        X = [float(line) for line in open('./results/%s_test.txt' % c)]
        X_test.append(X)
    X_train = np.array(zip(*X_train))
    X_test = np.array(zip(*X_test))

    train = pd.read_csv(train_file, header=0, delimiter='\t', quoting=3)
    test = pd.read_csv(test_file, header=0, delimiter='\t', quoting=3)
    y_train = np.array(map(int, list(train['sentiment'])))
    y_test = np.array(map(int, list(test['sentiment'])))

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    print 'stacking weighting!'
    print 'accuracy on training: %f' % accuracy_score(y_train, clf.predict(X_train))
    print 'accruacy: %f' % accuracy_score(y_test, clf.predict(X_test))
    print 'error rate: %f' % (1.0 - accuracy_score(y_test, clf.predict(X_test)))





