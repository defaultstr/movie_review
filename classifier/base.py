__author__ = 'defaultstr'

from sklearn.datasets import load_svmlight_files
from os import path
from os import system
import logging
from time import time
from sklearn.metrics import accuracy_score


def classify_test(feature_list=[], classifiers=[], root_path='./'):
    #load data set
    datasets = []
    for name in feature_list:
        logging.log(logging.DEBUG, 'loading data: %s ...' % name)
        filenames = tuple(['./feature/%s_%s' % (name, tag) for tag in ['train.txt', 'test.txt']])
        X_train, y_train, X_test, y_test = load_svmlight_files(filenames)
        datasets.append((name, X_train, y_train, X_test, y_test))

    #make directory to store results
    result_path = path.join(root_path, 'results')
    if path.exists(result_path):
        assert path.isdir(result_path), 'data must be a directory!'
    else:
        system('mkdir ' + result_path)

    for clf in classifiers:
        for feature in datasets:
            clf_name = clf.__class__.__name__
            feature_name, X_train, y_train, X_test, y_test = feature
            combine_name = feature_name+'_'+clf_name
            info = {}

            logging.log(logging.DEBUG, 'classification test: %s ...' % combine_name)

            logging.log(logging.DEBUG, 'training...')
            t0 = time()
            clf.fit(X_train, y_train)
            t1 = time()
            info['training_time'] = t1-t0

            logging.log(logging.DEBUG, 'testing on training...')
            pred_y = clf.predict(X_train)
            training_acc = accuracy_score(y_train, pred_y)
            logging.log(logging.DEBUG, 'error rate on training set: %f' % (1.0 - training_acc))
            info['training_error'] = 1.0 - training_acc
            fout = open(path.join(result_path, combine_name+'_train.txt'), 'w')
            for y in pred_y:
                print >>fout, y
            fout.close()

            logging.log(logging.DEBUG, 'testing...')
            t0 = time()
            pred_y = clf.predict(X_test)
            t1 = time()
            info['test_time'] = t1-t0
            test_acc = accuracy_score(y_test, pred_y)
            logging.log(logging.DEBUG, 'error rate on test set: %f' % (1.0 - test_acc))
            info['test_error'] = 1.0 - test_acc
            fout = open(path.join(result_path, combine_name+'_test.txt'), 'w')
            for y in pred_y:
                print >>fout, y
            fout.close()

            yield combine_name, feature_name, clf_name, info







