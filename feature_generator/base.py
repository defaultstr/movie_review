__author__ = 'defaultstr'

from sklearn.datasets import dump_svmlight_file
from os import system
from os import path
from time import time
import logging
import pandas as pd
import json


def clean_review(df):
    cleaned_review = []
    for idx, row in df.iterrows():
        #TODO clean review text
        cleaned_review.append(row.review)
    return cleaned_review


def extract_feature(train_file, test_file, unlabeled_file, vectorizers=[], root_path='./'):
    train = pd.read_csv(train_file, header=0, delimiter='\t', quoting=3)
    test = pd.read_csv(test_file, header=0, delimiter='\t', quoting=3)
    unlabeled = pd.read_csv(unlabeled_file, header=0, delimiter='\t', quoting=3)

    y_train = map(int, list(train['sentiment']))
    y_test = map(int, list(test['sentiment']))
    review_train = clean_review(train)
    review_test = clean_review(test)
    review_unlabeled = clean_review(unlabeled)

    #make directory to store features
    feature_path = path.join(root_path, 'feature')
    if path.exists(feature_path):
        assert path.isdir(feature_path), 'data must be a directory!'
    else:
        system('mkdir ' + feature_path)

    for vectorizer in vectorizers:
        info = {}
        name = vectorizer.__class__.__name__ + vectorizer.config_identifier
        logging.log(logging.DEBUG, 'Generating %s ...' % name)

        logging.log(logging.DEBUG, 'Initializing...')
        t0 = time()
        vectorizer.initialize(review_train,
                              review_test,
                              review_unlabeled)
        t1 = time()
        info['initial_time'] = t1-t0

        logging.log(logging.DEBUG, 'Fitting and transforming training set...')
        t0 = time()
        X_train = vectorizer.fit_transform(review_train, y_train)
        t1 = time()
        info['training_time'] = t1-t0
        train_file_path = path.join(feature_path, name+'_train.txt')
        dump_svmlight_file(X_train, y_train, open(train_file_path, 'w'))

        logging.log(logging.DEBUG, 'Transforming test set...')
        t0 = time()
        X_test = vectorizer.transform(review_test, y_test)
        t1 = time()
        info['test_time'] = t1-t0
        test_file_path = path.join(feature_path, name+'_test.txt')
        dump_svmlight_file(X_test, y_test, open(test_file_path, 'w'))

        info['total_time'] = info['initial_time'] + info['training_time'] + info['test_time']
        info['feature_dimension'] = vectorizer.get_dimension()
        logging.log(logging.DEBUG, 'Vectorizer info: ' + json.dumps(info))

        yield name, info



class Vectorizer(object):

    def __init__(self, **kwargs):
        self.config_identifier = ''
        if 'config_identifier' in kwargs:
            self.config_identifier = kwargs['config_identifier']

    def initialize(self, train_X, test_X, unlabeled_X):
        pass

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_dimension(self):
        raise NotImplementedError

    def _tokenize(self, sentence):
        words = sentence.split()
        tokens = []
        for gram in self.n_grams:
            for i in range(len(words) - gram + 1):
                tokens += ["_*_".join(words[i:i+gram])]
        return tokens

