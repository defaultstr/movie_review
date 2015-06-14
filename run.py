__author__ = 'defaultstr'

from feature_generator import *
from classifier import *
from ensemble import *
import logging
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s', level=logging.DEBUG)

    #generating feature
    vectorizers = [
                   LSIVectorizer(n_grams=[1], n_topics=100, config_identifier='_100'),
                   LSIVectorizer(n_grams=[1], n_topics=40, config_identifier='_40'),
                   LDAVectorizer(n_grams=[1], n_topics=100, config_identifier='_100'),
                   LDAVectorizer(n_grams=[1], n_topics=40, config_identifier='_40'),
                   TfIdfVectorizer(n_grams=[1], config_identifier=''),
                   TfIdfVectorizer(n_grams=[1, 2], config_identifier='_bigram'),
                   TfIdfVectorizer(n_grams=[1, 2, 3], config_identifier='_trigram'),
                   NaiveBayesVectorizer(n_grams=[1], config_identifier=''),
                   NaiveBayesVectorizer(n_grams=[1, 2], config_identifier='_bigram'),
                   NaiveBayesVectorizer(n_grams=[1, 2, 3], config_identifier='_trigram'),
                   Doc2VecVectorizer(min_count=1, dimension=40, config_identifier='_40'),
                   Doc2VecVectorizer(min_count=1, dimension=100, config_identifier='_100'),
                   Word2VecVectorizer(min_count=1, dimension=40, config_identifier='_40'),
                   Word2VecVectorizer(min_count=1, dimension=100, config_identifier='_100')
                  ]
    vectorizers = [
                   Word2VecVectorizer(min_count=1, dimension=40, config_identifier='_40'),
                   Word2VecVectorizer(min_count=1, dimension=100, config_identifier='_100')
                  ]
    g = extract_feature('./data/labeledTrainData.tsv',
                        './data/labeledTestData.tsv',
                        './data/unlabeledTrainData.tsv',
                        vectorizers=vectorizers)
    with open('./feature_list.txt', 'a') as fout:
        for name, info in g:
            print >>fout, '%s\t%s' % (name, json.dumps(info))
            fout.flush()

    #classify
    classifiers = [
                   XGBoost(),
                   LogisticRegression(penalty='l1', solver='liblinear'),
                   LinearSVC(),
                   RandomForestClassifier(n_estimators=30)
                  ]
    feature_list = []
    with open('feature_list.txt') as fin:
        for line in fin:
            feature_list.append(line.split('\t')[0])
    c = classify_test(feature_list=feature_list, classifiers=classifiers)

    with open('./result_list.txt', 'w') as fout:
        for combine_name, feature_name, clf_name, info in c:
            print >>fout, '%s\t%s\t%s\t%f\t%s' % (combine_name,
                                                  feature_name,
                                                  clf_name,
                                                  info['test_error'],
                                                  json.dumps(info))
            fout.flush()

    #ensemble
    base_classifiers = []
    with open('result_list.txt', 'r') as fin:
        for line in fin:
            base_classifiers.append(line.split('\t')[0])

    base_classifiers = [x for x in base_classifiers if x.find('SVC') != -1]
    e = accuracy_ensemble_test('./data/labeledTestData.tsv', base_classifiers=base_classifiers)


    e = stacking_ensemble_test('./data/labeledTrainData.tsv', './data/labeledTestData.tsv', base_classifiers=base_classifiers)





