__author__ = 'defaultstr'

from feature_generator import *
import logging
import json


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s', level=logging.DEBUG)
    vectorizers = [LSIVectorizer(n_grams=[1], n_topics=100, config_identifier='_100'),
                   LSIVectorizer(n_grams=[1], n_topics=40, config_identifier='_40'),
                   LDAVectorizer(n_grams=[1], n_topics=100, config_identifier='_100'),
                   LDAVectorizer(n_grams=[1], n_topics=40, config_identifier='_40'),
                   TfIdfVectorizer(n_grams=[1], config_identifier=''),
                   TfIdfVectorizer(n_grams=[1, 2], config_identifier='_bigram'),
                   TfIdfVectorizer(n_grams=[1, 2, 3], config_identifier='_trigram'),
                   NaiveBayesVectorizer(n_grams=[1], config_identifier=''),
                   NaiveBayesVectorizer(n_grams=[1, 2], config_identifier='_bigram'),
                   NaiveBayesVectorizer(n_grams=[1, 2, 3], config_identifier='_trigram')]
    vectorizers = [Word2VecVectorizer()]
    names, infos = extract_feature('./data/labeledTrainData.tsv',
                                   './data/labeledTestData.tsv',
                                   vectorizers=vectorizers)
    with open('./feature_list.txt', 'w') as fout:
        for name, info in zip(names, infos):
            print >>fout, '%s\t%s' % (name, json.dumps(info))

