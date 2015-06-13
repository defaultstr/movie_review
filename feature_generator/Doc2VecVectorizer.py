__author__ = 'defaultstr'

from .base import Vectorizer
from os import path
from os import system
import re
import numpy as np


class Doc2VecVectorizer(Vectorizer):
    def __init__(self,
                 n_grams=[1], #only use word
                 dimension=300, config_identifier='_300',
                 sample=1e-3,
                 window=10,
                 min_count=40,
                 negative=5,
                 hs=1,
                 ):
        super(Doc2VecVectorizer, self).__init__(config_identifier=config_identifier)
        self.dimension = dimension
        self.sample = sample
        self.window = window
        self.min_count = min_count
        self.n_grams = n_grams
        self.negative = negative
        self.hs = hs
        self.doc2id = None
        self.id2vec = None

    def initialize(self, train_X, test_X, unlabeled_X):
        if not path.exists('./feature_generator/doc2vec'):
            system('mkdir ./feature_generator/doc2vec')
        assert path.isdir('./feature_generator/doc2vec'), 'doc2vec should be a directory!'
        #build word2vec.c
        system('gcc ./feature_generator/word2vec.c -o ./feature_generator/word2vec -lm -pthread -O3 -march=native -funroll-loops')
        system('mv ./feature_generator/word2vec ./feature_generator/doc2vec/')

        #build dataset
        self.doc2id = {}
        _id = 0
        fout = open('./feature_generator/doc2vec/alldata-id.txt', 'w')
        for x in train_X + test_X + unlabeled_X:
            self.doc2id[x] = _id
            print >>fout, '_*%d %s' % (_id, x)
            _id += 1
        system('gshuf ./feature_generator/doc2vec/alldata-id.txt > ./feature_generator/doc2vec/alldata-id-shuf.txt')

        #run word2vec
        command = ('./feature_generator/doc2vec/word2vec -train ' +
                   './feature_generator/doc2vec/alldata-id-shuf.txt ' +
                   '-output ./feature_generator/doc2vec/vectors.txt ' +
                   '-cbow 0 ' +
                   ('-size %d ' % self.dimension) +
                   ('-window %d ' % self.window) +
                   ('-negative %d ' % self.negative) +
                   ('-hs %d ' % self.hs) +
                   ('-sample %f ' % self.sample) +
                   ('-min-count %d ' % self.min_count) +
                   '-threads 40 -binary 0 -iter 20 -sentence-vectors 1'
                  )
        print command
        system(command)

        #collect results
        self.id2vec = {}
        pattern = re.compile(r'^_\*\d+$')
        with open('./feature_generator/doc2vec/vectors.txt', 'r') as fin:
            fin.readline()
            for line in fin:
                e = line.rstrip().split(' ')
                if pattern.match(e[0]) is None:
                    continue
                _id = int(e[0][2:])
                vec = np.array(map(float, e[1:]))
                assert len(vec) == self.dimension
                self.id2vec[_id] = vec

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        ret = np.zeros((len(X), self.dimension), dtype='float32')
        for idx, review in enumerate(X):
            _id = self.doc2id[review]
            ret[idx] = self.id2vec[_id]
        return ret

    def get_dimension(self):
        return self.dimension






