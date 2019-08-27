"""
@author: Luiza Sayfullina
"""

import re, os
import numpy as np
import torchtext.vocab as vocab
from sklearn.feature_extraction.text import CountVectorizer
import unittest
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import csv

class JobData():
    def __init__(self, **kwargs):

        try:
            self.max_voc = kwargs['max_voc_size']
            self.max_num_words = kwargs['max_num_words']
            self.data_folder = kwargs['data_folder']
            self.mode = kwargs['mode']
        except:
            self.max_voc = 5000
            self.max_num_words = 50
            self.data_folder = './dataset'
            self.mode = 'tagged'  # tagged or masked or unmodified

        self.Xtrain, self.Ytrain, self.SoftSkillsTrain = self.load_data(filename='job_train.csv')
        self.Xtest, self.Ytest, self.SoftSkillsTest = self.load_data(filename='job_test.csv')

        max_number_words = -1

        for s in self.SoftSkillsTrain:

            if len(s.split(' ')) > max_number_words:
                max_number_words = len(s.split(' '))

        for s in self.SoftSkillsTest:
            if len(s.split(' ')) > max_number_words:
                max_number_words = len(s.split(' '))

        self.find_vocabulary()
        self.get_glove_embedding()
        # self.get_word_indices_all()

    def load_data(self, filename='job_train.csv'):

        """
        The list of mode options:
            'tagged'  - ... <b> soft skill </b> ...
            'masked' -  ... xxx xxx ...
            'unmodified' - ... soft skill ...
        """

        Y = []
        X = []
        soft_skills = []
        mode = self.mode

        with open(os.path.join(self.data_folder, filename)) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                Y.append(row['is_soft_skill'])
                soft_skills.append(row['soft_skill'])
                if mode == 'unmodified':
                    len_soft_skill = len(row['soft_skill'].split(' '))
                    xxx_string = ' '.join(['xxx'] * len_soft_skill)
                    X.append(row['context'].replace(xxx_string, row['soft_skill']))
                if mode == 'tagged':
                    len_soft_skill = len(row['soft_skill'].split(' '))
                    xxx_string = ' '.join(['xxx'] * len_soft_skill)
                    X.append(row['context'].replace(xxx_string, 'bbb ' + row['soft_skill'] + ' eee'))
                if mode == 'masked':
                    X.append(row['context'])

        return X, Y, soft_skills

    def find_vocabulary(self):

        '''Finds the vocabulary
        according to the preprocessing options
        like lemmatization, stemming selected
        '''

        self.count_vect = CountVectorizer(decode_error='replace', max_features=self.max_voc)
        self.count_vect.fit_transform(self.Xtrain).toarray()
        self.vocabulary = self.count_vect.get_feature_names()
        if self.mode == 'tagged' and ('bbb' not in self.vocabulary or 'eee' not in self.vocabulary):
            print('tags are not in the vocabulary')
        self.vocabulary.append('<UNK>')
        self.vocabulary_set = set(self.vocabulary)

        self.voc_to_index = dict({word: i for i, word in enumerate(self.vocabulary)})

    def get_glove_embedding(self):

        # "from:https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb"
        dim = 200  # fasttext.en.300d
        glove = vocab.GloVe(name='6B', dim=200)
        # model = Word2Vec.load('luiza_test')
        self.embed_matrix = np.random.uniform(-0.01, 0.01, size=(len(self.vocabulary), dim))
        # glove_indices = {i:model.wv[word] for i, word in enumerate(self.vocabulary) if word in model.wv}

        glove_indices = {glove.stoi[word]: i for i, word in enumerate(self.vocabulary) if word in glove.stoi}
        keys = np.array(list(glove_indices.keys()))
        values = np.array(list(glove_indices.values()))
        self.embed_matrix[values] = glove.vectors.numpy()[keys]

    def get_word_indices_all(self, return_lens=False):

        if return_lens:
            (widx_train, lens_train) = self.get_word_indices(self.Xtrain, return_lens=True)
            (widx_test, lens_test) = self.get_word_indices(self.Xtest, return_lens=True)
        else:
            widx_train = self.get_word_indices(self.Xtrain)
            widx_test = self.get_word_indices(self.Xtest)

        widx_ss_train = self.get_soft_skill_indices(self.SoftSkillsTrain)
        widx_ss_test = self.get_soft_skill_indices(self.SoftSkillsTest)

        if return_lens:
            return (widx_train, lens_train, self.Ytrain, widx_ss_train), (widx_test, lens_test, self.Ytest, widx_ss_test)
        else:
            return (widx_train, self.Ytrain, widx_ss_train), (widx_test, self.Ytest, widx_ss_test)

    def get_word_indices(self, X, return_lens=False):

        N = len(X)
        word_index_matrix = np.ones((N, self.max_num_words)) * self.voc_to_index['<UNK>']
        lens_array = []
        unk_index = self.voc_to_index['<UNK>']
        k = 0
        for i, sentence in enumerate(X):
            words = word_tokenize(sentence)
            words_ind = [self.voc_to_index.get(word, unk_index) for word in words if word in self.vocabulary]
            if len(words_ind) >= 3:
                word_index_matrix[k, :len(words_ind)] = words_ind
                k+=1
                lens_array.append(len(words_ind))
        if return_lens:
            return word_index_matrix[:k, :], lens_array
        else:
            return word_index_matrix[:k, :]

    def get_soft_skill_indices(self, X):

        N = len(X)
        word_index_matrix = np.ones((N, 10)) * self.voc_to_index['<UNK>']
        unk_index = self.voc_to_index['<UNK>']
        k = 0
        for i, sentence in enumerate(X):
            words = word_tokenize(sentence)
            words_ind = [self.voc_to_index.get(word, unk_index) for word in words if word in self.vocabulary]
            if words_ind:
                word_index_matrix[i, :len(words_ind)] = words_ind
            else:
                print(i, '-- no vector for soft skill:', sentence)
        return word_index_matrix

if __name__ == '__main__':
    dataset = JobData()
    (Xtrain, Ytrain, widx_ss_train), (Xtest, Ytest, widx_ss_test) = dataset.get_word_indices_all()