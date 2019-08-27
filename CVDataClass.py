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

class CVData():

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

        self.Xtest, self.Ytest, self.SoftSkillsTest = self.load_data()
        max_number_words = -1

        for s in self.SoftSkillsTest:
            if len(s.split(' ')) > max_number_words:
                max_number_words = len(s.split(' '))

        self.find_vocabulary()
        self.get_glove_embedding()

    def load_data(self, filename='./dataset/cv_test.csv'):

        """
        The list of mode options:
            'tagged'  - ... <b> soft skill </b> ...
            'masked' -  ... xxx xxx ...
            'unmodified' - ... soft skill ...

        The original formatting
            'tagged strangely' - ... < b > soft skill < / b >
        """

        Y = []
        X = []
        soft_skills = []
        mode = self.mode

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                res = re.search('<b>(.+)</b>', row['sentence'])
                soft_skill = res.group().replace('<b>', '').replace('</b>', '')
                soft_skills.append(soft_skill)
                Y.append(int(float(row['is_soft_skill'])>0.5))
                if mode == 'unmodified':
                    X.append(row['sentence'].replace('<b>', '').replace('</b>', ''))
                if mode == 'tagged':
                    X.append(row['sentence'].replace('<b>', 'bbb ').replace('</b>', ' eee'))
                if mode == 'masked':
                    len_soft_skill = len(soft_skill.split(' '))
                    xxx_string = ' '.join(['xxx'] * len_soft_skill)
                    X.append(row['sentence'].replace('<b>' + soft_skill + '</b>', xxx_string))
        X = [x.lower() for x in X]
        soft_skills = [s.lower() for s in soft_skills]
        return X, Y, soft_skills

    def find_vocabulary(self):

        '''Finds the vocabulary
        according to the preprocessing options
        like lemmatization, stemming selected
        '''

        self.count_vect = CountVectorizer(decode_error='replace', max_features=self.max_voc)
        self.count_vect.fit_transform(self.Xtest).toarray()
        self.vocabulary = self.count_vect.get_feature_names()
        if self.mode == 'tagged' and ('eee' not in self.vocabulary and 'bbb' not in self.vocabulary):
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
            (widx_test, lens_test) = self.get_word_indices(self.Xtest, return_lens=True)
        else:
            widx_test = self.get_word_indices(self.Xtest)

        widx_ss_test = self.get_soft_skill_indices(self.SoftSkillsTest)

        if return_lens:
            return (widx_test, self.Ytest, lens_test, widx_ss_test)
        else:
            return (widx_test, self.Ytest, widx_ss_test)

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
                word_index_matrix[k, :len(words_ind)] = words_ind[0:self.max_num_words]
                k+=1
                lens_array.append(min(len(words_ind),self.max_num_words))
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
    dataset = CVData()
    (Xtest, Ytest, widx_ss_test) = dataset.get_word_indices_all()