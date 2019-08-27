"""
@author: Luiza Sayfullina
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):

    def init_hidden(self, batch_size_= None):

        ''' Before we've done anything, we dont have any hidden state.
            Refer to the Pytorch documentation to see exactly why they have this dimensionality.
            The axes semantics are (num_layers, minibatch_size, hidden_dim)
        '''

        if not batch_size_:
            batch_size_ = self.batch_size

        if self.use_cuda:
            return (Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()),
                    Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()))
        else:
            return (Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size)))

    def get_ort_weight(self, m, n):
        return torch.nn.init.orthogonal(torch.FloatTensor(m,n))

    def __init__(self, dataset, input_size, hidden_size, num_layers, num_classes=2, batch_size=512, dropout=0.5, use_ort_weights=False, use_cuda=False):

        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        voc_size, dim = np.shape(dataset.embed_matrix)
        self.hidden = self.init_hidden()
        self.embed = nn.Embedding(voc_size, dim)
        if use_ort_weights:
            self.lstm.weight_ih_l0.data = torch.cat\
                ((self.get_ort_weight(hidden_size, input_size),self.get_ort_weight(hidden_size, input_size),self.get_ort_weight(hidden_size, input_size),self.get_ort_weight(hidden_size, input_size)),0)
            self.lstm.weight_hh_l0.data = torch.cat((self.get_ort_weight(hidden_size, hidden_size),\
                self.get_ort_weight(hidden_size,hidden_size),self.get_ort_weight(hidden_size, hidden_size),self.get_ort_weight(hidden_size, hidden_size)),0)

        self.embed.weight.data = torch.from_numpy(np.array(dataset.embed_matrix, dtype=np.float32))

    def forward(self, x, lens):

        x = self.embed(x)
        x = pack_padded_sequence(x, lens, batch_first=True)
        out, self.hidden = self.lstm(x, self.hidden)
        res = self.dropout(torch.cat([self.hidden[0][0], self.hidden[1][0]], 1))
        out = self.fc(res)

        return out

class LSTMWithEmbedding(nn.Module):

        def init_hidden(self, batch_size_=None):

            ''' Before we've done anything, we dont have any hidden state.
                Refer to the Pytorch documentation to see exactly why they have this dimensionality.
                The axes semantics are (num_layers, minibatch_size, hidden_dim)
            '''

            if not batch_size_:
                batch_size_ = self.batch_size

            if self.use_cuda:
                return (Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()),
                        Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size).cuda()))
            else:
                return (Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size)),
                        Variable(torch.zeros(self.num_layers, batch_size_, self.hidden_size)))

        def get_ort_weight(self, m, n):
            return torch.nn.init.orthogonal(torch.FloatTensor(m, n))

        def __init__(self, dataset, input_size, hidden_size, num_layers, num_classes=2, batch_size=512, dropout=0.5,
                     use_ort_weights=False, use_cuda=False):

            super(LSTMWithEmbedding, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
            self.dropout = nn.Dropout(dropout)
            self.use_cuda = use_cuda
            self.batch_size = batch_size
            voc_size, dim = np.shape(dataset.embed_matrix)
            self.fc = nn.Linear(2 * hidden_size + dim, num_classes)
            self.hidden = self.init_hidden()
            self.embed = nn.Embedding(voc_size, dim)
            if use_ort_weights:
                self.lstm.weight_ih_l0.data = torch.cat \
                    ((self.get_ort_weight(hidden_size, input_size), self.get_ort_weight(hidden_size, input_size),
                      self.get_ort_weight(hidden_size, input_size), self.get_ort_weight(hidden_size, input_size)), 0)
                self.lstm.weight_hh_l0.data = torch.cat((self.get_ort_weight(hidden_size, hidden_size), \
                                                         self.get_ort_weight(hidden_size, hidden_size),
                                                         self.get_ort_weight(hidden_size, hidden_size),
                                                         self.get_ort_weight(hidden_size, hidden_size)), 0)

            self.embed.weight.data = torch.from_numpy(np.array(dataset.embed_matrix, dtype=np.float32))

        def forward(self, x, skills, lens):

            x = self.embed(x)
            x = pack_padded_sequence(x, lens, batch_first=True)
            out, self.hidden = self.lstm(x, self.hidden)
            res = self.dropout(torch.cat([self.hidden[0][0], self.hidden[1][0]], 1))
            skills = torch.mean(self.embed(skills), dim=1)
            out = self.fc(self.dropout(torch.cat((res, skills), 1)))
            return out
