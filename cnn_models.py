"""
@author: Luiza Sayfullina
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class textCNN(nn.Module):
    def __init__(self, dataset, kernel_num=50, embed_dim=200, dropout=0.5, class_num=2):
        super(textCNN, self).__init__()
        kernel_sizes = [2, 3, 4, 5]
        voc_size = len(dataset.vocabulary_set)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
        self.embed = nn.Embedding(voc_size, embed_dim, padding_idx=dataset.voc_to_index['<UNK>'])
        self.embed.weight.data = torch.from_numpy(np.array(dataset.embed_matrix, dtype=np.float32))

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)  # (N,Cin,H,W) to make Cin = 1
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,H), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)  # to concat across all kernels
        x = self.dropout(x)  # (N,NUM_KERNEL_TYPES*NUM_CLASSES)
        logit = self.fc1(x)  # (N,NUM_CLASSES)
        return logit

class textCNNWithEmbedding(nn.Module):
    def __init__(self, dataset, kernel_num=50, embed_dim=200, dropout=0.5, class_num=2):
        super(textCNNWithEmbedding, self).__init__()
        kernel_sizes = [2, 3, 4, 5]
        voc_size = len(dataset.vocabulary_set)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num + embed_dim, class_num)
        self.embed = nn.Embedding(voc_size, embed_dim, padding_idx=dataset.voc_to_index['<UNK>'])
        self.embed.weight.data = torch.from_numpy(np.array(dataset.embed_matrix, dtype=np.float32))

    def forward(self, x, skill):
        x = self.embed(x)
        x = x.unsqueeze(1)  # (N,Cin,H,W) to make Cin = 1
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,H), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)  # to concat across all kernels
        skill_mean = torch.mean(self.embed(skill), dim=1)
        x = torch.cat((x, skill_mean), 1)
        x = self.dropout(x)  # (N,NUM_KERNEL_TYPES*NUM_CLASSES)
        logit = self.fc1(x)  # (N,NUM_CLASSES)
        return logit