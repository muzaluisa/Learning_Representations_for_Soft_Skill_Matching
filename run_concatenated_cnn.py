"""
@author: Luiza Sayfullina
Code for paper "Learning Representations for Soft Skills Matching"
https://arxiv.org/abs/1807.07741 or https://link.springer.com/chapter/10.1007/978-3-030-11027-7_15
"""

import numpy as np
from JobDataClass import JobData
from CVDataClass import CVData
from cnn_models import textCNN, textCNNWithEmbedding
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# downloaded from https://github.com/Bjarten/early-stopping-pytorch
from pytorchtools import EarlyStopping
from utilities import find_recall_for_fixed_precision

mode = 'unmodified' # the type of input representation
dataset = JobData(mode=mode)
(Xtrain, Ytrain, Skills_train), (Xtest, Ytest, Skills_test) = \
    dataset.get_word_indices_all(return_lens=False)

voc_size, dim = np.shape(dataset.embed_matrix)

Ntrain = len(Xtrain)//2
Ntest = len(Xtest)
Nvalid = len(Xtrain[Ntrain:])

print('The size of test data:', Ntest)
print('The size of train data:', Ntrain)

Xtrain, Xtest = np.array(Xtrain,dtype=np.int64), np.array(Xtest,dtype=np.int64)
Ytrain, Ytest = np.array(Ytrain,dtype=np.int64), np.array(Ytest,dtype=np.int64)
Skills_train = np.array(Skills_train,dtype=np.int64)
Skills_test = np.array(Skills_test,dtype=np.int64)

print('The number of positive training samples:', sum(Ytrain))
print('The number of negative training samples:', len(Ytrain)-sum(Ytrain))

print('The number of positive test samples:', sum(Ytest))
print('The number of negative test samples:', len(Ytest) - sum(Ytest))

N, num_words = np.shape(Xtrain)
batch_size = 512

train_data = torch.from_numpy(Xtrain[0:Ntrain])
train_labels = torch.from_numpy(Ytrain[0:Ntrain])
train_skills = torch.from_numpy(Skills_train[0:Ntrain])

test_data = torch.from_numpy(Xtest)
test_labels = torch.from_numpy(Ytest)
test_skills = torch.from_numpy(Skills_test)

valid_data = torch.from_numpy(Xtrain[Ntrain:])
valid_labels = torch.from_numpy(Ytrain[Ntrain:])
valid_skills = torch.from_numpy(Skills_train[Ntrain:])

print('size of valid data', np.shape(Xtrain[Ntrain:]))
print('size of valid skills', np.shape(Skills_train[Ntrain:]))

train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data,test_labels)
valid_dataset = torch.utils.data.TensorDataset(valid_data,valid_labels)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

num_epochs = 100
learning_rate = 0.001
embed_dim = 200
class_num = 2
dropout = 0.5
num_words = 50 # max number of words in the input
batch_size = 512
kernel_num = 50 # number of cnn kernels per each type
voc_size = len(dataset.vocabulary_set)

cnn = textCNNWithEmbedding(dataset, kernel_num=kernel_num, embed_dim=embed_dim, dropout=dropout, class_num=class_num)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

valid_acc_history = []
use_cuda = False

gradient_clipping_value = 0
var_len = True

early_stopping = EarlyStopping(patience=3, verbose=True, delta=0)

for epoch in range(num_epochs):

    print('Epoch:', epoch)
    train_loss_avg = 0

    idx = np.array(np.random.permutation(range(Ntrain)))
    idx_torch = torch.LongTensor(idx)
    train_data = torch.index_select(train_data, 0, idx_torch)
    train_labels = torch.index_select(train_labels, 0, idx_torch)

    for i in range(int(np.ceil(Ntrain // batch_size))):
        if (batch_size * (i + 1)) <= Ntrain:
            images = train_data[batch_size * i:batch_size * (i + 1)]
            labels = train_labels[batch_size * i:batch_size * (i + 1)]
            skills = train_skills[batch_size * i:batch_size * (i + 1)]
        else:
            images = train_data[batch_size * i:]
            labels = train_labels[batch_size * i:]
            skills = train_skills[batch_size * i:]

        optimizer.zero_grad()
        outputs = cnn(images, skills)
        loss = criterion(outputs, labels)
        loss.backward()

        if gradient_clipping_value > 0:
            torch.nn.utils.clip_grad_norm(cnn.parameters(), gradient_clipping_value)

        optimizer.step()
        train_loss_avg += loss.data[0]

    total = 0
    correct = 0
    cnn.eval()

    for i, (images, labels) in enumerate(valid_loader):

        if var_len:
            if batch_size * (i + 1) <= Nvalid:
                images = valid_data[i * batch_size:(i + 1) * batch_size]
                skills = valid_skills[i * batch_size:(i + 1) * batch_size]
            else:
                images = valid_data[i * batch_size:]
                skills = valid_skills[i * batch_size:]

        outputs = cnn(images, skills)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).numpy().sum()

    curr_acc = correct * 100.0 / total
    valid_acc_history.append(curr_acc)
    print('Current accuracy : ', curr_acc)

    early_stopping(-curr_acc, cnn)

    if early_stopping.early_stop:
        print("Early stopping")
        break
    cnn.train()

cnn.load_state_dict(torch.load('checkpoint.pt'))

cnn.eval()
pred_border = []
y_true_border = []

total = 0
correct = 0

for i, (images, labels) in enumerate(test_loader):

    if var_len:
        if batch_size * (i + 1) <= Ntest:
            skills = test_skills[i * batch_size:(i + 1) * batch_size]
        else:
            skills = test_skills[i * batch_size:]

    outputs = cnn(images, skills)

    _, predicted = torch.max(outputs.data, 1)
    pred = F.softmax(outputs, 1).data.cpu().numpy()[:, 1]
    pred_border.extend(pred)
    y_true_border.extend(labels.cpu().numpy())

    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model: %d %%' % (100.0 * correct / total))
test_acc = 100 * correct / (total + 0.0)

print('Mode:', mode)
desired_precision = 0.95
precision, recall, f1_w, f1 = find_recall_for_fixed_precision(y_true_border, pred_border, desired_precision)
print('Precision: {0}, Recall: {1}, F1_weighted: {2}'.format(precision, recall, f1_w))


def get_cv_data(mode=mode, batch_size=512):

    """

    :param mode: type of input representation(tagged, unmodified or masked), should be the same as used for training
    :param batch_size:
    :return:
    """

    dataset = CVData(mode=mode)
    (Xtest, Ytest, widx_ss_test) = dataset.get_word_indices_all()


    Xtest, Ytest = np.array(Xtest, dtype=np.int64), np.array(Ytest, dtype=np.int64)
    widx_ss_test = np.array(widx_ss_test, dtype=np.int64)

    test_data = torch.from_numpy(Xtest)
    test_labels = torch.from_numpy(Ytest)
    test_skills = torch.from_numpy(widx_ss_test)

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return test_loader, test_skills, Ntest

test_loader, test_skills, Ntest = get_cv_data(mode=mode, batch_size=512)

pred_border = []
y_true_border = []
total = 0
correct = 0

for i, (images, labels) in enumerate(test_loader):

    if var_len:
        if batch_size * (i + 1) <= Ntest:
            skills = test_skills[i * batch_size:(i + 1) * batch_size]
        else:
            skills = test_skills[i * batch_size:]

    outputs = cnn(images, skills)

    _, predicted = torch.max(outputs.data, 1)
    pred = F.softmax(outputs, 1).data.cpu().numpy()[:, 1]
    pred_border.extend(pred)
    y_true_border.extend(labels.cpu().numpy())

    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Mode:', mode)
print('Dataset: CV')
desired_precision = 0.90
precision, recall, f1_w, f1 = find_recall_for_fixed_precision(y_true_border, pred_border, desired_precision)
print('Precision: {0}, Recall: {1}, F1_weighted: {2}'.format(precision, recall, f1_w))