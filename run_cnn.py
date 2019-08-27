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
# downloaded from https://github.com/Bjarten/early-stopping-pytorch
from pytorchtools import EarlyStopping
import torch.nn.functional as F
from utilities import find_recall_for_fixed_precision

mode = 'masked' # other modes are 'tagged', 'masked'

dataset = JobData(mode=mode)
(Xtrain, Ytrain, widx_ss_train), (Xtest, Ytest, widx_ss_test) = dataset.get_word_indices_all()

Xtrain, Xtest = np.array(Xtrain, dtype=np.int64), np.array(Xtest, dtype=np.int64)
Ytrain, Ytest = np.array(Ytrain, dtype=np.int64), np.array(Ytest, dtype=np.int64)

widx_ss_train = np.array(widx_ss_train, dtype=np.int64)
widx_ss_test = np.array(widx_ss_test, dtype=np.int64)

num_epochs = 100
learning_rate = 0.001
embed_dim = 200
class_num = 2
dropout = 0.5
num_words = 50 # max number of words in the input
batch_size = 512
kernel_num = 50 # number of cnn kernels per each type
voc_size = len(dataset.vocabulary_set)

train_data = torch.from_numpy(Xtrain[0:1000,:])
train_labels = torch.from_numpy(Ytrain[0:1000])
train_skills = torch.from_numpy(widx_ss_train[0:1000,:])

valid_data = torch.from_numpy(Xtrain[1000:,:])
valid_labels = torch.from_numpy(Ytrain[1000:])
valid_skills = torch.from_numpy(widx_ss_train[1000:,:])

test_data = torch.from_numpy(Xtest)
test_labels = torch.from_numpy(Ytest)
test_skills = torch.from_numpy(widx_ss_test)

Ntrain = train_data.size(0)
Nvalid = valid_data.size(0)
Ntest =  test_data.size(0)

print('The sizes of data sets are:')
print(Ntrain, Nvalid, Ntest)

train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data,test_labels)
valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_labels)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

cnn = textCNN(dataset, kernel_num=50, embed_dim=200, dropout=0.5, class_num=2)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

best_valid_acc = -1
valid_acc_history = []
early_stopping = EarlyStopping(patience=3, verbose=True, delta=0)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)  # .cuda()
        labels = Variable(labels)  # .cuda()
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = Variable(images)  # .cuda()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    curr_acc = correct * 100.0 / total
    valid_acc_history.append(curr_acc)

    early_stopping(-curr_acc, cnn) # since early stoppping works in loss mode, negate the accuracy

    if early_stopping.early_stop:
        print("Early stopping")
        break

    # if curr_acc > best_valid_acc:
    #         best_valid_acc = curr_acc
    #         torch.save({
    #             'epoch': epoch,
    #             'model': cnn.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'accuracyHistory': valid_acc_history,
    #             'accuracy': best_valid_acc
    #         }, 'vanila_cnn_checkpoint')


# load the last checkpoint with the best model
cnn.load_state_dict(torch.load('checkpoint.pt'))
# Test the Model
cnn.eval()

correct = 0
total = 0

pred_border = []
y_true_border = []

for images, labels in test_loader:
    images = Variable(images)  # .cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    pred = F.softmax(outputs, 1).data.cpu().numpy()[:, 1]
    pred_border.extend(pred)
    y_true_border.extend(labels.cpu().numpy())

#print('Test Accuracy of the model: %d %%' % (100.0 * correct / total))
#test_acc = 100.0 * correct / (total * 1.0)

print('Mode:', mode)
print('Dataset: Job Data')
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

    return test_loader

test_loader = get_cv_data(mode=mode, batch_size=512)
pred_border = []
y_true_border = []

for images, labels in test_loader:
    images = Variable(images)  # .cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    pred = F.softmax(outputs, 1).data.cpu().numpy()[:, 1]
    pred_border.extend(pred)
    y_true_border.extend(labels.cpu().numpy())

print('Mode:', mode)
print('Dataset: CV')
desired_precision = 0.90
precision, recall, f1_w, f1 = find_recall_for_fixed_precision(y_true_border, pred_border, desired_precision)
print('Precision: {0}, Recall: {1}, F1_weighted: {2}'.format(precision, recall, f1_w))
