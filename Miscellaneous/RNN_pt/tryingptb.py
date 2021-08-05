# To add a new cell, type '#  '
# To add a new markdown cell, type '#   [markdown]'
#  

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import math
#np.random.seed(1)
import nn_mcmc_plots as mcmcplt
import os
import copy
import argparse
import time
#np.random.seed(1)
import datetime
import multiprocessing
import gc
import matplotlib as mpl
mpl.use('agg')
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse


mplt = mcmcplt.Mcmcplot()
weightdecay = 0.01

def f(): raise Exception("Found exit()")


#  

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    
    return word_to_id

def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data():
    # get the data paths
    train_path = 'data/ptb.train.txt'
    valid_path = 'data/ptb.valid.txt'
    test_path = 'data/ptb.test.txt'

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(vocabulary,' - length of vocabulary')
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        flag = 0
        while True:
            if(flag == 1):
                break
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                    flag = 1
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


#  
name = "ptb"
train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
num_steps = 25
batch_size = 20
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,skip_step=num_steps)
test_data_generator = KerasBatchGenerator(test_data, num_steps, batch_size, vocabulary,skip_step=num_steps)
trainData = list(train_data_generator.generate())
testData = list(test_data_generator.generate())
print(len(trainData), ' is the length of train dataset')
print(len(testData), ' is the length of test dataset')

'''x,y = trainData[0]
print(x.shape,y.shape)'''
train_x=[]
train_y=[]
test_x = []
test_y = []
idx = 0
breakat = 200
for iter1 in trainData:
    if(idx == breakat):
        break
    idx+=1
    a,b = iter1
    train_x.append(a.astype(int))
    train_y.append(b)

for iter2 in testData:
    c,d = iter2
    test_x.append(c.astype(int))
    test_y.append(d)

# shape of inputs
print('shape of input sample is (x,y) ',train_x[0].shape,train_y[0].shape)
print('number of samples is: for train and test - ', len(train_x), len(test_x))
# trimming x and y

#  
class Model(nn.Module):

        # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self, topo,lrate,rnn_net = 'LSTM'):
        super(Model, self).__init__()
        # assuming num_dicrections to be 1 for all the cases
        # Defining some parameters
        self.hidden_dim = topo[1] #hidden_dim
        self.n_layers = 1
        self.batch_size = batch_size
        self.lrate = lrate
        if rnn_net == 'RNN':
            self.hidden = torch.ones(self.n_layers, self.batch_size, self.hidden_dim)
            self.rnn = nn.RNN(input_size = topo[0], hidden_size = topo[1])

        if rnn_net == 'GRU':
            self.hidden = torch.ones((self.n_layers,self.batch_size,self.hidden_dim))
            self.rnn = nn.GRU(input_size = topo[0], hidden_size = topo[1])

        if rnn_net == 'LSTM':
            self.hidden = (torch.ones((self.n_layers,self.batch_size,self.hidden_dim)), torch.ones((self.n_layers,self.batch_size,self.hidden_dim)))
            self.rnn = nn.LSTM(input_size = topo[0], hidden_size = topo[1])

            # Fully connected layer
        self.fc = nn.Linear(topo[1],topo[2])
        self.topo = topo
        self.vocab_size = vocabulary
        self.embed = nn.Embedding(self.vocab_size,topo[0])
        self.softmax = nn.Softmax(dim=2)
        print(rnn_net, ' is rnn net')


    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))

    def forward(self, x):
        outmain=torch.zeros((len(x),self.batch_size,num_steps,vocabulary))
        for i,sample in enumerate(x):
            print(i,'th sample running')
            # sample = torch.FloatTensor(sample)
            sample = torch.LongTensor(sample)
            # print(sample)
            # input of shape (seq_len, batch, input_size):
            # print('shape 1',sample.shape)
            sample = self.embed(sample)
            sample = sample.reshape(num_steps,batch_size,self.topo[0])
            # print('shape 2',sample.shape)
            # sample = sample.view(sample.shape[0],1,self.topo[0])
            hidden = copy.deepcopy(self.hidden)
            out, h1 = self.rnn(sample, hidden)
            out = self.fc(out)
            # out = self.sigmoid(out)
            # print(' shape of out',out.shape)
            out = self.softmax(out) # check dimension right or not
            outmain[i,] = copy.deepcopy(out.detach()).reshape(self.batch_size,num_steps,vocabulary)
        return outmain

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden        # Create a brnn -- assuming h0 size = 5 i.e. hidden layer has 5 nuerons

    def dictfromlist(self,param):
        dic = {}
        i=0
        for name in sorted(self.state_dict().keys()):
            dic[name] = torch.FloatTensor(param[i:i+(self.state_dict()[name]).view(-1).shape[0]]).view(self.state_dict()[name].shape)
            i += (self.state_dict()[name]).view(-1).shape[0]
        #self.loadparameters(dic)
        return dic
        
    def evaluate_proposal(self,x,w=None):
        if w is None:
            y_pred = self.forward(x)
            return y_pred.detach().numpy()
        else:
            self.loadparameters(w)
            y_pred = self.forward(x)
            return np.array(y_pred.detach())


#    def loss(self,fx,y):
#        fx = torch.FloatTensor(fx)
#        y = torch.FloatTensor(y)
#        criterion = nn.MSELoss()
#        loss = criterion(fx,y)
#        return loss.item()

    def langevin_gradient(self,x,y,w):
        #print(w)
        self.loadparameters(w)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(),lr = self.lrate)
        for i,sample in enumerate(x):
            # sample = torch.FloatTensor(sample).view(sample.shape[0],1,self.topo[0])
            hidden = copy.deepcopy(self.hidden)
            optimizer.zero_grad()
            sample = self.embed(sample)
            out, h1 = self.rnn(sample, hidden)
            out = self.fc(out[-1])
            out = self.sigmoid(out)
            loss = criterion(out,torch.FloatTensor(y[i]).view(out.shape))
            loss.backward()
            optimizer.step()
        return copy.deepcopy(self.state_dict())


    #returns a np arraylist of weights and biases -- layerwise
    # in order i.e. weight and bias of input and hidden then weight and bias for hidden to out
    def getparameters(self,w = None):
        l=np.array([1,2])
        dic = {}
        if w is None:
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in sorted(dic.keys()):
            l=np.concatenate((l,np.array(copy.deepcopy(dic[name])).reshape(-1)),axis=None)
        l = l[2:]
        return l


    # input a dictionary of same dimensions
    def loadparameters(self,param):
        self.load_state_dict(param)

    # input weight dictionary, mean, std dev
    def addnoiseandcopy(self,w,mea,std_dev):
        dic = {}
        for name in (w.keys()):
            dic[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean = mea, std = std_dev)
        return dic


Input = 60
Output = vocabulary
Hidden = 100
#ip = 4 #input
#output = 1
topology = [Input, Hidden, Output,vocabulary]


learn_rate = 0.01
rnn = Model(topology,learn_rate)
w = copy.deepcopy(rnn.state_dict())
step_w = 0.025
w_size = sum(p.numel() for p in rnn.parameters())
print("yaha tak to done hai")
for i in range(1):
    pred_train  = rnn.evaluate_proposal(train_x,w) #
    print(' reached 1')
    pred_test  = rnn.evaluate_proposal(test_x, w) #
    print('reached 2')
    w_gd = rnn.langevin_gradient(train_x,train_y, copy.deepcopy(w)) # Eq 8
    w_proposal = rnn.addnoiseandcopy(w_gd,0,step_w) #np.random.normal(w_gd, step_w, w_size) # Eq 7
    w_prop_gd = rnn.langevin_gradient(train_x,train_y, copy.deepcopy(w_proposal))
    print('done all')
