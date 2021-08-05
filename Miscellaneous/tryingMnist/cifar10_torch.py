#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:19:09 2019

@author: ashrey
"""


import torch
import torch.nn as nn
import torchvision
import os
import torchvision.transforms as transforms
import copy


'''
mnist

'''
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

#path = os.getcwd()+'/mnist/'
train = torchvision.datasets.MNIST(root = './mnist',download=True,transform=trans)
#test = torchvision.datasets.MNIST(root='./mnist',train=False,download=True)
#print(len(test), len(train),' length of test and train')

n_epochs = 3
batch_size_train = 1
batch_size_test = 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10

#random_seed = 1
torch.backends.cudnn.enabled = False
#torch.manual_seed(random_seed)


#a,b = torch.utils.data.random_split(train, [10,len(train)-10])

data_loader_train = torch.utils.data.DataLoader(train,
                                                batch_size=1,
                                                shuffle=True )

'''
to show/display the image

image,target = train[0]
image.show()
'''

'''
for cifar10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
'''

''' for mnist
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])'''


''' standardisation of mnist, mean = 0 and variance = 1
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
'''

'''
for i,data in enumerate(data_loader_train,0):
    inputs,labels = data
    print(inputs.shape,labels.shape)
    break
'''
batch_Size = 64

def data_load(data='train'):
    #trainsize = 200
    #testsize = 40

    if data == 'test':        
        samples = torchvision.datasets.CIFAR10(root = './mnist',train=False,download=True,transform=trans)
        size = 64
        a,_ = torch.utils.data.random_split(samples, [size,len(samples)-size])

    else:
        samples = torchvision.datasets.CIFAR10(root = './mnist',train=True,download=True,transform=trans)
        size = 1024
        a,_ = torch.utils.data.random_split(samples,[size,len(samples)-size])

    data_loader = torch.utils.data.DataLoader(a,
                                              batch_size=batch_Size,
                                              shuffle=True )
    return data_loader

def f(): raise Exception("Found exit()")

train = data_load()
for a,b in train:
    #print(a,b)
    print(a.shape,b.shape)
    f()
    #exit()
print(train.shape,'train shape')
f()
rnn = Model([28,10,10],0.01)
fx = rnn.evaluate_proposal(train)