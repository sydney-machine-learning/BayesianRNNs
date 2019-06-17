#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:32:34 2019

@author: ashrey

python 3.6 --  working great

"""

#  %reset
#  %reset -sf


import torch
import torch.nn as nn
import numpy as np
import random
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
#np.random.seed(1)
import numpy as np
import nn_mcmc_plots as mcmcplt
import os
import copy


mplt = mcmcplt.Mcmcplot()

MinimumError = 0.00001
trainsize=299
testsize=99
weightdecay = 0.01

def data_loader(filename):
    f=open(filename,'r')
    x=[[]]
    count=0
    y=[[]]
    while(True):
        count+=1
        #print(count)
        text = f.readline()
        #print(text)
        if(text==''):
           break
        if(len(text.split()) == 0):
            #print(text)
            text=f.readline()
        if(text==''):
           break
        #print(text)
        t=int(text)
        a=[]
        ya=[]
        for i in range(0,t):
            temp=f.readline().split(' ')
            b=0.0
            for j in range(0,len(temp)):
                b=(float(temp[j]))
            a.append(b)
        #del a[0]
        x.append(a)
        temp=f.readline().split(' ')
        #print(temp)
        for j in range(0,len(temp)):
            if temp[j] != "\n":
                ya.append(float(temp[j]))
        y.append(ya)
    del x[0]
    del y[0]
    return x,y



def print_data(x,y):
    # assuming x is 3 dimensional and y is 2 dimensional
    for i in range(0,len(x)):
        for j in range(0,len(x[i])):
            print(x[i][j])
        print(y[i])
        print(' ')



def shuffledata(x,y):
    a=[]
    for i in range(0,len(x)):
        a.append(i)
    random.shuffle(a)
    x1 = []
    y1=[]
    for item in a:
        x1.append(x[item])
        y1.append(y[item])
    return x1,y1





class NN:

    def __init__(self,topo,lrate,x,y):
        # Defining input size, hidden layer size, output size and batch size respectively
        self.n_in,self.n_h, self.n_out, self.batch_size = topo[0], topo[1], topo[2], 50
        # Create dummy input and target tensors (data)
        #x = torch.randn(batch_size, n_in)
        #y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

        #pt_tensor_from_list = torch.FloatTensor(py_list)
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.learnRate = lrate
        print('hi')
        # Create a model
        self.model = nn.Sequential(nn.Linear(self.n_in, self.n_h),
                                   nn.Sigmoid(),
                                   nn.Linear(self.n_h, self.n_out),
                                   nn.Sigmoid())



    def evaluate_proposal(self,x,w=None):
        x = torch.FloatTensor(x)
        if w is None:
            y_pred = self.model(x)
            return copy.deepcopy(y_pred.detach().numpy())
        else:
            d = copy.deepcopy(self.model.state_dict())
            self.loadparameters(w)
            y_pred = self.model(x)
            self.loadparameters(d)
            return copy.deepcopy(y_pred.detach().numpy())


#    def loss(self,fx,y):
#        fx = torch.FloatTensor(fx)
#        y = torch.FloatTensor(y)
#        criterion = nn.MSELoss()
#        loss = criterion(fx,y)
#        return loss.item()


    #returns a np arraylist of weights and biases -- layerwise
    # in order i.e. weight and bias of input and hidden then weight and bias for hidden to out
    def getparameters(self,w = None):
        l=np.array([1,2])
        dic = {}
        if w is None:
            dic = self.model.state_dict()
        else:
            dic = w
        for name in dic.keys():
            l=np.concatenate((l.reshape(-1,1),(copy.deepcopy(dic[name])).numpy().reshape(-1,1)))
        l = l[2:]
        return l

    # input a dictionary of same dimensions
    def loadparameters(self,param):
        self.model.load_state_dict(param)
#        for name in (self.model.state_dict().keys()):
#            #print(param[name])
#            self.model.state_dict()[name] = param[name]

    def addnoiseandcopy(self,mea,std_dev):
        dict = {}
        for name in (self.model.state_dict().keys()):
            dict[name] = copy.deepcopy(self.model.state_dict()[name]) + torch.zeros(self.model.state_dict()[name].size()).normal_(mean = mea, std = std_dev)
        return dict

## Construct the loss function
#criterion = torch.nn.MSELoss()
#
## Construct the optimizer (Stochastic Gradient Descent in this case)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
## Gradient Descent
#for epoch in range(5):
#    # Forward pass: Compute predicted y by passing x to the model
#    y_pred = model(x)
#
#    # Compute and print loss
#    loss = criterion(y_pred, y)
#    print('epoch: ', epoch,' loss: ', loss.item())
#
#    # Zero gradients, perform a backward pass, and update the weights.
#    optimizer.zero_grad()
#
#    # perform a backward pass (backpropagation)
#    loss.backward()
#
#    # Update the parameters
#    #optimizer.step()
#
#
#model.parameters()
#
#
#params = list(model.parameters())
#print(len(params))
#print(params[0].size())
#
#w1 = []
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        #print(name, param.data)
#        w1 = param.data
#        break
#

class MCMC:
    def __init__(self, samples, learnrate, train_x, train_y,test_x,test_y, topology):
        self.samples = samples  # max epocs
        self.topology = topology  # NN topology [input, hidden, output]
        self.train_x = train_x#
        self.test_x = test_x
        self.train_y=train_y
        self.test_y=test_y
        self.learnrate = learnrate
        # ----------------

    def rmse(self, predictions, targets):
        predictions = (predictions)
        targets=np.array(targets)
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, x,y, w, tausq):
        #y = data[:, self.topology[0]]
        y=y
        fx = neuralnet.evaluate_proposal(x, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(np.array(y) - np.array(fx)) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w_list, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w_list)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def sampler(self):

        # ------------------- initialize MCMC
        testsize = len(self.test_x)   #self.testdata.shape[0]
        trainsize = len(self.train_x)
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.test_y  #self.testdata[:, netw[0]]
        y_train = self.train_y #self.traindata[:, netw[0]]
        #print(len(y_train))
        #print(len(y_test))

        # here
        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))

        # original -->    fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        #print('shape: ',np.array(y_train).shape[1])
        fxtrain_samples = np.ones((samples, trainsize,int(np.array(y_train).shape[1])))  # fx of train data over all samples
        # original --> fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples || probably for 1 dimensional data
        fxtest_samples = np.ones((samples, testsize,np.array(self.test_y).shape[1]))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)

        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)
        '''
        weight already declared when NN was created with topology defined in it
        '''
        step_w = 0.02  # defines how much variation you need in changes to w
        step_eta = 0.01
        # --------------------- Declare FNN and initialize

        neuralnet = NN(self.topology,self.learnrate,self.train_x,self.train_y)
        w = copy.deepcopy(neuralnet.model.state_dict())

        print ('evaluate Initial w')


        '''
        pred1 = neuralnet.evaluate_proposal(self.train_x[:3])
        print(neuralnet.model.state_dict())
        d = neuralnet.addnoiseandcopy(0,1)
        print(d)
        neuralnet.loadparameters(d)
        print(neuralnet.model.state_dict())
        pred2 = neuralnet.evaluate_proposal(self.train_x[:3])
        print(pred1,'\n',pred2)
        return

        '''

        #print(w,np.array(self.train_x).shape)
        pred_train = neuralnet.evaluate_proposal(self.train_x,w)
        pred_test = neuralnet.evaluate_proposal(self.test_x,w)

        eta = np.log(np.var((pred_train) - np.array(y_train)))
        tau_pro = np.exp(eta)

#        err_nn = np.sum(np.square((pred_train) - np.array(y_train)))/(len(pred_train)) #added by ashray mean square sum
#        print('err_nn is: ',err_nn)
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        #print(pred_train)
        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, neuralnet.getparameters(w), tau_pro)  # takes care of the gradients
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.train_x,self.train_y, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x,self.test_y, w, tau_pro)

        #print(likelihood,' is likelihood of train')
        #print(pred_train)
        #print(pred_train, ' is pred_train')
        naccept = 0
        print ('begin sampling using mcmc random walk')
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        if os.path.exists('mcmcresults') is False:
            os.makedits('mcmcresults')
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):
            #print(i)

            #w_proposal = w + np.random.normal(0, step_w, w_size)
            w_proposal = neuralnet.addnoiseandcopy(0,step_w) # adding gaussian normal distributed noise to all weights and all biases

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.train_x,self.train_y, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x,self.test_y, w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, neuralnet.getparameters(w_proposal),
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            #mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))
            mh_prob = min(0, (diff_likelihood + diff_priorliklihood))
            mh_prob = math.exp(mh_prob)
            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                print(i, ' is the accepted sample')
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                neuralnet.loadparameters(copy.deepcopy(w_proposal))
                #w = neuralnet.getparameters(w = w_proposal)
                eta = eta_pro
                # if i % 100 == 0:
                #     #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                #     print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

                #pos_w[i + 1,] = w_proposal
                pos_w[i+1,] = copy.deepcopy(neuralnet.getparameters(w_proposal)).reshape(-1)
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest

                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

            if i % 100 == 0:
                #print ( likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted')
                print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

        print (naccept, ' num accepted')
        print ((naccept*100) / (samples * 1.0), '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)





def main():
        outres = open('resultspriors.txt', 'w')
        learnRate = 0.1

        #for mackey
        fname = "train_mackey.txt"
        x,y = data_loader(fname)
        #print_data(x,y)
        x,y = shuffledata(x,y)
        train_x= x[:int(len(x)*0.8)]
        test_x=x[int(len(x)*0.8):]
        train_y= y[:int(len(y)*0.8)]
        test_y=y[int(len(y)*0.8):]
        #Input = len(train_x[0][0])
        #Output = len(train_y[0])
        Input = 3
        Output = 1
        #print(traindata)
        Hidden = 5
        topology = [Input, Hidden, Output]
        numSamples = 5000  # need to decide yourself

        mcmc = MCMC(numSamples,learnRate,train_x,train_y,test_x,test_y, topology)  # declare class

        [pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler()
        print ('sucessfully sampled')

        burnin = 0.1 * numSamples  # use post burn in samples

        pos_w = pos_w[int(burnin):, ]
        pos_tau = pos_tau[int(burnin):, ]


        '''
        to plots the histograms of weight destribution
        '''

        mplt.initialiseweights(len(pos_w),len(pos_w[0]))
        for i in range(len(pos_w)):
            mplt.addweightdata(i,pos_w[i])
        mplt.saveplots()

        fx_mu = fx_test.mean(axis=0)
        fx_high = np.percentile(fx_test, 95, axis=0)
        fx_low = np.percentile(fx_test, 5, axis=0)

        fx_mu_tr = fx_train.mean(axis=0)
        fx_high_tr = np.percentile(fx_train, 95, axis=0)
        fx_low_tr = np.percentile(fx_train, 5, axis=0)

        rmse_tr = np.mean(rmse_train[int(burnin):])
        rmsetr_std = np.std(rmse_train[int(burnin):])
        rmse_tes = np.mean(rmse_test[int(burnin):])
        rmsetest_std = np.std(rmse_test[int(burnin):])
        print (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)
        np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

        #ytestdata = testdata[:, input]
        #ytraindata = traindata[:, input]
        ytestdata = test_y
        ytraindata = train_y

        # converting everything to np arrays
        x_test = np.array(x_test)
        fx_low = np.array(fx_low)
        fx_high = np.array(fx_high)
        fx_mu = np.array(fx_mu)
        ytestdata = np.array(ytestdata)
        x_train= np.array(x_train)
        ytraindata= np.array(ytraindata)
        fx_mu_tr = np.array(fx_mu_tr)
        fx_low_tr = np.array(fx_low_tr)
        fx_high_tr = np.array(fx_high_tr)



        plt.plot(x_test, ytestdata, label='actual')
        plt.plot(x_test, fx_mu, label='pred. (mean)')
        plt.plot(x_test, fx_low, label='pred.(5th percen.)')
        plt.plot(x_test, fx_high, label='pred.(95th percen.)')
        #print(np.array(x_test).shape,np.array(fx_low).shape,np.array(fx_high).shape)
        #print(fx_low[:,0],fx_high,x_test)
        plt.fill_between(x_test, fx_low[:,0], fx_high[:,0], facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Test Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestest.png')
        plt.savefig('mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
        plt.clf()
        # -----------------------------------------
        plt.plot(x_train, ytraindata, label='actual')
        plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
        plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
        plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
        plt.fill_between(x_train, fx_low_tr[:,0], fx_high_tr[:,0], facecolor='g', alpha=0.4)
        plt.legend(loc='upper right')

        plt.title("Plot of Train Data vs MCMC Uncertainty ")
        plt.savefig('mcmcresults/mcmcrestrain.png')
        plt.savefig('mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
        plt.clf()

        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        ax.boxplot(pos_w)

        ax.set_xlabel('[W1] [B1] [W2] [B2]')
        ax.set_ylabel('Posterior')

        plt.legend(loc='upper right')

        plt.title("Boxplot of Posterior W (weights and biases)")
        plt.savefig('mcmcresults/w_pos.png')
        plt.savefig('mcmcresults/w_pos.svg', format='svg', dpi=600)

        plt.clf()




if __name__ == "__main__": main()
