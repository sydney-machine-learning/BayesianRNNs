#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:32:34 2019

@author: ashray

python 3.6 --  working great

RNN, GRU, LSTM
with MCMC in torch.

you can also speciffy optional arguments when running file
python rnn_gru_lstm_mcmc_torch.py -n=1 -s=1000
or
python rnn_gru_lstm_mcmc_torch.py
to run with default configurations
"""

#  %reset
#  %reset -sf


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

mplt = mcmcplt.Mcmcplot()

weightdecay = 0.01


#Initialise and parse inputs
parser=argparse.ArgumentParser(description='RNN-GRU and LSTM MCMC implementation in pytorch')


parser.add_argument('-n','--net', help='Choose rnn net, "1" for RNN, "2" for GRU, "3" for LSTM', default = 1, dest="net",type=int)
parser.add_argument('-s','--samples', help='Number of samples', default=50000, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=10,dest="num_chains",type=int)
#parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=10,dest="mt_val",type=int)
#parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.02,type=float)
#parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default=0.25,type=float)
#parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)
#parser.add_argument('-step','--step', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",default=0.05,type=float)

args = parser.parse_args()
def f(): raise Exception("Found exit()")
class Model(nn.Module):
        # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self, topo,lrate,rnn_net = 'RNN'):
        super(Model, self).__init__()
        # assuming num_dicrections to be 1 for all the cases
        # Defining some parameters
        self.hidden_dim = topo[1]#hidden_dim
        self.n_layers = 1#len(topo)-2 #n_layers
        self.batch_size = 1
        #Defining the layers
        # RNN Layer
        #self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        #batch_size = 1
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
        print(rnn_net, ' is rnn net')
    
    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))

    def forward(self, x):
        batch_size = self.batch_size #x.size(0)
        outmain=torch.zeros((len(x),self.topo[2]))
        for i,sample in enumerate(x):
            # Passing in the input and hidden state into the model and obtaining outputs
            sample = torch.FloatTensor(sample)
            sample = sample.view(sample.shape[0],1,self.topo[0])
            hidden = copy.deepcopy(self.hidden)
            out, h1 = self.rnn(sample, hidden)
            out = self.fc(out[-1])
            out = self.sigmoid(out)
            outmain[i,] = copy.deepcopy(out.detach())
        return copy.deepcopy(outmain)

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden        # Create a brnn -- assuming h0 size = 5 i.e. hidden layer has 5 nuerons


    def evaluate_proposal(self,x,w=None):
        if w is None:
            y_pred = self.forward(x)
            return copy.deepcopy(y_pred.detach().numpy())
        else:
            self.loadparameters(w)
            y_pred = self.forward(x)
            return copy.deepcopy(np.array(y_pred.detach()))


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
            dic = self.state_dict()
        else:
            dic = copy.deepcopy(w)
        for name in dic.keys():
            l=np.concatenate((l.reshape(-1,1),(copy.deepcopy(dic[name])).numpy().reshape(-1,1)))
        l = l[2:]
        return l

    # input a dictionary of same dimensions
    def loadparameters(self,param):
        self.load_state_dict(param)

    def addnoiseandcopy(self,w,mea,std_dev):
        dict = {}
        #self.load_state_dict(w)
        for name in (w.keys()):
            dict[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean = mea, std = std_dev)
        return dict

class MCMC:
    def __init__(self, samples, learnrate, train_x, train_y,test_x,test_y, topology,rnn_net = 'RNN'):
        self.samples = samples  # max epocs
        self.topology = topology  # NN topology [input, hidden, output]
        self.train_x = train_x#
        self.test_x = test_x
        self.train_y=train_y
        self.test_y=test_y
        self.learnrate = learnrate
        self.rnn_net = rnn_net
        # ----------------

    def rmse(self, predictions, targets):
        predictions = (predictions)
        targets=np.array(targets)
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, x,y, w, tausq):
        fx = neuralnet.evaluate_proposal(x, w)
        fx = np.array(fx)
        y = np.array(y).reshape(fx.shape)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [np.sum(loss), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w_list, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((len(w_list)) / 2) * np.log(sigma_squared)
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
        fxtrain_samples = np.ones((samples, trainsize,int(np.array(y_train).shape[1])))  # fx of train data over all samples
        fxtest_samples = np.ones((samples, testsize,np.array(self.test_y).shape[1]))  # fx of test data over all samples
        rmse_train = np.zeros(samples)
        rmse_test = np.zeros(samples)
        step_w = 0.02  # defines how much variation you need in changes to w
        step_eta = 0.01
        # --------------------- Declare FNN and initialize

        neuralnet = Model(self.topology,self.learnrate,rnn_net = self.rnn_net)
        w = copy.deepcopy(neuralnet.state_dict())
        # weight defined when neural net created
        # no need for w_size
        w_size = len(neuralnet.getparameters(w))
        print ('evaluate Initial w')
        pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        pos_tau = np.ones((samples, 1))
        pred_train = neuralnet.evaluate_proposal(self.train_x,w)
        pred_test = neuralnet.evaluate_proposal(self.test_x,w)
        eta = np.log(np.var((pred_train) - np.array(y_train)))
        tau_pro = np.exp(eta)
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, neuralnet.getparameters(w), tau_pro)  # takes care of the gradients
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.train_x,self.train_y, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.test_x,self.test_y, w, tau_pro)
        naccept = 0
        print ('begin sampling using mcmc random walk')
        print(np.array(x_train).shape,np.array(y_train).shape,np.array(pred_train).shape)
        plt.plot(x_train, np.array(y_train)[:,:,0])
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        if os.path.exists(self.rnn_net+'mcmcresults') is False:
            os.makedirs(self.rnn_net+'mcmcresults')
        plt.savefig(self.rnn_net+'mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, np.array(y_train)[:,:,0])

        for i in range(samples - 1):
            w_proposal = neuralnet.addnoiseandcopy(w,0,step_w) # adding gaussian normal distributed noise to all weights and all biases
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
                print(i,rmsetrain,rmsetest ,' accepted sample and train and test RMSE')
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = copy.deepcopy(w_proposal)
                eta = eta_pro
                pos_w[i+1,] = copy.deepcopy(neuralnet.getparameters(w_proposal)).reshape(-1)
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train.reshape(fxtrain_samples.shape[1],1)
                fxtest_samples[i + 1,] = pred_test.reshape(fxtest_samples.shape[1],1)
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

            if i % 100 == 0:
                print ('Sample:',i, 'RMSE train:', rmsetrain, 'RMSE test:',rmsetest)

        print (naccept, ' num accepted')
        print ((naccept*100) / (samples * 1.0), '% was accepted')
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig(self.rnn_net+'mcmcresults/proposals.png')
        plt.savefig(self.rnn_net+'mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)
def data_loader(filename):
    f=open(filename,'r')
    x=[[[]]]
    count=0
    y=[[[]]]
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
        a=[[]]
        ya=[[]]
        for i in range(0,t):
            temp=f.readline().split(' ')
            b=0.0
            for j in range(0,len(temp)):
                b=[float(temp[j])]
            a.append(b)
        del a[0]
        x.append(a)
        temp=f.readline().split(' ')
        #print(temp)
        for j in range(0,len(temp)):
            if temp[j] != "\n":
                ya.append([float(temp[j])])
        del ya[0]
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
        Input = 1
        Output = 1
        Hidden = 5
        topology = [Input, Hidden, Output]
        numSamples = args.samples
        networks = ['RNN','GRU','LSTM']
        net1 = networks[args.net-1]
        for net in [net1]:
            mcmc = MCMC(numSamples,learnRate,train_x,train_y,test_x,test_y, topology,rnn_net = net)  # declare class
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
            ytestdata = np.array(ytestdata)[:,:,0]
            x_train= np.array(x_train)
            ytraindata= np.array(ytraindata)[:,:,0]
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
            plt.savefig(net+'mcmcresults/mcmcrestest.png')
            plt.savefig(net+'mcmcresults/mcmcrestest.svg', format='svg', dpi=600)
            plt.clf()
            # -----------------------------------------
            plt.plot(x_train, ytraindata, label='actual')
            plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
            plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
            plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
            plt.fill_between(x_train, fx_low_tr[:,0], fx_high_tr[:,0], facecolor='g', alpha=0.4)
            plt.legend(loc='upper right')

            plt.title("Plot of Train Data vs MCMC Uncertainty ")
            plt.savefig(net+'mcmcresults/mcmcrestrain.png')
            plt.savefig(net+'mcmcresults/mcmcrestrain.svg', format='svg', dpi=600)
            plt.clf()

            mpl_fig = plt.figure()
            ax = mpl_fig.add_subplot(111)

            ax.boxplot(pos_w)

            ax.set_xlabel('[W1] [B1] [W2] [B2]')
            ax.set_ylabel('Posterior')

            plt.legend(loc='upper right')

            plt.title("Boxplot of Posterior W (weights and biases)")
            plt.savefig(net+'mcmcresults/w_pos.png')
            plt.savefig(net+'mcmcresults/w_pos.svg', format='svg', dpi=600)

            plt.clf()

if __name__ == "__main__": main()
