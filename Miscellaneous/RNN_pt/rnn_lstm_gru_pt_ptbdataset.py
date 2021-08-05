#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:02:16 2019

@author: ashrey
"""


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


#Initialise and parse inputs
parser=argparse.ArgumentParser(description='PTBayeslands modelling')


parser.add_argument('-n','--net', help='Choose rnn net, "1" for RNN, "2" for GRU, "3" for LSTM', default = 1, dest="net",type=int)
parser.add_argument('-s','--samples', help='Number of samples', default=500, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=4,dest="num_chains",type=int)
parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=3,dest="mt_val",type=int)
parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.1,type=float)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default=0.25,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)
parser.add_argument('-step','--step', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",default=0.05,type=float)
parser.add_argument('-lr','--learn', help='learn rate for langevin gradient', dest="learn_rate",default=0.1,type=float)

args = parser.parse_args()
batch_size = 100
def f(): raise Exception("Found exit()")

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

    print(vocabulary)
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





class Model(nn.Module):

        # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self, topo,lrate,rnn_net = 'LSTM'):
        super(Model, self).__init__()
        # assuming num_dicrections to be 1 for all the cases
        # Defining some parameters
        self.hidden_dim = topo[1]#hidden_dim
        self.n_layers = 1#len(topo)-2 #n_layers
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
        self.vocab_size = topo[3]
        self.embed = nn.Embedding(self.vocab_size,topo[0])
        print(rnn_net, ' is rnn net')


    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))

    def forward(self, x):
        outmain=torch.zeros((len(x),self.topo[2]))
        for i,sample in enumerate(x):
            sample = torch.FloatTensor(sample)
            sample = self.embed(sample)
            # sample = sample.view(sample.shape[0],1,self.topo[0])
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



class ptReplica(multiprocessing.Process):

    def __init__(self, use_langevin_gradients, learn_rate,  w,  minlim_param, maxlim_param, samples,trainx,trainy,testx,testy, topology, burn_in, temperature, swap_interval, langevin_prob, path, parameter_queue, main_process,event,rnn_net):
        #MULTIPROCESSING VARIABLES
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        self.rnn = Model(topology,learn_rate,rnn_net=rnn_net)
        self.temperature = temperature
        self.adapttemp = temperature
        self.swap_interval = swap_interval
        self.path = path
        self.burn_in = burn_in
        #FNN CHAIN VARIABLES (MCMC)
        self.samples = samples
        self.topology = topology
        self.train_x = trainx
        self.train_y = trainy
        self.test_x = testx
        self.test_y = testy
        self.w = w
        self.minY = np.zeros((1,1))
        self.maxY = np.zeros((1,1))
        #self.rnn
        self.minlim_param = minlim_param
        self.maxlim_param = maxlim_param
        self.use_langevin_gradients = use_langevin_gradients
        self.sgd_depth = 1 # always should be 1
        self.learn_rate = learn_rate
        self.l_prob = langevin_prob  # can be evaluated for diff problems - if data too large keep this low value since the gradients cost comp time

    def rmse(self, pred, actual):
        return np.sqrt(((pred-actual)**2).mean())

    '''def likelihood_func(self, fnn, data, w):
        y = data[:, self.topology[0]]
        fx  = fnn.evaluate_proposal(data,w)
        rmse = self.rmse(fx,y)
        z = np.zeros((data.shape[0],self.topology[2]))
        lhood = 0
        for i in range(data.shape[0]):
            for j in range(self.topology[2]):
                if j == y[i]:
                    z[i,j] = 1
                lhood += z[i,j]*np.log(prob[i,j])
        return [lhood/self.temperature, fx, rmse]'''

    def likelihood_func(self, rnn,x,y, w, tau_sq):
        fx = rnn.evaluate_proposal(x,w)
        fx = np.array(fx)[:,0]
        y = np.array(y).reshape(fx.shape)#data[:, self.topology[0]]
        rmse = self.rmse(fx, y)
        loss = np.sum(-0.5*np.log(2*math.pi*tau_sq) - 0.5*np.square(y-fx)/tau_sq)
        return [np.sum(loss)/self.adapttemp, fx, rmse]

    '''def prior_likelihood(self, sigma_squared, nu_1, nu_2, w):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + self.topology[2]+h*self.topology[2]) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2
        return log_loss'''

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq,rnn):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((len(rnn.getparameters(w))) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(rnn.getparameters(w))))
        log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def run(self):
        #INITIALISING FOR RNN
        samples = self.samples
        x_test = self.test_x
        x_train = self.train_x
        y_train = self.train_y
        batch_save = 10  # batch to append to file
        learn_rate = self.learn_rate
        rnn = self.rnn#Model(self.topology,learn_rate,rnn_net = self.rnn_net)
        w_size = sum(p.numel() for p in rnn.parameters())
        pos_w = np.ones((samples, w_size)) #Posterior for all weights
        rmse_train  = np.zeros(samples)
        rmse_test = np.zeros(samples)
        acc_train = np.zeros(samples)
        acc_test = np.zeros(samples)
        #Random Initialisation of weights
        w = copy.deepcopy(rnn.state_dict())
        eta = 0 #Junk variable
        step_w = 0.025
        step_eta = 0.2

        #Declare RNN
        pred_train  = rnn.evaluate_proposal(x_train,w) #
        pred_test  = rnn.evaluate_proposal(x_test, w) #
        eta = np.log(np.var(pred_train - np.array(y_train)))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        np.fill_diagonal(sigma_diagmat, step_w)

        #delta_likelihood = 0.5 # an arbitrary position
        prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro,rnn)  # takes care of the gradients
        [likelihood, pred_train, rmsetrain] = self.likelihood_func(rnn, self.train_x,self.train_y, w, tau_pro)
        [_, pred_test, rmsetest] = self.likelihood_func(rnn, self.test_x,self.test_y, w, tau_pro)
        prop_list = np.zeros((samples,w_size))
        likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
        likeh_list[0,:] = [-100, -100] # to avoid prob in calc of 5th and 95th percentile later
        surg_likeh_list = np.zeros((samples,2))
        accept_list = np.zeros(samples)
        num_accepted = 0
        langevin_count = 0
        pt_samples = samples * 0.6 # this means that PT in canonical form with adaptive temp will work till pt  samples are reached
        init_count = 0
        self.event.clear()
        for i in range(samples-1):  # Begin sampling --------------------------------------------------------------------------
            timer1 = time.time()
            if i < pt_samples:
                self.adapttemp =  self.temperature #* ratio  #
            if i == pt_samples and init_count ==0: # move to MCMC canonical
                self.adapttemp = 1
                [likelihood, pred_train, rmsetrain ] = self.likelihood_func(rnn, self.train_x,self.train_y, w, tau_pro)
                [_, pred_test, rmsetest ] = self.likelihood_func(rnn, self.test_x,self.test_y, w, tau_pro)
                init_count = 1


            lx = np.random.uniform(0,1,1)

            if (self.use_langevin_gradients is True) and (lx< self.l_prob):
                w_gd = rnn.langevin_gradient(self.train_x,self.train_y, copy.deepcopy(w)) # Eq 8
                w_proposal = rnn.addnoiseandcopy(w_gd,0,step_w) #np.random.normal(w_gd, step_w, w_size) # Eq 7
                w_prop_gd = rnn.langevin_gradient(self.train_x,self.train_y, copy.deepcopy(w_proposal))
                #first = np.log(multivariate_normal.pdf(w , w_prop_gd , sigma_diagmat))
                #second = np.log(multivariate_normal.pdf(w_proposal , w_gd , sigma_diagmat)) # this gives numerical instability - hence we give a simple implementation next that takes out log
                wc_delta = (rnn.getparameters(w)- rnn.getparameters(w_prop_gd))
                wp_delta = (rnn.getparameters(w_proposal) - rnn.getparameters(w_gd))
                sigma_sq = step_w
                first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq
                diff_prop =  first - second
                diff_prop =  diff_prop/self.adapttemp
                langevin_count = langevin_count + 1
            else:
                diff_prop = 0
                w_proposal = rnn.addnoiseandcopy(w,0,step_w) #np.random.normal(w, step_w, w_size)
            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)
            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(rnn, self.train_x,self.train_y, w_proposal,tau_pro)
            [_, pred_test, rmsetest] = self.likelihood_func(rnn, self.test_x,self.test_y, w_proposal,tau_pro)
            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,tau_pro,rnn)  # takes care of the gradients
            diff_prior = prior_prop - prior_current
            diff_likelihood = likelihood_proposal - likelihood
            surg_likeh_list[i+1,0] = likelihood_proposal * self.adapttemp
            #surg_likeh_list[i+1,1] = np.nan
            try:
                mh_prob = min(0, (diff_likelihood+diff_prior+ diff_prop))
                mh_prob = math.exp(mh_prob)
            except OverflowError as e:
                mh_prob = 1
            accept_list[i+1] = num_accepted
            if (i % batch_save+1) == 0: # just for saving posterior to file - work on this later
                x = 0
            u = random.uniform(0, 1)
            prop_list[i+1,] = rnn.getparameters(w_proposal).reshape(-1)
            likeh_list[i+1,0] = likelihood_proposal
            if u < mh_prob:
                num_accepted  =  num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)
                eta = eta_pro
                acc_train[i+1,] = 0
                acc_test[i+1,] = 0
                #print(i,rmsetrain,rmsetest,'accepted')
                pos_w[i+ 1,] = rnn.getparameters(w_proposal).reshape(-1)
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest
            else:
                pos_w[i+1,] = pos_w[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]
                acc_train[i+1,] = acc_train[i,]
                acc_test[i+1,] = acc_test[i,]

            if ((i+1) % self.swap_interval == 0 and i != 0 ):
                w_size = rnn.getparameters(w).reshape(-1).shape[0]
                param = np.concatenate([rnn.getparameters(w).reshape(-1), np.asarray([eta]).reshape(1), np.asarray([likelihood*self.temperature]),np.asarray([self.temperature])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()
                param1 = self.parameter_queue.get()
                w1 = param1[0:w_size]
                eta = param1[w_size]
                w1_dict = rnn.dictfromlist(w1)
                rnn.loadparameters(w1_dict)

        param = np.concatenate([rnn.getparameters(w).reshape(-1), np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.adapttemp]),np.asarray([i])])
        self.parameter_queue.put(param)
        self.signal_main.set()
        print ((num_accepted*100 / (samples * 1.0)), '% was accepted')
        accept_ratio = num_accepted / (samples * 1.0) * 100
        print ((langevin_count*100 / (samples * 1.0)), '% was Lsnngrevin ')
        langevin_ratio = langevin_count / (samples * 1.0) * 100
        file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,pos_w )
        file_name = self.path+'/predictions/rmse_test_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.8f')
        file_name = self.path+'/predictions/rmse_train_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.8f')
        file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, acc_test, fmt='%1.2f')
        file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, acc_train, fmt='%1.2f')
        file_name = self.path+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,likeh_list, fmt='%1.4f')
        file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
        np.savetxt(file_name, [accept_ratio], fmt='%1.4f')
        file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, accept_list, fmt='%1.4f')
        print('exiting this thread')

class ParallelTempering:

    def __init__(self,  use_langevin_gradients, learn_rate, train_x,train_y,test_x,test_y, topology, num_chains, maxtemp, NumSample, swap_interval, langevin_prob, path,rnn_net = 'RNN'):
        #FNN Chain variables
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.topology = topology
        self.rnn = Model(self.topology,learn_rate,rnn_net = rnn_net)
        self.num_param = sum(p.numel() for p in self.rnn.parameters())#(topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2] + (topology[1] * topology[1])
        #Parallel Tempering variables
        self.swap_interval = swap_interval
        self.path = path
        self.maxtemp = maxtemp
        self.rnn_net = rnn_net
        self.langevin_prob = langevin_prob
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = int(NumSample/self.num_chains)
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        # create queues for transfer of parameters between process chain
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        self.all_param = None
        self.geometric = True # True (geometric)  False (Linear)
        self.minlim_param = 0.0
        self.maxlim_param = 0.0
        self.minY = np.zeros((1,1))
        self.maxY = np.ones((1,1))
        self.model_signature = 0.0
        self.learn_rate = learn_rate
        self.use_langevin_gradients = use_langevin_gradients

    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.

        """

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')


        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                        2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                        2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                        1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                        1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                        1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                        1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                        1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                        1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                        1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                        1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                        1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                        1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                        1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                        1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                        1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                        1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                        1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                        1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                        1.26579, 1.26424, 1.26271, 1.26121,
                        1.25973])


        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):
        # #Linear Spacing
        # temp = 2
        # for i in range(0,self.num_chains):
        # 	self.temperatures.append(temp)
        # 	temp += 2.5 #(self.maxtemp/self.num_chains)
        # 	print (self.temperatures[i])
        #Geometric Spacing

        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)
            for i in range(0, self.num_chains):
                self.temperatures.append(np.inf if betas[i] == 0 else 1.0/betas[i])
                #print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp /self.num_chains)
            temp = 1
            for i in range(0, self.num_chains):
                self.temperatures.append(temp)
                temp += tmpr_rate
                #print(self.temperatures[i])


    def initialize_chains(self,  burn_in):
        self.burn_in = burn_in
        self.assign_temperatures()
        self.minlim_param = np.repeat([-100] , self.num_param)  # priors for nn weights
        self.maxlim_param = np.repeat([100] , self.num_param)
        for i in range(0, self.num_chains):
            w = np.random.randn(self.num_param)
            self.chains.append(ptReplica(self.use_langevin_gradients, self.learn_rate, w,  self.minlim_param, self.maxlim_param, self.NumSamples, self.train_x,self.train_y,self.test_x,self.test_y,self.topology,self.burn_in,self.temperatures[i],self.swap_interval, self.langevin_prob, self.path,self.parameter_queue[i],self.wait_chain[i],self.event[i],self.rnn_net))


    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        param1 = parameter_queue_1.get()
        param2 = parameter_queue_2.get()
        w1 = param1[0:self.num_param]
        eta1 = param1[self.num_param]
        lhood1 = param1[self.num_param+1]
        T1 = param1[self.num_param+2]
        w2 = param2[0:self.num_param]
        eta2 = param2[self.num_param]
        lhood2 = param2[self.num_param+1]
        T2 = param2[self.num_param+2]
        #print('yo')
        #SWAPPING PROBABILITIES
        try:
            swap_proposal =  min(1,0.5*np.exp(lhood2 - lhood1))
        except:
            swap_proposal = 1
        u = np.random.uniform(0,1)
        if u < swap_proposal:
            swapped = True
            self.total_swap_proposals += 1
            self.num_swap += 1
            param_temp = param1
            param1 = param2
            param2 = param_temp
        else:
            swapped = False
            self.total_swap_proposals+=1
        return param1, param2, swapped


    def run_chains(self):
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1)
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))
        lhood = np.zeros(self.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples-1
        number_exchange = np.zeros(self.num_chains)
        filen = open(self.path + '/num_exchange.txt', 'a')
        #RUN MCMC CHAINS
        for l in range(0,self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0,self.num_chains):
            self.wait_chain[j].clear()
            self.event[j].clear()
            self.chains[j].start()
        #SWAP PROCEDURE
        '''
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        '''
        #SWAP PROCEDURE
        swaps_appected_main =0
        total_swaps_main =0
        for i in range(int(self.NumSamples/self.swap_interval)):
        #while(True):
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count+=1
                    self.wait_chain[index].set()
                    print(str(self.chains[index].temperature) +" Dead"+str(index))
            if count == self.num_chains:
                break
            #if count == 0:
            #break
            print(count,' is count')
            print(datetime.datetime.now())
            timeout_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait()
                if flag:
                    print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                print("Skipping the swap!")
                continue
            
            print("Event occured")
            for index in range(0,self.num_chains-1):
                param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                self.parameter_queue[index].put(param_1)
                self.parameter_queue[index+1].put(param_2)
                if index == 0:
                    if swapped:
                        swaps_appected_main += 1
                    total_swaps_main += 1
            #blabla = input("whazzzuppp")
            for index in range (self.num_chains):
                self.wait_chain[index].clear()
                self.event[index].set()


        print("Joining processes")

        #JOIN THEM TO MAIN PROCESS
        for index in range(0,self.num_chains):
            print(index, ' waiting to join')
            self.chains[index].join()
        print('now waiting for chain queue')
        self.chain_queue.join()
        pos_w, fx_train, fx_test,   rmse_train, rmse_test, acc_train, acc_test,  likelihood_vec ,   accept_vec, accept  = self.show_results()
        print("NUMBER OF SWAPS =", self.num_swap)
        print('total swap proposal',self.total_swap_proposals)
        print('num samples', self.NumSamples)
        print('swap interval',self.swap_interval)
        swap_perc = self.num_swap*100 /self.total_swap_proposals
        return pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test,   likelihood_vec , swap_perc,    accept_vec, accept
        # pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test,   likelihood_rep , swap_perc,accept_vec, accept = pt.run_chains()


    def show_results(self):

        burnin = int(self.NumSamples*self.burn_in)

        likelihood_rep = np.zeros((self.num_chains, self.NumSamples - 1, 2)) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        accept_percent = np.zeros((self.num_chains, 1))
        accept_list = np.zeros((self.num_chains, self.NumSamples ))

        pos_w = np.zeros((self.num_chains,self.NumSamples - burnin, self.num_param))

        fx_train_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.train_x.shape[0]))
        rmse_train = np.zeros((self.num_chains,self.NumSamples - burnin))
        acc_train = np.zeros((self.num_chains,self.NumSamples - burnin))
        fx_test_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.test_x.shape[0]))
        rmse_test = np.zeros((self.num_chains,self.NumSamples - burnin))
        acc_test = np.zeros((self.num_chains,self.NumSamples - burnin))



        for i in range(self.num_chains):
            file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            pos_w[i,:,:] = dat[burnin:,:]

            file_name = self.path + '/posterior/pos_likelihood/'+'chain_' + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            likelihood_rep[i, :] = dat[1:]


            file_name = self.path + '/posterior/accept_list/' + 'chain_'  + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            accept_list[i, :] = dat


            #file_name = self.path+'/predictions/fxtrain_samples_chain_'+ str(self.temperatures[i])+ '.txt'
            #dat = np.loadtxt(file_name)
            #fx_train_all[i,:,:] = dat[burnin:,:]

            #file_name = self.path+'/predictions/fxtest_samples_chain_'+ str(self.temperatures[i])+ '.txt'
            #dat = np.loadtxt(file_name)
            #fx_test_all[i,:,:] = dat[burnin:,:]

            file_name = self.path+'/predictions/rmse_test_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            rmse_test[i,:] = dat[burnin:]

            file_name = self.path+'/predictions/rmse_train_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            rmse_train[i,:] = dat[burnin:]

            file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            acc_test[i,:] = dat[burnin:]

            file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            acc_train[i,:] = dat[burnin:]


        posterior = pos_w.transpose(2,0,1).reshape(self.num_param,-1)

        fx_train = fx_train_all.transpose(2,0,1).reshape(self.train_x.shape[0],-1)  # need to comment this if need to save memory
        fx_test = fx_test_all.transpose(2,0,1).reshape(self.test_x.shape[0],-1)

        #fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0]) # konarks version


        likelihood_vec = likelihood_rep.transpose(2,0,1).reshape(2,-1)

        rmse_train = rmse_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        acc_train = acc_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        rmse_test = rmse_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        acc_test = acc_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)


        accept_vec  = accept_list

        #print(accept_vec)







        accept = np.sum(accept_percent)/self.num_chains

        #np.savetxt(self.path + '/pos_param.txt', posterior.T)  # tcoment to save space

        np.savetxt(self.path + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

        np.savetxt(self.path + '/accept_list.txt', accept_list, fmt='%1.2f')

        np.savetxt(self.path + '/acceptpercent.txt', [accept], fmt='%1.2f')


        return posterior, fx_train_all, fx_test_all,   rmse_train, rmse_test,  acc_train, acc_test,  likelihood_vec.T, accept_vec , accept

    def make_directory (self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)



def main():


    networks = ['RNN','GRU','LSTM']
    net = networks[args.net-1]
    networks = ['LSTM']

    for net in networks:
        for j in range(4, 5) :
            # print(j, ' out of 15','\n\n\n\n\n\n\n')
            i = j//2
            problem =	-100

    
            # if problem ==	4:
            #     #traindata = np.loadtxt("Data_OneStepAhead/Lorenz/train.txt")
            #     #testdata	= np.loadtxt("Data_OneStepAhead/Lorenz/test.txt")  #
            #     name	= "Lorenz"
            #     x,y = load_horizontal("Data_OneStepAhead/Lorenz/train.txt")
            #     train_x= x#x[:int(len(x)*0.8)]
            #     train_y= y#y[:int(len(y)*0.8)]
            #     Input = len(train_x[0][0])
            #     Output = len(train_y[0])
            #     test_x,test_y =load_horizontal("Data_OneStepAhead/Lorenz/test.txt")
    
            name = "ptb"
            train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
            num_steps = 30
            train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,skip_step=num_steps)
            test_data_generator = KerasBatchGenerator(test_data, num_steps, batch_size, vocabulary,skip_step=num_steps)
            trainData = list(train_data_generator.generate())
            testData = list(test_data_generator.generate())
            print(len(trainData), ' is the length of dataset')
            '''x,y = trainData[0]
            print(x.shape,y.shape)'''
            train_x=[]
            train_y=[]
            test_x = []
            test_y = []
            for iter1,iter2 in zip(trainData,testData):
                a,b = iter1
                c,d = iter2
                train_x.append(a)
                train_y.append(b)
                test_x.append(c)
                test_y.append(d)
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            test_x = np.array(test_x)
            test_y = np.array(test_y)

            ###############################
            #THESE ARE THE HYPERPARAMETERS#
            ###############################


            Input = vocabulary
            Output = vocabulary
            Hidden = 100
            #ip = 4 #input
            #output = 1
            topology = [Input, Hidden, Output,vocabulary]
            NumSample = args.samples
            #NumSample = 500
    
    
            ###############################
            #THESE ARE THE HYPERPARAMETERS#
            ###############################
            # topology = [Input, Hidden, Output]
    
            netw = topology
    
            #print(traindata)
    
    
            #y_test =  testdata[:,netw[0]]
            #y_train =  traindata[:,netw[0]]
    
    
    
            maxtemp = args.mt_val
    
            #swap_ratio =  0.04
    
            swap_ratio = args.swap_ratio
    
            num_chains =  args.num_chains
    
            swap_interval = int(swap_ratio * NumSample/num_chains)    # int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours. note if swap is more than Num_samples, its off
            burn_in = args.burn_in
    
    
            learn_rate = args.learn_rate  # in case langevin gradients are used. Can select other values, we found small value is ok.
    
            langevn = ""
            if j%2 == 0:
                use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
                langevn = "T"
            else:
                use_langevin_gradients = False
                langevn = "F"
                pass # we dont want to execute this.
    
    
    
            problemfolder = os.getcwd()+'/Res_LG-Lprob_'+net+'/'  #'/home/rohit/Desktop/PT/Res_LG-Lprob/'  # change this to your directory for results output - produces large datasets
    
            problemfolder_db = 'Res_LG-Lprob_'+net+'/'  # save main results
    
    
    
    
            filename = ""
            run_nb = 0
            while os.path.exists( problemfolder+name+langevn+'_%s' % (run_nb)):
                run_nb += 1
            if not os.path.exists( problemfolder+name+langevn+'_%s' % (run_nb)):
                os.makedirs(  problemfolder+name+langevn+'_%s' % (run_nb))
                path = (problemfolder+ name+langevn+'_%s' % (run_nb))
    
            filename = ""
            run_nb = 0
            while os.path.exists( problemfolder_db+name+langevn+'_%s' % (run_nb)):
                run_nb += 1
            if not os.path.exists( problemfolder_db+name+langevn+'_%s' % (run_nb)):
                os.makedirs(  problemfolder_db+name+langevn+'_%s' % (run_nb))
                path_db = (problemfolder_db+ name+langevn+'_%s' % (run_nb))
    
    
    
            resultingfile = open( path+'/master_result_file.txt','a+')
    
            resultingfile_db = open( path_db+'/master_result_file.txt','a+')
    
            timer = time.time()
    
            langevin_prob = 1/10
    
    
    
            pt = ParallelTempering( use_langevin_gradients,  learn_rate,  train_x,train_y,test_x,test_y, topology, num_chains, maxtemp, NumSample, swap_interval, langevin_prob, path,rnn_net = net)
    
            directories = [  path+'/predictions/', path+'/posterior', path+'/results', path+'/surrogate', path+'/surrogate/learnsurrogate_data', path+'/posterior/pos_w',  path+'/posterior/pos_likelihood',path+'/posterior/surg_likelihood',path+'/posterior/accept_list'  ]
    
            for d in directories:
                pt.make_directory((filename)+ d)
    
    
    
            pt.initialize_chains(  burn_in)
    
    
            pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test,   likelihood_rep , swap_perc,    accept_vec, accept = pt.run_chains()
    
            list_end = accept_vec.shape[1]
            #print(accept_vec.shape)
            #print(accept_vec)
            accept_ratio = accept_vec[:,  list_end-1:list_end]/list_end
            accept_per = np.mean(accept_ratio) * 100
    
            print(accept_per, ' accept_per')
    
    
    
    
    
            timer2 = time.time()
    
            timetotal = (timer2 - timer) /60
            print ((timetotal), 'min taken')
    
            #PLOTS
    
            '''acc_tr = np.mean(acc_train [:])
            acctr_std = np.std(acc_train[:])
            acctr_max = np.amax(acc_train[:])
    
            acc_tes = np.mean(acc_test[:])
            acctest_std = np.std(acc_test[:])
            acctes_max = np.amax(acc_test[:])'''
    
    
    
            rmse_tr = np.mean(rmse_train[:])
            rmsetr_std = np.std(rmse_train[:])
            rmsetr_max = np.amin(rmse_train[:])
    
            rmse_tes = np.mean(rmse_test[:])
            rmsetest_std = np.std(rmse_test[:])
            rmsetes_max = np.amin(rmse_test[:])
    
            outres = open(path+'/result.txt', "a+")
            outres_db = open(path_db+'/result.txt', "a+")
    
            resultingfile = open(problemfolder+'/master_result_file.txt','a+')
            resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+')
    
            xv = name+langevn+'_'+ str(run_nb)
    
            allres =  np.asarray([ problem, NumSample, maxtemp, swap_interval, langevin_prob, learn_rate,  rmse_tr, rmsetr_std, rmsetr_max, rmse_tes, rmsetest_std, rmsetes_max, swap_perc, accept_per, timetotal])
    
            np.savetxt(outres_db,  allres   , fmt='%1.4f', newline=' '  )
            np.savetxt(resultingfile_db,   allres   , fmt='%1.4f',  newline=' ' )
            np.savetxt(resultingfile_db, [xv]   ,  fmt="%s", newline=' \n' )
    
    
            np.savetxt(outres,  allres   , fmt='%1.4f', newline=' '  )
            np.savetxt(resultingfile,   allres   , fmt='%1.4f',  newline=' ' )
            np.savetxt(resultingfile, [xv]   ,  fmt="%s", newline=' \n' )
    
            x = np.linspace(0, rmse_train.shape[0] , num=rmse_train.shape[0])
    
    
    
            '''plt.plot(x, rmse_train, '.',   label='Test')
            plt.plot(x, rmse_test,  '.', label='Train')
            plt.legend(loc='upper right')
    
            plt.xlabel('Samples', fontsize=12)
            plt.ylabel('RMSE', fontsize=12)
    
            plt.savefig(path+'/rmse_samples.png')
            plt.clf()	'''
    
            plt.plot(  rmse_train, '.',  label='Test')
            plt.plot(  rmse_test,  '.',  label='Train')
            plt.legend(loc='upper right')
    
            plt.xlabel('Samples', fontsize=12)
            plt.ylabel('RMSE', fontsize=12)
    
            plt.savefig(path_db+'/rmse_samples.pdf')
            plt.clf()
    
            plt.plot( rmse_train, '.',   label='Test')
            plt.plot( rmse_test, '.',   label='Train')
            plt.legend(loc='upper right')
    
            plt.xlabel('Samples', fontsize=12)
            plt.ylabel('RMSE', fontsize=12)
    
            plt.savefig(path+'/rmse_samples.pdf')
            plt.clf()
    
    
    
    
            likelihood = likelihood_rep[:,0] # just plot proposed likelihood
            likelihood = np.asarray(np.split(likelihood, num_chains))
    
    
    
    
        # Plots
            plt.plot(likelihood.T)
    
            plt.xlabel('Samples', fontsize=12)
            plt.ylabel(' Log-Likelihood', fontsize=12)
    
            plt.savefig(path+'/likelihood.png')
            plt.clf()
    
            plt.plot(likelihood.T)
    
            plt.xlabel('Samples', fontsize=12)
            plt.ylabel(' Log-Likelihood', fontsize=12)
    
            plt.savefig(path_db+'/likelihood.png')
            plt.clf()
    
    
            plt.plot(accept_vec.T )
            plt.xlabel('Samples', fontsize=12)
            plt.ylabel(' Number accepted proposals', fontsize=12)
    
            plt.savefig(path_db+'/accept.png')
    
    
            plt.clf()
    
    
            #mpl_fig = plt.figure()
            #ax = mpl_fig.add_subplot(111)
    
            # ax.boxplot(pos_w)
    
            # ax.set_xlabel('[W1] [B1] [W2] [B2]')
            # ax.set_ylabel('Posterior')
    
            # plt.legend(loc='upper right')
    
            # plt.title("Boxplot of Posterior W (weights and biases)")
            # plt.savefig(path+'/w_pos.png')
            # plt.savefig(path+'/w_pos.svg', format='svg', dpi=600)
    
            # plt.clf()
            #dir()
            gc.collect()
            outres.close()
            resultingfile.close()
            resultingfile_db.close()
            outres_db.close()
    

if __name__ == "__main__": main()


