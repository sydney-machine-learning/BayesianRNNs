import multiprocessing

from torch import device
from model import Model
import copy
import numpy as np
import math
import random
import time
import torch

class ptReplica(multiprocessing.Process):
    def __init__(self, use_langevin_gradients, learn_rate,  w,  minlim_param, maxlim_param, samples,trainx,trainy,testx,testy, topology, burn_in, temperature, swap_interval, langevin_prob, path, parameter_queue, main_process,event,rnn_net):
        #MULTIPROCESSING VARIABLES
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        self.rnn = Model(topology,learn_rate,rnn_net=rnn_net)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.rnn.to(device)
        # print(device," device of rnn")
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
        np.fill_diagonal(sigma_diagmat, step_w**2)
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
        i=0
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
                sigma_sq = step_w ** 2
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
                # mh_prob = math.exp(mh_prob)
            except OverflowError as e:
                mh_prob = 1
            accept_list[i+1] = num_accepted
            if (i % batch_save+1) == 0: # just for saving posterior to file - work on this later
                x = 0
            u = np.log(random.uniform(0, 1))
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
                print(i,rmsetrain,rmsetest,'accepted')
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
                print(str(i)+'th sample running')
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
        print ((langevin_count*100 / (samples * 1.0)), '% was Langevin')
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
