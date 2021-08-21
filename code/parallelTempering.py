from model import Model
import os
import numpy as np
import copy
import math
import multiprocessing
import datetime
from ptReplica import ptReplica

class ParallelTempering:
    def __init__(self,  use_langevin_gradients, learn_rate, train_x,train_y,test_x,test_y, topology, num_chains, maxtemp, NumSample, swap_interval, langevin_prob, path,rnn_net = 'RNN', optimizer = 'SGD'):
        #FNN Chain variables
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.topology = topology
        self.rnn = Model(self.topology,learn_rate,rnn_net = rnn_net, optimizer= optimizer)
        self.num_param = sum(p.numel() for p in self.rnn.parameters())#(topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2] + (topology[1] * topology[1])
        #Parallel Tempering variables
        self.swap_interval = swap_interval
        self.path = path
        self.maxtemp = maxtemp
        self.rnn_net = rnn_net

        self.optimizer = optimizer
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

    def default_beta_ladder(self, ndim, ntemps,
                            Tmax):   

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')
 

        maxtemp = Tmax
        numchain = ntemps
        b = []
        b.append(maxtemp)
        last = maxtemp
        for i in range(maxtemp):
            last = last * (numchain ** (-1 / (numchain - 1)))
            b.append(last)
        tstep = np.array(b)

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
        else:
            tstep = tstep[ndim - 1]

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
        
        
        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)
            for i in range(0, self.num_chains):
                self.temperatures.append(np.inf if betas[i] is 0 else 1.0 / betas[i])
                print(self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp / self.num_chains)
            temp = 1
            for i in range(0, self.num_chains):
                self.temperatures.append(temp)
                temp += tmpr_rate
                print(self.temperatures[i])
    def initialize_chains(self,  burn_in):
        self.burn_in = burn_in
        self.assign_temperatures()
        self.minlim_param = np.repeat([-100] , self.num_param)  # priors for nn weights
        self.maxlim_param = np.repeat([100] , self.num_param)
        for i in range(0, self.num_chains):
            w = np.random.randn(self.num_param)
            w = self.rnn.dictfromlist(w)  #from ayush's code
            self.chains.append(ptReplica(self.use_langevin_gradients, self.learn_rate,
                                         w,  self.minlim_param, self.maxlim_param, 
                                         self.NumSamples, self.train_x,self.train_y,
                                         self.test_x,self.test_y,self.topology,self.burn_in,
                                         self.temperatures[i],self.swap_interval, 
                                         self.langevin_prob, self.path,
                                         self.parameter_queue[i],self.wait_chain[i],
                                         self.event[i],self.rnn_net, self.optimizer))
            print(f'chain {i} initiated')

 
    rmse = ptReplica.rmse
    step_wise_rmse = ptReplica.step_wise_rmse
    likelihood_func = ptReplica.likelihood_func
    

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        param1 = parameter_queue_1.get()
        param2 = parameter_queue_2.get()
        w1 = param1[0:self.num_param]
        w1 = self.rnn.dictfromlist(w1)
        eta1_vec = param1[self.num_param:self.num_param + 10]
        #tau_sq_1_vec = np.exp(eta1_vec)


        # lhood1 = param1[self.num_param+10]
        lhood1 = param1[-2]
        # T1 = param1[self.num_param+10+1]
        T1 = param1[-1]

        w2 = param2[:self.num_param]
        w2 = self.rnn.dictfromlist(w2)
        eta2_vec = param2[self.num_param: self.num_param + 10]


        tau_sq_2_vec = np.exp(eta2_vec)
        # lhood2 = param2[self.num_param + 10 + 0 ]
        lhood2 = param2[-2]
        # T2 = param2[self.num_param + 10 + 1]
        T2 = param2[-1]

        #Swapping Probabilities
        y_train = np.squeeze(self.train_y, axis = -1)
        
        #lhood12, dump1, dump2, dump3 = self.likelihood_func(self.rnn, self.train_x, y_train, w1, tau_sq_1_vec, temp = T1)
        #lhood21, dump1, dump2, dump3 = self.likelihood_func(self.rnn, self.train_x, y_train, w2, tau_sq_2_vec, temp = T2)

        #lhood12 = 0
        #lhood21 = 0


        try:

            swap_proposal = min(1, 0.5 * np.exp(min(709, lhood2 - lhood1)))
            #swap_proposal = min(1, np.exp((lhood12 - lhood1) + (lhood21 - lhood2)))
            #print('Swap Proposal: ',swap_proposal)
        except OverflowError as e:
            #print(f'lhood12: {lhood12}, lhood1: {lhood1}, lhood21: {lhood21}, lhood2: {lhood2}')
            # print("swap_proposal = min(1, np.exp((lhood12 - lhood1) + (lhood21 - lhood2)))")
            #print(e)
            swap_proposal = 1
        u = np.random.uniform(0,1)
        if u < swap_proposal:
            swapped = True
            self.num_swap += 1
            param_temp = param1
            param1 = param2
            param2 = param_temp
            print( lhood2,lhood1, ' have been swapped   **** **** *** ')
            #param1[self.num_param + 1] = lhood21
            #param2[self.num_param + 1] = lhood12
            #param1[self.num_param + 2] = T2
            #param2[self.num_param + 2] = T1

        else:
            swapped = False

        self.total_swap_proposals += 1
        return param1, param2, swapped

    def run_chains(self):
        # # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        # swap_proposal = np.ones(self.num_chains-1)
        # # create parameter holders for paramaters that will be swapped
        # replica_param = np.zeros((self.num_chains, self.num_param))
        # lhood = np.zeros(self.num_chains)
        # # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples-1
        # number_exchange = np.zeros(self.num_chains)
        # filen = open(self.path + '/num_exchange.txt', 'a')
        #RUN MCMC CHAINS
        for l in range(0,self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0,self.num_chains):
            self.wait_chain[j].clear()
            self.event[j].clear()
            self.chains[j].start()    #any process.start() would call upon the ptReplica.run() function
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
                if not self.chains[index].is_alive(): #count the number of dead chains 
                    count+=1
                    self.wait_chain[index].set()
                    print(str(self.chains[index].temperature) +" Dead"+str(index))
            if count == self.num_chains:
                break 
            
            timeout_count = 0
            for index in range(0,self.num_chains):
                #print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait()
                if flag:
                    #print("Signal from chain: {}".format(index+1))
                    timeout_count += 1
            if timeout_count != self.num_chains:
                #print("Skipping the swap!")
                continue
            #print("Event occured")
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
        pos_w, fx_train, fx_test, rmse_train, rmse_test, indiv_rmse_train, indiv_rmse_test, likelihood_vec, accept_vec, accept  = self.show_results()
        
        print("NUMBER OF SWAPS =", self.num_swap)
        print('total swap proposal',self.total_swap_proposals)
        print('num samples', self.NumSamples)
        print('swap interval',self.swap_interval)
        try:
            swap_perc = self.num_swap*100 /self.total_swap_proposals
        except ZeroDivisionError:
            swap_perc = 0
        return pos_w, fx_train, fx_test, rmse_train, rmse_test, indiv_rmse_train, indiv_rmse_test, likelihood_vec, swap_perc, accept_vec, accept
        # pos_w, fx_train, fx_test,  rmse_train, rmse_test, acc_train, acc_test,   likelihood_rep , swap_perc,accept_vec, accept = pt.run_chains()

    def show_results(self):
        burnin = int(self.NumSamples*self.burn_in)
        likelihood_rep = np.zeros((self.num_chains, self.NumSamples - 1, 2)) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        accept_percent = np.zeros((self.num_chains, 1))
        accept_list = np.zeros((self.num_chains, self.NumSamples ))
        pos_w = np.zeros((self.num_chains,self.NumSamples - burnin, self.num_param))
        fx_train_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.train_x.shape[0]))
        rmse_train = np.zeros((self.num_chains,self.NumSamples - burnin))
        step_wise_rmse_train = np.zeros((self.num_chains, self.NumSamples - burnin, 10))
        step_wise_rmse_test = np.zeros((self.num_chains, self.NumSamples - burnin, 10))
        # acc_train = np.zeros((self.num_chains,self.NumSamples - burnin))
        fx_test_all  = np.zeros((self.num_chains,self.NumSamples - burnin, self.test_x.shape[0]))
        rmse_test = np.zeros((self.num_chains,self.NumSamples - burnin))
        # acc_test = np.zeros((self.num_chains,self.NumSamples - burnin))
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
            # print(f'origin shape of rmse_train_chain_{self.temperatures[i]}: {dat.shape} \nYou may stop the code now')
            rmse_train[i,:] = dat[burnin:]
            file_name = self.path+'/predictions/stepwise_rmse_train_chain_'+ str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            step_wise_rmse_train[i,:] = dat[burnin:]
            file_name = self.path+'/predictions/stepwise_rmse_test_chain_'+ str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name)
            step_wise_rmse_test[i,:] = dat[burnin:]

            # file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperatures[i])+ '.txt'
            # dat = np.loadtxt(file_name)
            # acc_test[i,:] = dat[burnin:]
            # file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperatures[i])+ '.txt'
            # dat = np.loadtxt(file_name)
            # acc_train[i,:] = dat[burnin:]
        posterior = pos_w.transpose(2,0,1).reshape(self.num_param,-1)
        fx_train = fx_train_all.transpose(2,0,1).reshape(self.train_x.shape[0],-1)  # need to comment this if need to save memory
        fx_test = fx_test_all.transpose(2,0,1).reshape(self.test_x.shape[0],-1)
        #fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0]) # konarks version
        likelihood_vec = likelihood_rep.transpose(2,0,1).reshape(2,-1)
        rmse_train = rmse_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        step_wise_rmse_train = step_wise_rmse_train.reshape(self.num_chains*(self.NumSamples - burnin), 10)
        # acc_train = acc_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        rmse_test = rmse_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        step_wise_rmse_test = step_wise_rmse_test.reshape(self.num_chains*(self.NumSamples - burnin), 10)
        # acc_test = acc_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)
        accept_vec  = accept_list
        #print(accept_vec)
        accept = np.sum(accept_percent)/self.num_chains
        #np.savetxt(self.path + '/pos_param.txt', posterior.T)  # tcoment to save space
        np.savetxt(self.path + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')
        np.savetxt(self.path + '/accept_list.txt', accept_list, fmt='%1.2f')
        np.savetxt(self.path + '/acceptpercent.txt', [accept], fmt='%1.2f')
            
        return posterior, fx_train_all, fx_test_all,   rmse_train, rmse_test,  step_wise_rmse_train, step_wise_rmse_test, likelihood_vec.T, accept_vec , accept
    def make_directory (self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
