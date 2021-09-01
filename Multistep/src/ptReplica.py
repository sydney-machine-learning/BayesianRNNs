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
    def __init__(self, use_langevin_gradients, learn_rate,  w,  minlim_param, maxlim_param, samples,trainx,trainy,testx,testy, topology, burn_in, temperature, swap_interval, langevin_prob, path, parameter_queue, main_process,event,rnn_net, optimizer):
        #MULTIPROCESSING VARIABLES
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event =  event
        self.batch_size = {
            'train_x': trainx.shape,
            'train_y': trainy.shape,
            'test_x' : testx.shape,
            'test_y' : testy.shape
        }
        self.rnn = Model(topology,learn_rate,batch_size= self.batch_size, input_size= 1,rnn_net=rnn_net, optimizer = optimizer)
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
        if len(actual.shape)>2 and actual.shape[2]==1: actual = np.squeeze(actual,axis = 2)
        if len(pred.shape) > 2 and actual.shape[2]==1: pred = np.squeeze(pred, axis = 2)
        step_wise_rmse = np.sqrt(np.mean((pred - actual)**2, axis = 0, keepdims= True))
        # old_rmse = np.sqrt(((pred-actual)**2).mean())
        # print(f"old rmse in code: {old_rmse}")
        new_rmse = np.mean(step_wise_rmse)
        # print(f'rmse : {new_rmse}')
        return new_rmse

    def step_wise_rmse(self, pred, actual):
        if len(actual.shape)>2 and actual.shape[2]==1: actual = np.squeeze(actual,axis = 2)
        if len(pred.shape) > 2 and actual.shape[2]==1: pred = np.squeeze(pred, axis = 2)
        ans = np.sqrt(np.mean((pred - actual)**2, axis = 0, keepdims= True))
        
        # print("shape of step_wise_rmse: ",ans.shape)
        return ans




    def likelihood_func(self, rnn,x,y, w, tau_sq, temp):
 
        fx = rnn.evaluate_proposal(x,w)
 
        rmse = self.rmse(fx, y)
        step_wise_rmse = self.step_wise_rmse(fx,y) 

        n = y.shape[0] * y.shape[1]

        p1 = -(n/2)*np.log(2*math.pi*tau_sq) 

        p2 = (1/(2*tau_sq)) 

        log_lhood = p1 -  (p2 * np.sum(np.square(y -fx)) )

        pseudo_loglhood =  log_lhood/temp       
        #question: what should go into parameter queue? log_lhood or pseudo_loglhood
        return [pseudo_loglhood, fx, rmse, step_wise_rmse]


  

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w,  tausq,rnn):  #leave this tausq as it is
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
        y_train = np.squeeze(self.train_y, axis = -1)
        y_test = np.squeeze(self.test_y, axis = -1)
        batch_save = 10  # batch to append to file
        learn_rate = self.learn_rate
        rnn = self.rnn#Model(self.topology,learn_rate,rnn_net = self.rnn_net)
        w_size = sum(p.numel() for p in rnn.parameters())
        pos_w = np.ones((samples, w_size)) #Posterior for all weights
        step_wise_rmse_train = np.zeros(shape = (samples,10))
        step_wise_rmse_test  = np.zeros(shape = (samples,10))
        rmse_train  = np.zeros(samples)
        rmse_test = np.zeros(samples)
        # acc_train = np.zeros(samples)
        # acc_test = np.zeros(samples)
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
        # print("tau_pro: ", tau_pro)

        # eta = np.log(np.var(pred_train - np.array(y_train) , axis = 0, keepdims= True )) # [1,10,1]
        # tau_pro = np.exp(eta)
        # tau_pro = np.squeeze(tau_pro)[0]
        """experiment under consideration"""
        # print("Shape of pred train: ", pred_train.shape)
        # print("Shape of y_train: ", y_train.shape)
        # eta_vec = np.log(np.var(pred_train - np.array(y_train) , axis = 0, keepdims= False )) # [1,10,1]
        # print(eta_vec.shape)
        # print("ETA_VEC: ", eta_vec )
        
        
        # eta = eta_vec[0]
        #tau_pro to be used in prior func
        # tau_pro = np.exp(eta)

        #make another tau_pro_vec to be used in likelihood func
        # tau_pro_vec = np.exp(eta_vec)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
        np.fill_diagonal(sigma_diagmat, step_w**2)
        #delta_likelihood = 0.5 # an arbitrary position
        prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, rnn)  # takes care of the gradients
        # print("prior current shape ", prior_current)
        [likelihood, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(rnn, self.train_x, y_train, w, tau_pro, temp= self.temperature)
        [_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(rnn, self.test_x, y_test, w, tau_pro, temp= self.temperature)
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
                [likelihood, pred_train, rmsetrain, step_wise_rmsetrain ] = self.likelihood_func(rnn, self.train_x, y_train, w, tau_pro, temp= self.adapttemp)
                [_, pred_test, rmsetest, step_wise_rmsetest ] = self.likelihood_func(rnn, self.test_x, y_test, w, tau_pro, temp= self.adapttemp)
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
            # eta_pro = eta_pro_vec[0]
            # print('eta ki dusri baar me shape: ', eta_pro.shape)
            # tau_pro_vec = np.exp(eta_pro_vec)
            tau_pro = np.exp(eta_pro)
            # print("TAU_PRO during training: ", tau_pro)
            # tau_pro = np.squeeze(tau_pro)[0]
            # print("tau pro ki baad me shape: ", tau_pro.shape)
            [likelihood_proposal, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(rnn, self.train_x, y_train, w_proposal,tau_pro, temp= self.adapttemp)
            [_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(rnn, self.test_x, y_test, w_proposal,tau_pro, temp= self.adapttemp)
            # print(step_wise_rmsetest.shape,"shape of step wise rmse")
            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,tau_pro,rnn)  # takes care of the gradients
            # print("prior proposal ", prior_prop)
            diff_prior = prior_prop - prior_current
            diff_likelihood = likelihood_proposal - likelihood
            surg_likeh_list[i+1,0] = likelihood_proposal * self.adapttemp
            #surg_likeh_list[i+1,1] = np.nan
            try:
                # mh_prob = min(0, (diff_likelihood + diff_prior + diff_prop))
               
                mh_prob = diff_likelihood + diff_prior + diff_prop   #https://github.com/sydney-machine-learning/BayesianAutoencoders/blob/239a2ab9e81fc6b95cb60303303377b0dac37b7c/Bayesian/Parallel_Tempering_Tabular.py#L519
                # mh_prob = min(1, np.exp(diff_likelihood+ diff_prior + diff_prop , dtype = np.float128)) #as given in paper
            except OverflowError as e:
                mh_prob = 1
            accept_list[i+1] = num_accepted
            if (i % batch_save+1) == 0: # just for saving posterior to file - work on this later
                x = 0
            u = np.log(random.uniform(0, 1))
            # u = random.uniform(0,1)

            prop_list[i+1,] = rnn.getparameters(w_proposal).reshape(-1)
            likeh_list[i+1,0] = likelihood_proposal
            if u < mh_prob:
                num_accepted  =  num_accepted + 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)
                eta = eta_pro
                # acc_train[i+1,] = 0
                # acc_test[i+1,] = 0
                print(i,'RMSE Train: ',rmsetrain,"RMSE Test: ", rmsetest,' accepted')
                pos_w[i+ 1,] = rnn.getparameters(w_proposal).reshape(-1)
                rmse_train[i + 1,] = rmsetrain
                step_wise_rmse_train[i+1,] = step_wise_rmsetrain
                rmse_test[i + 1,] = rmsetest
                step_wise_rmse_test[i+1,] = step_wise_rmsetest          
            else:
                pos_w[i+1,] = pos_w[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                step_wise_rmse_train[i+1,] = step_wise_rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]
                step_wise_rmse_test[i+1,] = step_wise_rmse_test[i,]
                # acc_train[i+1,] = acc_train[i,]
                # acc_test[i+1,] = acc_test[i,]
            if ((i+1) % self.swap_interval == 0 and i != 0 ):
                print(str(i)+'th sample running')
                w_size = rnn.getparameters(w).reshape(-1).shape[0]
                # print(rnn.getparameters(w).reshape(-1).shape)
                # print(np.asarray([eta]).reshape(1).shape)
                # print(np.asarray([likelihood*self.temperature]).shape)
                # print(np.asarray([self.temperature]).shape)    
                param = np.concatenate([rnn.getparameters(w).reshape(-1), np.asarray([eta]).reshape(1), 
                                        np.asarray([likelihood*self.temperature]).reshape(-1),np.asarray([self.temperature])])
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                # print("line 199")
                try:
                    self.event.wait()   #the issue is here 
                except Exception as e:
                    print(e)
                    continue
                # print("line 201")
                param1 = self.parameter_queue.get()
                w1 = param1[0:w_size]
                eta = param1[w_size]
                w1_dict = rnn.dictfromlist(w1)
                rnn.loadparameters(w1_dict)


        # print(f'length of step_wise rmse before burnin: {step_wise_rmse_train.shape}, {step_wise_rmse_test.shape}')

        param = np.concatenate([rnn.getparameters(w).reshape(-1), np.asarray([eta]).reshape(1), np.asarray([likelihood]).reshape(1),np.asarray([self.adapttemp]),np.asarray([i])]) #eta_vec.reshape(1)
        # print("THIS IS THE PARAMETER QUEUE: ", param)
        # print("SHAPE OF PARAMETER QUUEU:" , param.shape)
        self.parameter_queue.put(param)
        self.signal_main.set()
        print ((num_accepted*100 / (samples * 1.0)), f'% samples out of {samples} were accepted')
        accept_ratio = num_accepted / (samples * 1.0) * 100
        print ((langevin_count*100 / (samples * 1.0)), '% of total samples were Langevin')
        langevin_ratio = langevin_count / (samples * 1.0) * 100
        file_name = self.path+'/posterior/pos_w/'+'chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,pos_w )
        file_name = self.path+'/predictions/rmse_test_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_test, fmt='%1.8f')
        file_name = self.path+'/predictions/rmse_train_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_train, fmt='%1.8f')
        file_name = self.path+'/predictions/stepwise_rmse_train_chain_'+ str(self.temperature) + '.txt'
        np.savetxt(file_name, step_wise_rmse_train, fmt = '%1.8f')
        file_name = self.path + '/predictions/stepwise_rmse_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, step_wise_rmse_test, fmt = '%1.8f')
        # file_name = self.path+'/predictions/acc_test_chain_'+ str(self.temperature)+ '.txt'
        # np.savetxt(file_name, acc_test, fmt='%1.2f')
        # file_name = self.path+'/predictions/acc_train_chain_'+ str(self.temperature)+ '.txt'
        # np.savetxt(file_name, acc_train, fmt='%1.2f')
        file_name = self.path+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,likeh_list, fmt='%1.4f')
        file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
        np.savetxt(file_name, [accept_ratio], fmt='%1.4f')
        file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, accept_list, fmt='%1.4f')

        # print(pred_train.shape, " pred_train shape")
        # print(pred_test.shape, " pred test shape")
        # print(pred_train, pred_test)
        file_name = self.path + '/predictions/final_pred_train_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, np.squeeze(pred_train), fmt='%1.4f')
        file_name = self.path + '/predictions/final_pred_test_chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, np.squeeze(pred_test), fmt= '%1.4f')

        print('exiting this thread')
    