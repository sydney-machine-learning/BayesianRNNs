#sweet simple single chain directly sampled from ayush's Platenoid_single_chain_code.

from src.model import Model
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import argparse
import random
import copy


parser = argparse.ArgumentParser(description='PTBayeslands modelling Single Chain')
parser.add_argument('-s','--samples', help='Number of samples', default=50000, dest="samples",type=int)
# parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=10,dest="num_chains",type=int)
# parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=2,dest="mt_val",type=int)
# parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.1,type=float)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default = 0.5,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)
parser.add_argument('-step','--step', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",default=0.025,type=float)
parser.add_argument('-lr','--learn', help='learn rate for langevin gradient', dest="learn_rate",default=0.1,type=float)
parser.add_argument('-m','--model', help='1 to select RNN, 2 to select LSTM', dest = "net", default = 1, type= int)
parser.add_argument('-o','--optim', help='1 to select SGD, 2 to select Adam', dest = 'optimizer', default = 1, type = int)
parser.add_argument('-prob', '--problem', help= 'Sunspot: 1', dest = 'problem', default= 1, type = int)
parser.add_argument('-lf', '--lgprob', help= 'Langevin_prob, default: 0.9', dest = 'l_prob', default = 0.9, type= int)
args = parser.parse_args()



class MCMC:
    def __init__(self, use_langevin_gradients, learn_rate, samples, train_x, train_y, test_x, test_y, topology, rnn_net= 'RNN', optimizer= 'SGD', path):
        self.use_langevin_gradients = use_langevin_gradients
        self.samples = samples
        self.topology = topology
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.learn_rate = learn_rate
        self.rnn = Model(topo= topology, lrate = learn_rate, rnn_net= rnn_net, optimizer = optimizer )


    def rmse(self, predictions, target):
        return np.sqrt(((predictions - target)**2).mean())

    def step_wise_rmse(self, pred, actual):
        return np.sqrt(np.mean((pred - actual)**2, axis = 0, keepdims= True))

    def likelihood_func(self, rnn,x,y, w, tau_sq):
        fx = rnn.evaluate_proposal(x,w)
        rmse = self.rmse(fx, y)
        step_wise_rmse = self.step_wise_rmse(fx,y)
        loss = np.sum(-0.5*np.log(2*math.pi*tau_sq) - 0.5*np.square(y-fx)/tau_sq)
        #the likelihood function returns the sum of rmse values for all 10 steps. 
        return [np.sum(loss), fx, rmse, step_wise_rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq,rnn):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((len(rnn.getparameters(w))) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(rnn.getparameters(w))))
        log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss


    def sampler(self):
        samples = self.samples
        rnn = self.rnn
        x_test = self.test_x
        y_train = self.train_y
        x_train = self.train_x

        learn_rate = self.learn_rate
        w_size = sum(p.numel() for p in rnn.parameters())
        pos_w = np.ones((samples, w_size))
        step_wise_rmse_train = np.zeros((samples, 10))
        step_wise_rmse_test = np.zeros((samples, 10))
        rmse_train = np.zeros((samples))
        rmse_test = np.zeros(shape = (samples))
        pos_tau = np.zeros(shape = (samples))

        w = copy.deepcopy(rnn.state_dict())
        eta = 0 
        step_w = args.step_size
        step_eta = 0.2

        pred_train = rnn.evaluate_proposal(x_train, w)
        pred_test = rnn.evaluate_proposal(x_test, w)
        eta = np.log(np.var(pred_train - np.array(y_train)))
        tau_pro = np.exp(eta)
        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0
        sigma_diagmat = np.zeros((w_size, w_size))
        np.fill_diagonal(sigma_diagmat, step_w**2)

        prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, rnn)
        [likelihood, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(rnn, x_train, y_train, w, tau_pro)
        [_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(rnn, x_test, y_test, w, tau_pro)
        prop_list = np.zeros((samples, w_size))
        likeh_list = np.zeros((samples, 2))
        likeh_list[0, :] = [-100, -100]

        accept_list = np.zeros(samples)
        num_accepted = 0 
        langevin_count = 0 
        pt_samples = samples * args.pt_samples
        init_count = 0 
        
        for i in range(samples-1):
            # timer1 = time.time()
            lx = np.random.uniform(0,1,1)
            if (self.use_langevin_gradients is True) and lx < args.l_prob : 
                w_gd = rnn.langevin_gradient(self.train_x, self.train_y, copy.deepcopy(w)))
                w_proposal = rnn.addnoiseandcopy(w_gd, 0, step_w)
                w_prop_gd = rnn.langevin_gradient(self.train_x, self.train_y, copy.deepcopy(w_proposal))
                
                wc_delta = (rnn.getparameters(w) - rnn.getparameters(w_prop_gd))
                wp_delta = (rnn.getparameters(w_proposal))
                sigma_sq = step_w
                """
                    doubt to be confirmed
                """
                first = -0.5 * np.sum(wc_delta * wc_delta) / sigma_sq
                second = -0.5 * np.sum(wp_delta * wp_delta)/ sigma_sq
                diff_prop = first - second
                #diff_prop = (first- second)/self.temp
                langevin_count += 1
            else:
                diff_prop = 0 
                w_proposal = rnn.addnoiseandcopy(w, 0, step_w) #step_w = args.step_size

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)
            [likelihood_proposal, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(rnn, self.train_x, self.train_y, w_proposal, tau_pro)
            [_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(rnn, self.test_x, self.test_y, w_proposal, tau_pro)

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, rnn)
            diff_prior = prior_prop - prior_current
            diff_likelihood = likelihood_proposal - likelihood

            try: 
                mh_prob = diff_likelihood + diff_prior + diff_prop
            except OverflowError as e: 
                mh_prob = 1
            
            accept_list[i+1] = num_accepted
            # if (i % batch_save + 1) == 0: 
            #     x = 0 
            """
                I don't see any use for this snippet
            """    
            prop_list = [i+1,] = rnn.get_parameters(w_proposal).reshape(-1)
            likeh_list[i+1,0] = likelihood_proposal
            """likeh_list is the list of all likelihood_proposals irrespective of whether they were accepted or not"""
            u = np.log(random.uniform(0,1))

            if u < mh_prob: 
                num_accepted += 1
                likelihood = likelihood_proposal
                prior_current = prior_prop
                w = copy.deepcopy(w_proposal)
                eta = eta_pro  """Here pro is short for proposal"""
                print(f'{i}:: RMSE Train: {rmsetrain} :: RMSE Test: {rmsetest}')
                pos_w[i+1,] = rnn.getparameters(w_proposal).reshape(-1)
                rmse_train[i+1,] = rmsetrain
                rmse_test[i+1,] = rmsetest
                step_wise_rmse_train[i+1,] = np.squeeze(step_wise_rmse_train, axis = -1)
                step_wise_rmse_test[i+1,] = np.squeeze(step_wise_rmse_test, axis= -1)
                pos_tau[i+1,] = tau_pro

            else:
                pos_w[i+1,] = pos_w[i,]
                pos_tau[i+1,] = pos_tau[i,]
                rmse_train[i+1,] = rmse_train[i,]
                rmse_test[i+1,] = rmse_test[i,]
                step_wise_rmse_train[i+1,] = step_wise_rmse_train[i,]
                step_wise_rmse_test[i+1,] = step_wise_rmse_test[i,]

            print(f'{num_accepted} proposals were accepted out of {samples}')
            accept_ratio = num_accepted*100/(samples*1.0) 
            print(f'{num_accepted*100/(samples*1.0)} % is the accept_percentage')
            
            print(f'{langevin_count*100/(samples*1.0)} % of total samples were langevin')
            
            """
            files to be saved:
            pos_w
            rmse_train
            rmse_test
            step_wise_rmse_train
            step_wise_rmse_test
            prop_list
            likeh_list
            pos_tau
            """
            #saving files skipped in this. Save them later in main.py directly after applying burn_in period

            return rmse_train, rmse_test, step_wise_rmse_train, step_wise_rmse_test, pos_w, pos_tau, prop_list, likeh_list
            


def main():
    n_steps_in, n_steps_out = 5,10
    net = 'RNN' if args.net == 1 else 'LSTM' 
    optimizer = 'SGD' if args.optimizer == 1 else 'Adam'
    print(f"Network is {net} and optimizer is {optimizer}")
    print("Name of folder to look for: ",os.getcwd()+'/Res_LG-Lprob_'+net+f'_{optimizer}')

    # problem = args.problem
    for j in [1,2,3,6]:
        problem = j
        folder = ".."
        if problem ==1:
            TrainData = pd.read_csv("../data/Lazer/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Lazer/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Lazer"
        if problem ==2:
            TrainData = pd.read_csv("../data/Sunspot/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Sunspot/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Sunspot"
        if problem ==3:
            TrainData = pd.read_csv("../data/Mackey/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Mackey/test1.csv",index_col = 0)
            TestData = TestData.values
            name="Mackey"
        if problem ==4:
            TrainData = pd.read_csv("../data/Lorenz/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Lorenz/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Lorenz"
        if problem ==5:
            TrainData = pd.read_csv("../data/Rossler/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Rossler/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Rossler"
        if problem ==6:
            TrainData = pd.read_csv("../data/Henon/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Henon/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Henon"
        if problem ==7:
            TrainData = pd.read_csv("../data/ACFinance/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/ACFinance/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "ACFinance"

        train_x = np.array(TrainData[:,0:n_steps_in])
        train_y = np.array(TrainData[:,n_steps_in : n_steps_in+n_steps_out ])
        test_x = np.array(TestData[:,0:n_steps_in])
        test_y = np.array(TestData[:,n_steps_in : n_steps_in+n_steps_out])

        train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
        train_y = train_y.reshape(train_y.shape[0],train_y.shape[1],1)
        test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],1)
        test_y = test_y.reshape(test_y.shape[0],test_y.shape[1],1)
        print("shapes of train x and y",train_x.shape,train_y.shape)
        
        Hidden = 10 
        topology = [n_steps_in, Hidden, n_steps_out]
        NumSamples = args.samples
        netw = topology
        burn_in = args.burn_in
        learn_rate = args.learn_rate
        langevin = ""
        use_langevin_gradients = True

        problemfolder = os.getcwd() + f'/Res_LG-Lprob_{net}_{optimizer}_{num_chains}chains/'
        filename = ""
        run_nb = 0
        while os.path.exists(problemfolder + name + langevin + '_%s' % (run_nb)):
            run_nb += 1
        if not os.path.exists(problemfolder + name + langevin + '_%s' % (run_nb)):
            os.makedirs(problemfolder + name + langevin + '_%s' % (run_nb))
            path = (problemfolder + name + langevin + '_%s' % (run_nb))
        
        start_timer = time.time()
        langevin_prob = args.l_prob
         
        mcmc = MCMC(use_langevin_gradients, learn_rate, NumSamples, train_x, train_y, test_x, test_y, topology, rnn_net = 'RNN', optimizer = 'SGD',path)
        rmse_train, rmse_test, step_wise_rmse_train, step_wise_rmse_test, pos_w, pos_tau, prop_list, likeh_list = mcmc.sampler()

        stop_timer = time.time()
        training_time = stop_timer - start_timer
        print(f'Time take to sample: {training_time}')
        #Creating directories to store the results/ return objects from sampler

        directories = [path + '/predictions',
                       path + '/posterior',
                       path + '/results',
                    #    path + '/posterior/pos_w',
                    #    path + '/posterior/pos_likelihood',
                    #    path + '/posterior/accept_list'
                       ]

        for d in directories:
            os.makedirs(d)

        #Apply Burn_in to the results:
        burnin = int(NumSamples * burn_in)
        # likelihood_rep = 
        # accept_percent = 
        accept_list = np.zeros((1,NumSamples))
        pos_w_afterburn = np.zeros((1, NumSamples - burn_in, pos_w.shape[-1]))
        fx_train_all = np.zeros((1, NumSamples-burn_in, train_x.shape[0]))
        rmse_train_afterburn = np.zeros((1, NumSamples-burn_in))
        step_wise_rmse_train_afterburn = np.zeros((1, NumSamples- burn_in, 10))
        step_wise_rmse_test_afterburn = np.zeros((1, NumSamples- burn_in, 10))
        fx_test_all = np.zeros((1, NumSamples- burn_in, x_test.shape[0]))
        rmse_test_afterburn = np.zeros((1, NumSamples- burn_in))

        pos_w[0,:,:] = pos_w        

        
