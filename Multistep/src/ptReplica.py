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
	def __init__(self, use_langevin_gradients, learn_rate, w, minlim_param, 
				 maxlim_param, samples, trainx, trainy, testx, testy,
				 topology, burn_in, temperature, swap_interval, langevin_prob,
				 path, parameter_queue,main_process, event, rnn_net, optimizer):
		multiprocessing.Process.__init__(self)
		self.processID = temperature 
		self.parameter_queue = parameter_queue 
		self.signam_main = main_process 
		self.event = event 
		self.rnn = Model(topology, learn_rate, input_size = 1, rnn_net = rnn_net, optimzer = optimizer)
		self.temperature = temperature
		self.adapttemp = temperature
		self.swap_interval = swap_interval
		self.path = path
		self.burn_in = burn_in
		self.rnn_net = rnn_net 
		self.samples = samples
		self.topology = topology
		self.train_x = torch.FloatTensor(trainx)
		self.train_y = torch.FloatTensor(trainy)
		self.test_x = torch.FloatTensor(testx)
		self.test_y = torch.FloatTensor(testy)
		self.w = w
		self.minY = np.zeros((1,1))
		self.maxY = np.zeros((1,1))
		#self.rnn
		self.minlim_param = minlim_param
		self.maxlim_param = maxlim_param
		self.use_langevin_gradients = use_langevin_gradients
		self.sgd_depth = 1 # always should be 1
		self.learn_rate = learn_rate
		self.l_prob = langevin_prob  

	def rmse(self, pred, actual):
		if len(actual.shape)>2 and actual.shape[2]==1: actual = np.squeeze(actual,axis = 2)
		if len(pred.shape) > 2 and pred.shape[2]==1: pred = np.squeeze(pred, axis = 2)
		step_wise_rmse = np.sqrt(np.mean((pred - actual)**2, axis = 0))
		return np.mean(step_wise_rmse)

	def step_wise_rmse(self, pred, actual):
		if len(actual.shape)>2 and actual.shape[2]==1: actual = np.squeeze(actual,axis = 2)
		if len(pred.shape) > 2 and pred.shape[2]==1: pred = np.squeeze(pred, axis = 2)
		step_wise_rmse = np.sqrt(np.mean((pred - actual)**2, axis = 0))
		return np.sqrt(np.mean((pred - actual)**2, axis = 0))

	def likelihood_func(self, rnn, x, y, w, tau_sq, temp):
		fx = rnn.evaluate_proposal(x, w)
		y = y.numpy()
		rmse = self.rmse(fx, y)
		step_wise_rmse = self.step_wise_rmse(fx, y)
		n = y.shape[0]* y.shape[1]
		p1 = -(n/2)*np.log(2*math.pi*tau_sq)
		p2 = (1/(2*tau_sq))
		# print("is this it?",(y-fx).shape)
		# print(p1.shape)
		# print(np.sum(np.square(y-fx)))
		log_lhood = p1 - (p2*np.sum(np.square(y-fx)))
		pseudo_loglhood = log_lhood/temp
		return [pseudo_loglhood, fx, rmse, step_wise_rmse]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2 ,w, tausq, rnn):
		h = self.topology[1]
		d = self.topology[0]
		part1 = -1 * ((len(rnn.getparameters(w))) / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(rnn.getparameters(w))))
		log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
		return log_loss


	def run(self):
		w_size = sum(p.numel() for p in self.rnn.parameters())
		pos_w = np.ones((self.samples, w_size))
		step_wise_rmse_train = np.zeros(shape = (self.samples,10))
		step_wise_rmse_test  = np.zeros(shape = (self.samples,10))
		rmse_train  = np.zeros(self.samples)
		rmse_test = np.zeros(self.samples)
		prop_list = np.zeros((self.samples, w_size))
		likeh_list = np.zeros((self.samples,2 ))
		likeh_list[0,:] = [-100,-100]
		accept_list = np.zeros(self.samples)
		num_accepted = 0
		langevin_count = 0
		pt_samples = self.samples*self.burn_in
		init_count = 0
		if self.optimizer == 'SGD':
			optimizer = torch.optim.SGD(self.rnn.parameters(), lr= self.learn_rate)
		elif self.optimizer == 'Adam':
			optimizer = torch.optim.Adam(self.rnn.parameters(), lr= self.learn_rate)

		w = copy.deepcopy(self.rnn.state_dict())
		pred_train = self.rnn.evaluate_proposal(self.train_x, w)
		pred_test = self.rnn.evaluate_proposal(self.test_x, w)
		eta = np.log(np.var(pred_train - np.array(self.train_y)))
		tau_pro = np.exp(eta)

		eta = 0
		step_w = 0.01
		step_eta = 0.2
		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0
		
		sigma_diagmat = np.zeros((w_size, w_size))
		np.fill_diagonal(sigma_diagmat, step_w**2)
		prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, self.rnn)
		[likelihood, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(self.rnn, self.train_x, self.train_y, copy.deepcopy(w), tau_pro, temp=self.temperature)
		[_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(self.rnn, self.test_x, self.test_y, copy.deepcopy(w), tau_pro, temp=self.temperature)
		
		scaling_factor = 0.01 #0.05 works great too
		self.event.clear()
		i = 0 
		for i in range(samples -1):
			timer1 = time.time()
			if i < pt_samples:
				self.adapttemp = self.temperature 
			if i == pt_samples and init_count == 0:
				self.adapttemp = 1
				[likelihood, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(self.rnn, self.train_x, self.train_y, copy.deepcopy(w), tau_pro, temp = self.adapttemp)
				[_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(self.rnn, self.test_x, self.test_y, copy.deepcopy(w), tau_pro, temp= self.adapttemp)
				init_count = 1
			
			lx = np.random.uniform(0,1,1)
			if(self.use_langevin_gradients and lx < self.l_prob):
				w_gd = copy.deepcopy(self.rnn.langevin_gradient(self.train_x, self.train_y, copy.deepcopy(w),optimizer = optimizer))
				w_proposal = self.rnn.addnoiseandcopy(w_gd, mean = 0, std_dev=step_w)
				w_prop_gd = copy.deepcopy(self.rnn.langevin_gradient(self.train_x, self.train_y, copy.deepcopy(w), optimzer= optimizer))
				wc_delta = self.rnn.getparameters(w) - self.rnn.getparameters(w_prop_gd)
				wp_delta = self.rnn.getparameters(w_proposal) - self.rnn.getparameters(w_gd)
				sigma_sq = step_w*step_w
				# sigma_sq = step_w
				first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
				second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq
				diff_prop =  (first - second)*scaling_factor
				langevin_count = langevin_count + 1

			else:
				diff_prop = 0
				w_proposal = self.rnn.addnoiseandcopy(w,0,step_w)
			
			eta_pro = eta + np.random.normal(0, step_eta, 1)
			tau_pro = np.exp(eta_pro)

			[likelihood_proposal, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(self.rnn, self.train_x, self.train_y, copy.deepcopy(w_proposal), tau_pro, temp=self.adapttemp)
			[_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(self.rnn, self.train_x, self.train_y, copy.deepcopy(w_proposal), tau_pro, temp=self.adapttemp)

			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, self.rnn)
			diff_prior = prior_prop - prior_current
			diff_likelihood = likelihood_proposal - likelihood 

			try: 
				mh_prob = diff_likelihood + diff_prior + diff_prop 
			except OverflowError as e:
				mh_prob = 1
			accept_list[i+1] = num_accepted
			u = np.log(random.uniform(0,1)) 

			prop_list[i+1,] = rnn.getparameters(w_proposal).reshape(-1)
			likeh_list[i+1,0] = likelihood_proposal
			if u < mh_prob:
				num_accepted = num_accepted + 1
				likelihood = likelihood_proposal
				prior_current = prior_prop 
				w = copy.deepcopy(w_proposal)
				eta = eta_pro 
				print(i, likelihood_proposal, rmsetrain, rmsetest, ' accepted!')
				pos_w[i+1,] = self.rnn.getparameters(w_proposal).reshape(-1)
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


			if((i+1)% self.swap_interval == 0 and i!= 0):
				print(str(i) + 'th sample running')
				w_size = rnn.getparameters(copy.deepcopy(w)).reshape(-1).shape[0]
				param = np.concatenate([rnn.getparameters(w).reshape(-1),
										np.asarray([eta]).reshape(1),
										np.asarray([likelihood*self.temperature]).reshape(-1),
										np.asarray([self.temperature])]
										)
				self.parameter_queue.put(param)
				self.signal_main.set()
				self.event.clear() 

				try:
					self.event.wait()
				except Exception as e:
					print(e)
					continue 
			
				param1 = self.parameter_queue.get()
				w1 = param1[0:w_size]
				eta = param1[w_size]
				w1_dict = rnn.dictfromlist(w1)
				rnn.load_state_dict(w1_dict)

		
		param = np.concatenate(
			[
				rnn.getparameters(w).reshape(-1), 
				np.asarray([eta]).reshape(1), 
				np.asarray([likelihood]).reshape(1),
				np.asarray([self.adapttemp]),
				np.asarray([i])
			]
		)
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
		file_name = self.path+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,likeh_list, fmt='%1.4f')
		file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.4f')
		file_name = self.path + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
		np.savetxt(file_name, accept_list, fmt='%1.4f')
		file_name = self.path + '/predictions/final_pred_train_chain_' + str(self.temperature) + '.txt'
		np.savetxt(file_name, np.squeeze(pred_train), fmt='%1.4f')
		file_name = self.path + '/predictions/final_pred_test_chain_' + str(self.temperature) + '.txt'
		np.savetxt(file_name, np.squeeze(pred_test), fmt= '%1.4f')

		print('exiting this thread')
