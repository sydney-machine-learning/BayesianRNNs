import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import argparse
import random
import copy
import time
import math
import gc

mpl.use('agg')
parser=argparse.ArgumentParser(description='PTBayeslands modelling')
parser.add_argument('-s','--samples', help='Number of samples', default=30, dest="samples",type=int)
# parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=8,dest="num_chains",type=int)
# parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=2,dest="mt_val",type=int)
# parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.001,type=float)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default = 0.5,type=float)
# parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)
parser.add_argument('-step','--step', help='Step size for proposals (0.02, 0.05, 0.1 etc)', dest="step_size",default=0.025,type=float)
parser.add_argument('-lr','--learn', help='learn rate for langevin gradient', dest="learn_rate",default=0.01,type=float)
parser.add_argument('-m','--model', help='1 to select RNN, 2 to select LSTM', dest = "net", default = 2, type= int)
parser.add_argument('-o','--optim', help='1 to select SGD, 2 to select Adam', dest = 'optimizer', default = 2, type = int)
parser.add_argument('-debug','--debug', help='debug = 0 or 1; when 1 trace plots will not be produced', dest= 'DEBUG', default= 1, type= bool)
args = parser.parse_args()



def histogram_trace(pos_points, fname): # this is function to plot (not part of class)

	size = 15

	plt.tick_params(labelsize=size)
	params = {'legend.fontsize': size, 'legend.handlelength': 2}
	plt.rcParams.update(params)
	plt.grid(alpha=0.75)
	
	plt.hist(pos_points,  bins = 20, color='#0504aa', alpha=0.7)
	plt.title("Posterior distribution ", fontsize = size)
	plt.xlabel(' Parameter value  ', fontsize = size)
	plt.ylabel(' Frequency ', fontsize = size)
	plt.tight_layout()
	plt.savefig(fname + '_posterior.png')
	plt.clf()


	plt.tick_params(labelsize=size)
	params = {'legend.fontsize': size, 'legend.handlelength': 2}
	plt.rcParams.update(params)
	plt.grid(alpha=0.75)
	plt.plot(pos_points)

	plt.title("Parameter trace plot", fontsize = size)
	plt.xlabel(' Number of Samples  ', fontsize = size)
	plt.ylabel(' Parameter value ', fontsize = size)
	plt.tight_layout()
	plt.savefig(fname  + '_trace.png')
	plt.clf()

def plot_figure(lista, title,path):

	list_points =  lista
	size = 20
	fname = path+'/trace_plots'
	width = 9

	font = 12
	fig = plt.figure(figsize=(10,12))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)

	ax1 = fig.add_subplot(211)

	n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', density=False)


	color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']

	ax1.grid(True)
	ax1.set_ylabel('Frequency',size= font+1)
	ax1.set_xlabel('Parameter values', size= font+1)

	ax2 = fig.add_subplot(212)

	list_points = np.asarray(np.split(list_points,  args.num_chains ))

	ax2.set_facecolor('#f2f2f3')
	ax2.plot( list_points.T , label=None)
	ax2.set_title(r'Trace plot',size= font+2)
	ax2.set_xlabel('Samples',size= font+1)
	ax2.set_ylabel('Parameter values', size= font+1)

	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	if not os.path.exists(fname):
		os.makedirs(fname)

	plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
	plt.close()




class Model(nn.Module):
		# Defining input size, hidden layer size, output size and batch size respectively
	def __init__(self, topo,lrate,batch_size, input_size = 1,rnn_net = 'RNN', optimizer = 'SGD'):
		"""
			rnn_net = RNN/LSTM
			optimizer = Adam/ SGD
			topology = [input size, hidden size, output size]
			output size would be the same as num_classes
			num_classes =  topo[2]
			input_size = 1
			hidden_state = hidden_size = 10

		"""
		super(Model, self).__init__()
		# assuming num_dicrections to be 1 for all the cases
		# Defining some parameters
		self.hidden_dim = topo[1]#hidden_dim
		self.input_seq_len = topo[0]
		self.num_outputs = topo[2]
		self.input_size = input_size
		self.n_layers = 1   #len(topo)-2 #n_layers
		self.batch_size = 1
		self.lrate = lrate
		self.optimizer = optimizer
		
		self.rnn_net = rnn_net
		if rnn_net == 'RNN':
			self.hidden = torch.ones(self.n_layers, self.batch_size, self.hidden_dim)
			self.rnn = nn.RNN(input_size = self.input_size, hidden_size = topo[1], num_layers = self.n_layers, batch_first= True)
		if rnn_net == 'LSTM':
			self.hidden = (torch.ones((self.n_layers,self.batch_size,self.hidden_dim)), torch.ones((self.n_layers,self.batch_size,self.hidden_dim)))
			self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = topo[1],num_layers= self.n_layers, batch_first= True)
			# Fully connected layer
		self.fc = nn.Linear(topo[1],topo[2])
		self.topo = topo
		print(rnn_net, ' is rnn net')
	def sigmoid(self, z):
		return 1/(1+torch.exp(-z))
			 

	def forward(self, x):
		# print(x.shape)
		outmain=torch.zeros((len(x),self.topo[2],1))
		for i,sample in enumerate(x):
			sample = torch.FloatTensor(sample)
			# sequence length = 1
			sample = sample.view(1,self.input_seq_len, self.input_size)
			# sample = sample.view(sample.shape[0],1,self.topo[0])
			hidden = copy.deepcopy(self.hidden)
			if self.rnn_net == 'LSTM':
				out, (h1,c1) = self.rnn(sample, hidden)
			elif self.rnn_net == 'RNN':
				out, h1 = self.rnn(sample, hidden)
			h1 = h1.view(-1, self.hidden_dim)
			output = self.fc(h1)
			output = self.sigmoid(output)
			output = output.reshape(outmain[i].shape)
			# print(",out.shape)
			outmain[i,] = copy.deepcopy(output.detach())
			# outmain[i,] = copy.deepcopy(output)
		# print("shape of outmain", np.squeeze(outmain,axis = -1).shape)
		# print("shape of x",x.shape)
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
	def evaluate_proposal(self,x,w):
		# if w is None:
		# 	y_pred = self.forward(x)
		# 	return copy.deepcopy(y_pred.detach().numpy())
		# else:
		# print(f"param given into evaluate proposal: {w['rnn.weight_ih_l0'][:4].T}")
		
		self.loadparameters(w)
		# print(f"param of model class: {self.state_dict()}")
		y_pred = self.forward(x)


		return copy.deepcopy(np.array(y_pred.detach()))

	def langevin_gradient(self,x,y,w,optimizer):
		try:
			self.loadparameters(torch.load('parameters.pt'))
			# self.loadparameters(w)
		except:
			self.loadparameters(w)
		criterion = torch.nn.MSELoss()
		# if self.optimizer == 'SGD':
		# 	optimizer = torch.optim.SGD(self.parameters(),lr = self.lrate)
		# elif self.optimizer == 'Adam':
		# 	optimizer = torch.optim.Adam(self.parameters(),lr = self.lrate)
		# optimizer.zero_grad()
		# outmain = self.forward(x)
		# print(outmain.shape)
		# # outmain=torch.zeros((len(x),self.topo[2],1))
		# # loss = criterion(outmain, torch.FloatTensor(y.view(outmain.shape)))
		# print(y.shape)
		# loss = criterion(outmain, torch.FloatTensor(y))
		# loss.backward()
		# optimizer.step()
		# output = copy.deepcopy(outmain.detach())
		for k in range(2):#computing gradients 5 times 
			outmain=torch.zeros((len(x),self.topo[2],1))
			for i,sample in enumerate(x):
				optimizer.zero_grad()
				sample = torch.FloatTensor(sample)
				sample = sample.view(1,self.input_seq_len,self.input_size)
				hidden = copy.deepcopy(self.hidden)
				
				if self.rnn_net == 'LSTM':
					out, (h1,c1) = self.rnn(sample, hidden)
				elif self.rnn_net == 'RNN':
					out, h1 = self.rnn(sample, hidden)
				h1 = h1.view(-1, self.hidden_dim)
				output = self.fc(h1)
				output = self.sigmoid(output)

				loss = criterion(output,torch.FloatTensor(y[i]).view(output.shape))
				loss.backward()
				optimizer.step()
				output = output.reshape(outmain[i].shape)
				outmain[i,] = copy.deepcopy(output.detach())

		loss_val = criterion(outmain,torch.FloatTensor(y))
		print(f'root of MSEloss for k = {k}: {np.sqrt(loss_val.detach().numpy())}')


		parameters = copy.deepcopy(self.state_dict())
		
		actual = y
		pred = outmain
		# print(actual.shape)
		# print(outmain.shape)
		if len(actual.shape)>2 and actual.shape[2]==1: actual = np.squeeze(actual,axis = 2)
		if len(pred.shape) > 2 and pred.shape[2]==1: pred = np.squeeze(pred.numpy(), axis = 2)
		step_wise_rmse = np.sqrt(np.mean((pred - actual)**2, axis = 0, keepdims= True))
		new_rmse = np.mean(step_wise_rmse)
		print(f'rmse as calculated in mcmc.rmse():{new_rmse}')

		torch.save(parameters,'parameters.pt')
		return parameters
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


# mcmc = MCMC(use_langevin_gradients, l_prob = langevin_prob,
# 					learn_rate = learn_rate, samples= NumSample,trainx= train_x, 
# 					trainy= train_y, testx= test_x, testy= test_y,
# 					topology= topology, )

class MCMC:
	def __init__(self,  use_langevin_gradients , l_prob,  learn_rate,  samples, trainx, trainy, testx, testy, topology,path, rnn_net = 'LSTM', optimizer = 'Adam'):
		self.samples = samples  # NN topology [input, hidden, output]
		self.topology = topology  # max epocs
		self.train_x = trainx
		self.train_y = trainy
		self.test_x = testx
		self.test_y = testy
		self.rnn_net = rnn_net
		self.path = path
		self.optimizer = optimizer
		self.batch_size = {
			'train_x': trainx.shape,
			'train_y': trainy.shape,
			'test_x' : testx.shape,
			'test_y' : testy.shape,
		}
		self.rnn = Model(topology,learn_rate,batch_size= self.batch_size, input_size= 1,rnn_net=self.rnn_net, optimizer = self.optimizer)
		self.use_langevin_gradients  =  use_langevin_gradients 

		self.l_prob = l_prob # likelihood prob

		self.learn_rate =  learn_rate
		# ----------------

	def rmse(self, pred, actual):
		if len(actual.shape)>2 and actual.shape[2]==1: actual = np.squeeze(actual,axis = 2)
		if len(pred.shape) > 2 and pred.shape[2]==1: pred = np.squeeze(pred, axis = 2)
		step_wise_rmse = np.sqrt(np.mean((pred - actual)**2, axis = 0, keepdims= True))
		new_rmse = np.mean(step_wise_rmse)
		return new_rmse

	def step_wise_rmse(self, pred, actual):
		if len(actual.shape)>2 and actual.shape[2]==1: actual = np.squeeze(actual,axis = 2)
		if len(pred.shape) > 2 and pred.shape[2]==1: pred = np.squeeze(pred, axis = 2)
		ans = np.sqrt(np.mean((pred - actual)**2, axis = 0, keepdims= True))
		return ans


	def likelihood_func(self, rnn,x,y, w, tau_sq, temp):
		# fx = rnn.evaluate_proposal(x,w)
		rnn.loadparameters(w)
		fx = copy.deepcopy(np.array(rnn.forward(x).detach()))
		# print(fx.shape)
		# print(y.shape)
		
		rmse = self.rmse(fx, y)
		print(f'rmse as calculated in likelihood func:{rmse}')
		step_wise_rmse = self.step_wise_rmse(fx,y) 
		n = y.shape[0] * y.shape[1]
		p1 = -(n/2)*np.log(2*math.pi*tau_sq) 
		p2 = (1/(2*tau_sq)) 
		log_lhood = p1 -  (p2 * np.sum(np.square(y -fx)) )
		pseudo_loglhood =  log_lhood/temp       
		return [pseudo_loglhood, fx, rmse, step_wise_rmse]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2, w,  tausq,rnn):  #leave this tausq as it is
		h = self.topology[1]  # number hidden neurons
		d = self.topology[0]  # number input neurons
		part1 = -1 * ((len(rnn.getparameters(w))) / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(rnn.getparameters(w))))
		log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
		return log_loss


	def sampler(self):
		samples = self.samples
		x_test = self.test_x
		x_train = self.train_x
		y_train = self.train_y
		y_test = self.test_y
		learn_rate = self.learn_rate
		rnn = self.rnn
		w_size = sum(p.numel() for p in rnn.parameters())
		pos_w = np.ones((samples, w_size)) #Posterior for all weights
		step_wise_rmse_train = np.zeros(shape = (samples,10))
		step_wise_rmse_test  = np.zeros(shape = (samples,10))
		rmse_train  = np.zeros(samples)
		rmse_test = np.zeros(samples)

		w = copy.deepcopy(rnn.state_dict())
		eta = 0 #Junk variable
		step_w = 0.025
		step_eta = 0.2
		#Declare RNN
		pred_train  = rnn.evaluate_proposal(x_train,w) #
		pred_test  = rnn.evaluate_proposal(x_test, w) #
		print(pred_train.shape)
		print(y_train.shape)
		eta = np.log(np.var(pred_train - np.array(y_train)))
		tau_pro = np.exp(eta)


		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0
		sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
		np.fill_diagonal(sigma_diagmat, step_w**2)
		#delta_likelihood = 0.5 # an arbitrary position
		prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro, rnn)  # takes care of the gradients
		# print("prior current shape ", prior_current)
		[likelihood, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(rnn, self.train_x, y_train, w, tau_pro, temp= 1)
		[_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(rnn, self.test_x, y_test, w, tau_pro, temp= 1)
		prop_list = np.zeros((samples,w_size))
		likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
		likeh_list[0,:] = [-100, -100] # to avoid prob in calc of 5th and 95th percentile later
		surg_likeh_list = np.zeros((samples,2))
		accept_list = np.zeros(samples)
		num_accepted = 0
		langevin_count = 0
		pt_samples = samples * 0.6 # this means that PT in canonical form with adaptive temp will work till pt  samples are reached
		init_count = 0

		#initializing optimizer:
		if self.optimizer == 'SGD':
			optimizer = torch.optim.SGD(rnn.parameters(),lr= self.learn_rate)
		elif self.optimizer == 'Adam':
			optimizer = torch.optim.Adam(rnn.parameters(),lr = self.learn_rate)
		
		i=0
		# print(f'weights before loop: {w["rnn.weight_ih_l0"][:4].T}')
		for i in range(samples-1):  # Begin sampling --------------------------------------------------------------------------
			timer1 = time.time()
			lx = np.random.uniform(0,1,1)
			# print(w['rnn.weight_ih_l0'][:4].T)
			# old_w = rnn.state_dict()
			if (self.use_langevin_gradients is True) and (lx< self.l_prob):
				w_gd = rnn.langevin_gradient(self.train_x,self.train_y, copy.deepcopy(w),optimizer= optimizer ) # Eq 8
				# w_gd = torch.load('parameters.pt')
				w_proposal = rnn.addnoiseandcopy(w_gd,0,step_w) #np.random.normal(w_gd, step_w, w_size) # Eq 7
				w_prop_gd = rnn.langevin_gradient(self.train_x,self.train_y, copy.deepcopy(w_proposal),optimizer= optimizer)
				# w_prop_gd = torch.load('parameters.pt')
				#first = np.log(multivariate_normal.pdf(w , w_prop_gd , sigma_diagmat))
				#second = np.log(multivariate_normal.pdf(w_proposal , w_gd , sigma_diagmat)) # this gives numerical instability - hence we give a simple implementation next that takes out log
				wc_delta = (rnn.getparameters(w)- rnn.getparameters(w_prop_gd))
				wp_delta = (rnn.getparameters(w_proposal) - rnn.getparameters(w_gd))
				sigma_sq = step_w ** 2   #should this value be squared or not? squared is fine
				first = -0.5 * np.sum(wc_delta  *  wc_delta  ) / sigma_sq  # this is wc_delta.T  *  wc_delta /sigma_sq
				second = -0.5 * np.sum(wp_delta * wp_delta ) / sigma_sq
				diff_prop =  first - second
				langevin_count = langevin_count + 1
			else:
				diff_prop = 0
				w_proposal = rnn.addnoiseandcopy(w,0,step_w) #np.random.normal(w, step_w, w_size)
			

			# print(f'w_proposal: {w_proposal["rnn.weight_ih_l0"][:4].T}')

			eta_pro = eta + np.random.normal(0, step_eta, 1)
			tau_pro = np.exp(eta_pro)

			[likelihood_proposal, pred_train, rmsetrain, step_wise_rmsetrain] = self.likelihood_func(rnn, self.train_x, y_train, w_proposal,tau_pro, temp= 1)
			[_, pred_test, rmsetest, step_wise_rmsetest] = self.likelihood_func(rnn, self.test_x, y_test, w_proposal,tau_pro, temp= 1)
				
			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,tau_pro, rnn)  # takes care of the gradients
			diff_prior = prior_prop - prior_current
			diff_likelihood = likelihood_proposal - likelihood
		   
			try:
				mh_prob = diff_likelihood + diff_prior + diff_prop
			except OverFlowError as e:
				mh_prob = 1

			accept_list[i+1] = num_accepted
			# if (i % batch_save+1) == 0: # just for saving posterior to file - work on this later
			# 	x = 0
			u = np.log(random.uniform(0, 1))
			# u = random.uniform(0,1)

			prop_list[i+1,] = rnn.getparameters(w_proposal).reshape(-1)
			likeh_list[i+1,0] = likelihood_proposal
			if u < mh_prob:
				num_accepted  =  num_accepted + 1
				likelihood = likelihood_proposal
				prior_current = prior_prop
				w = copy.deepcopy(torch.load('parameters.pt'))
				eta = eta_pro
				# acc_train[i+1,] = 0
				# acc_test[i+1,] = 0
				print(i,'RMSE Train: ',rmsetrain,"RMSE Test: ", rmsetest,' accepted')
				pos_w[i+ 1,] = rnn.getparameters(w_proposal).reshape(-1)
				rmse_train[i + 1,] = rmsetrain
				step_wise_rmse_train[i+1,] = step_wise_rmsetrain
				rmse_test[i + 1,] = rmsetest
				step_wise_rmse_test[i+1,] = step_wise_rmsetest  

				# print(f"if accepted w becomes: {w['rnn.weight_ih_l0'][:4].T}")        
			else:
				# w = old_w
				pos_w[i+1,] = pos_w[i,]
				rmse_train[i + 1,] = rmse_train[i,]
				step_wise_rmse_train[i+1,] = step_wise_rmse_train[i,]
				rmse_test[i + 1,] = rmse_test[i,]
				step_wise_rmse_test[i+1,] = step_wise_rmse_test[i,]
				# print(f"if rejected w becomes: {w['rnn.weight_ih_l0'][:4].T}")
				# acc_train[i+1,] = acc_train[i,]
				# acc_test[i+1,] = acc_test[i,]
		path = self.path
		directories = [  path+'/predictions/', path+'/posterior', path+'/results', #path+'/surrogate', 
							# path+'/surrogate/learnsurrogate_data', 
							path+'/posterior/pos_w',  path+'/posterior/pos_likelihood',
							path+'/posterior/accept_list'  ] #path+'/posterior/surg_likelihood',
		for d in directories:
			if not os.path.exists(d):
				os.makedirs(d)

		print ((num_accepted*100 / (samples * 1.0)), f'% samples out of {samples} were accepted')
		accept_ratio = num_accepted / (samples * 1.0) * 100
		print ((langevin_count*100 / (samples * 1.0)), '% of total samples were Langevin')
		langevin_ratio = langevin_count / (samples * 1.0) * 100
		file_name = self.path+'/posterior/pos_w/'+'single_chain' + '.txt'
		np.savetxt(file_name,pos_w )
		file_name = self.path+'/predictions/rmse_test_single_chain' + '.txt'
		np.savetxt(file_name, rmse_test, fmt='%1.8f')
		file_name = self.path+'/predictions/rmse_train_single_chain' + '.txt'
		np.savetxt(file_name, rmse_train, fmt='%1.8f')
		file_name = self.path+'/predictions/stepwise_rmse_train_single_chain'  + '.txt'
		np.savetxt(file_name, step_wise_rmse_train, fmt = '%1.8f')
		file_name = self.path + '/predictions/stepwise_rmse_test_csingle_chain'  + '.txt'
		np.savetxt(file_name, step_wise_rmse_test, fmt = '%1.8f')
		file_name = self.path+'/posterior/pos_likelihood/single_chain' + '.txt'
		np.savetxt(file_name,likeh_list, fmt='%1.4f')
		file_name = self.path + '/posterior/accept_list/csingle_chain'  + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.4f')
		file_name = self.path + '/posterior/accept_list/csingle_chain'  + '.txt'
		np.savetxt(file_name, accept_list, fmt='%1.4f')
		file_name = self.path + '/predictions/final_pred_train_csingle_chain'  + '.txt'
		np.savetxt(file_name, np.squeeze(pred_train), fmt='%1.4f')
		file_name = self.path + '/predictions/final_pred_test_csingle_chain'  + '.txt'
		np.savetxt(file_name, np.squeeze(pred_test), fmt= '%1.4f')


		return(pos_w, rmse_train, step_wise_rmse_train, rmse_test, step_wise_rmse_test ,accept_list, accept_ratio, likeh_list)            


def main():
	n_steps_in, n_steps_out = 5,10
	net = 'RNN' if args.net == 1 else 'LSTM' 
	optimizer = 'SGD' if args.optimizer == 1 else 'Adam'
	print(f"Network is {net} and optimizer is {optimizer}")
	print("Name of folder to look for: ",os.getcwd()+'/Res_LG-Lprob_'+net+f'_{optimizer}_single_chain/')

	
	for j in [1] :
	# for j in [2]:
		print(j, ' out of 15','\n\n\n')
		#i = j//2
		problem=j
		folder = ".."
		if problem ==1:
			TrainData = pd.read_csv("./data/Lazer/train1.csv",index_col = 0)
			TrainData = TrainData.values
			TestData = pd.read_csv("./data/Lazer/test1.csv",index_col = 0)
			TestData = TestData.values
			name= "Lazer"
		if problem ==2:
			TrainData = pd.read_csv("./data/Sunspot/train1.csv",index_col = 0)
			TrainData = TrainData.values
			TestData = pd.read_csv("./data/Sunspot/test1.csv",index_col = 0)
			TestData = TestData.values
			name= "Sunspot"
		if problem ==3:
			TrainData = pd.read_csv("./data/Mackey/train1.csv",index_col = 0)
			TrainData = TrainData.values
			TestData = pd.read_csv("./data/Mackey/test1.csv",index_col = 0)
			TestData = TestData.values
			name="Mackey"
		if problem ==4:
			TrainData = pd.read_csv("./data/Lorenz/train1.csv",index_col = 0)
			TrainData = TrainData.values
			TestData = pd.read_csv("./data/Lorenz/test1.csv",index_col = 0)
			TestData = TestData.values
			name= "Lorenz"
		if problem ==5:
			TrainData = pd.read_csv("./data/Rossler/train1.csv",index_col = 0)
			TrainData = TrainData.values
			TestData = pd.read_csv("./data/Rossler/test1.csv",index_col = 0)
			TestData = TestData.values
			name= "Rossler"
		if problem ==6:
			TrainData = pd.read_csv("./data/Henon/train1.csv",index_col = 0)
			TrainData = TrainData.values
			TestData = pd.read_csv("./data/Henon/test1.csv",index_col = 0)
			TestData = TestData.values
			name= "Henon"
		if problem ==7:
			TrainData = pd.read_csv("./data/ACFinance/train1.csv",index_col = 0)
			TrainData = TrainData.values
			TestData = pd.read_csv("./data/ACFinance/test1.csv",index_col = 0)
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

		Hidden = 10 #originally it was 5; but in the paper to compare it is 10
		topology = [n_steps_in, Hidden,n_steps_out]
		NumSample = args.samples
		netw = topology
		
				
		burn_in = args.burn_in
		learn_rate = args.learn_rate  # in case langevin gradients are used. Can select other values, we found small value is ok.
		langevn = ""
		#if j%2 == 1:
		use_langevin_gradients = True  # False leaves it as Random-walk proposals. Note that Langevin gradients will take a bit more time computationally
		#langevn = "T"
		#else:
		#   use_langevin_gradients = False
		#   langevn = "F"
		#   pass # we dont want to execute this.
		print(f'langevin is {use_langevin_gradients}')
		problemfolder = os.getcwd()+'/Res_LG-Lprob_'+net+f'_{optimizer}_single_chain/'  #'/home/rohit/Desktop/PT/Res_LG-Lprob/'  # change this to your directory for results output - produces large datasets
		problemfolder_db = 'Res_LG-Lprob_'+net+ f'_{optimizer}_single_chain/'  # save main results
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
		# resultingfile = open( path+'/master_result_file.txt','a+')
		# resultingfile_db = open( path_db+'/master_result_file.txt','a+')
		timer = time.time()
		langevin_prob = 0.9

		mcmc = MCMC(use_langevin_gradients, l_prob = langevin_prob,
					learn_rate = learn_rate, samples= NumSample,trainx= train_x, 
					trainy= train_y, testx= test_x, testy= test_y,
					topology= topology,path = path, rnn_net = net, optimizer= optimizer)
		[pos_w, rmse_train, indiv_rmse_train, rmse_test, indiv_rmse_test ,accept_vec, accept_ratio, likelihood] = mcmc.sampler()

		timer2 = time.time()


		pos_w = pos_w[int(burn_in):]
		rmse_train = rmse_train[int(burn_in):]
		rmse_test = rmse_test[int(burn_in):]

		rmse_tr = np.mean(rmse_train[int(burn_in):])
		rmsetr_std = np.std(rmse_train[int(burn_in):])
		rmse_tes = np.mean(rmse_test[int(burn_in):])
		rmsetest_std = np.std(rmse_test[int(burn_in):])
		print(rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)

 


		print(pos_w.shape," is the shape of pos_w")
		plot_fname = path

		if args.DEBUG == 0:
			for s in range(pos_w.shape[0]): # change this if you want to see all pos plots
				plot_figure(pos_w[s,:], 'pos_distri_'+str(s) , path)
			print(' images placed in folder with name: ',path)

			if not os.path.exists(path+ '/trace_plots_better'):
				os.makedirs(path+ '/trace_plots_better')
			for i in range(pos_w.shape[0]):
				histogram_trace(pos_w[i,:], path+ '/trace_plots_better/'+ str(i))

		
		
		
		accept_per = accept_ratio
		print(accept_per, ' accept_percentage')
		# timer2 = time.time()
		timetotal = (timer2 - timer) /60
		print ((timetotal), 'min taken for training')

		step_wise_rmse_train_mean = np.mean(indiv_rmse_train, axis= 0, keepdims= True)
		# print(step_wise_rmse_train_mean.shape)
		step_wise_rmse_train_std = np.std(indiv_rmse_train, axis = 0, keepdims= True)
		step_wise_rmse_train_best = np.amin(indiv_rmse_train, axis = 0, keepdims= True)
		rmse_tr = np.mean(rmse_train[:])   
		rmsetr_std = np.std(rmse_train[:])
		rmsetr_max = np.amin(rmse_train[:])
		step_wise_rmse_test_mean = np.mean(indiv_rmse_test, axis = 0, keepdims= True)
		step_wise_rmse_test_std = np.std(indiv_rmse_test, axis = 0, keepdims= True)
		step_wise_rmse_test_best = np.amin(indiv_rmse_test, axis = 0, keepdims= True)
		rmse_tes = np.mean(rmse_test[:])
		rmsetest_std = np.std(rmse_test[:])
		rmsetes_max = np.amin(rmse_test[:])
		outres = open(path+'/results/result.txt', "a+")
		outres_step = open(path + '/results/result_stepwise.txt', 'a+')
		# outres_db = open(path_db+'/result.txt', "a+")
		# outres_step_db = open(path + '/result_stepwise.txt', 'a+')
		resultingfile = open(problemfolder+'/master_result_file.txt','a+')
		resultingfile_step = open(problemfolder + '/master_step_result_file.txt', 'a+')
		#in this file i will store the table after confirming my doubt from ayush
		resultingfile_db = open( problemfolder_db+'/master_result_file.txt','a+')
		xv = name+langevn+'_'+ str(run_nb)
		step_wise_results = np.concatenate(
											(
												step_wise_rmse_train_mean,
												step_wise_rmse_train_std,
												step_wise_rmse_train_best,
												step_wise_rmse_test_mean,
												step_wise_rmse_test_std,
												step_wise_rmse_test_best
											),
											axis= 0
										)
		# print(step_wise_results.shape)
		columns = [f'Step {i}' for i in range(1,11)]
		indices = ['train_mean', 'train_std','train_best', 'test_mean', 'test_std', 'test_best']
		df = pd.DataFrame(step_wise_results, index = indices, columns= columns)
		df['overall'] = [rmse_tr, rmsetr_std, rmsetr_max, rmse_tes, rmsetest_std, rmsetes_max]
		columns = ['overall'] + columns
		df = df.reindex(columns= columns)
		df = df.transpose()

		df.to_csv(path_or_buf= path + '/results/result_stepwise.csv', header = f'{net} {optimizer}')

		np.savetxt(outres_step, step_wise_results, fmt = '%1.4f', newline= ' \n')
		# np.savetxt(outres_step_db, step_wise_results, fmt= '%1.4f', newline = ' \n')
		allres =  np.asarray([ problem, NumSample, langevin_prob, learn_rate,  
								rmse_tr, rmsetr_std, rmsetr_max, rmse_tes, rmsetest_std, rmsetes_max
								, accept_per, timetotal])
		# np.savetxt(outres_db,  allres   , fmt='%1.4f', newline=' '  )
		np.savetxt(resultingfile_db,   allres   , fmt='%1.4f',  newline=' ' )
		np.savetxt(resultingfile_db, [xv]   ,  fmt="%s", newline=' \n' )
		np.savetxt(outres,  allres   , fmt='%1.4f', newline=' '  )
		np.savetxt(resultingfile,   allres   , fmt='%1.4f',  newline=' ' )
		np.savetxt(resultingfile, [xv]   ,  fmt="%s", newline=' \n' )
		x = np.linspace(0, rmse_train.shape[0] , num=rmse_train.shape[0])

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
		# likelihood = likelihood_rep[:,0] # just plot proposed likelihood
		# likelihood = np.asarray(np.split(likelihood, num_chains))
	# Plots
		plt.plot(likelihood.T)
		plt.xlabel('Samples', fontsize=12)
		plt.ylabel(' Log-Likelihood', fontsize=12)
		plt.savefig(path+'/likelihood.png')
		plt.clf()
		# plt.plot(likelihood.T)
		# plt.xlabel('Samples', fontsize=12)
		# plt.ylabel(' Log-Likelihood', fontsize=12)
		# plt.savefig(path_db+'/likelihood.png')
		# plt.clf()
		plt.plot(accept_vec.T )
		plt.xlabel('Samples', fontsize=12)
		plt.ylabel(' Number accepted proposals', fontsize=12)
		plt.savefig(path+'/accept.png')
		plt.clf()
		# plt.plot(accept_vec.T )
		# plt.xlabel('Samples', fontsize=12)
		# plt.ylabel(' Number accepted proposals', fontsize=12)
		# plt.savefig(path_db+'/accept.png')
		# plt.clf()

		gc.collect()
		outres.close()
		outres_step.close()
		# outres_step_db.close()
		resultingfile.close()
		resultingfile_db.close()
		# outres_db.close()

if __name__ == "__main__": main()

# pos_w, fx_train, fx_test, rmse_train, rmse_test, indiv_rmse_train, indiv_rmse_test, likelihood_rep, swap_perc, accept_vec, accept = pt.run_chains()
		