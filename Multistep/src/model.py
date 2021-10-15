import torch.nn as nn
import torch
import copy
import numpy as np

class Model(nn.Module):
	def __init__(self, topo, lrate, input_size = 1, rnn_net = 'LSTM', optimizer = 'Adam'):
		"""

		"""
		super(Model, self).__init__()
		self.hidden_dim = topo[1]#hidden_dim
		self.input_seq_len = topo[0]
		self.num_outputs = topo[2]
		self.input_size = input_size
		self.n_layers = 1   #len(topo)-2 #n_layers
		# self.batch_size = 1 # no longer required since we are now useing the vectorized implementation
		self.lrate = lrate
		self.optimizer = optimizer
		self.rnn_net = rnn_net
		if rnn_net == 'RNN':
			# self.hidden = torch.ones(self.n_layers, self.batch_size, self.hidden_dim)
			self.rnn = nn.RNN(input_size = self.input_size, hidden_size = topo[1], num_layers = self.n_layers, batch_first= True)
		if rnn_net == 'LSTM':
			# self.hidden = (torch.ones((self.n_layers,self.batch_size,self.hidden_dim)), torch.ones((self.n_layers,self.batch_size,self.hidden_dim)))
			self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = topo[1],num_layers= self.n_layers, batch_first= True)
			# Fully connected layer
		self.fc = nn.Linear(topo[1],topo[2])
		self.topo = topo
		print('Log::Model class init:: ', rnn_net, ' is rnn net')



	def sigmoid(self,z):
		return torch.nn.Sigmoid(z)

	def forward(self, x):
		h_0 = Variable(torch.zeros(self.n_layers, x.shape[0], self.hidden_dim))
		c_0 = Variable(torch.zeros(self.n_layers, x.shape[0], self.hidden_dim))

		batch_size = len(x)
		output_size = self.topo[2]
		# outmain = torch.zeros((batch_size, output_size, 1))
		
		if(self.rnn_net == 'RNN'):
			ula, h_out = self.rnn(x, h_0)
		elif(self.rnn_net == 'LSTM'):
			# ula, (h_out, _) = self.rnn(x,(h_0, c_0))
			ula, (h_out, _) = self.rnn(x)
			h_out = h_out.view(-1, self.hidden_dim)
			
		out = self.fc(h_out)
		out = out.view(x.shape[0], self.num_outputs, 1)
		return out #returning the output tensor directly without detaching or deep copying
		

	def langevin_gradient(self, x, y, w, optimizer):
		self.load_state_dict(w)

		criterion = torch.nn.MSELoss()
		for k in range(1):
			output = self.forward(x)
			optimizer.zero_grad()
			loss = criterion(output, y.view(output.shape))

			loss.backward()
			optimizer.step()

			# print(f'root of MSEloss for k = {k}: {np.sqrt(loss.detach().numpy())}')

		parameters = self.state_dict()
		return parameters
	
	def dictfromlist(self, param):
		dic = {}
		i = 0
		for name in sorted(self.state_dict().keys()):
			dic[name] = torch.FloatTensor(param[i:i+(self.state_dict()[name]).view(-1).shape[0]]).view(self.state_dict()[name].shape)
			i += (self.state_dict()[name]).view(-1).shape[0]
		#self.loadparameters(dic)
		return dic


	def evaluate_proposal(self, x, w= None):
		backup = copy.deepcopy(self.state_dict())
		self.load_state_dict(w)
		y_pred = self.forward(x)
		self.load_state_dict(backup)
		return copy.deepcopy(y_pred.detach().numpy())


	def addnoiseandcopy(self, w, mean, std_dev):
		d = dict()
		for name in w.keys():
			d[name] = copy.deepcopy(w[name]) + torch.zeros(w[name].size()).normal_(mean = mean, std = std_dev)
		return d

	def getparameters(self, w= None):
		l = np.array([1,2])
		dic = {}
		if w is None: 
			dic = self.state_dict()
		else:
			dic = copy.deepcopy(w)
		for name in sorted(dic.keys()):
			l = np.concatenate((l, np.array(copy.deepcopy(dic[name])).reshape(-1)),axis= None)
		l = l[2:]
		return l