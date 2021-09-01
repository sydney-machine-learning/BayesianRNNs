import torch.nn as nn
import torch
import copy
import numpy as np

class Model(nn.Module):
        # Defining input size, hidden layer size, output size and batch size respectively
    def __init__(self, topo,lrate,batch_size = 1, input_size = 1,rnn_net = 'RNN', optimizer = 'SGD'):
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
        # if rnn_net == 'GRU':
        #     self.hidden = torch.ones((self.n_layers,self.batch_size,self.hidden_dim))
        #     self.rnn = nn.GRU(input_size = topo[0], hidden_size = topo[1])
        if rnn_net == 'LSTM':
            self.hidden = (torch.ones((self.n_layers,self.batch_size,self.hidden_dim)), torch.ones((self.n_layers,self.batch_size,self.hidden_dim)))
            self.rnn = nn.LSTM(input_size = self.input_size, hidden_size = topo[1],num_layers= self.n_layers, batch_first= True)
            # Fully connected layer
        self.fc = nn.Linear(topo[1],topo[2])
        self.topo = topo
        print(rnn_net, ' is rnn net')
    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))


    # forward()
    # input : batchsize, sequence length, input_size   Output: batchsize, sequence length, hidden size
    # 100, 5, 1 --for univariate dataset
    # 100, 5, 2 --for bivariate
    # 100, 5, n --for multivariate

    # trying out the vectorization approach, hidden cell gives trouble
    # def forward(self, x):
    #     outmain = torch.zeros((len(x), self.topo[2], 1))
    #     xdata = torch.FloatTensor(x)
    #     xdata = xdata.view(len(x),self.topo[0],self.input_size)
    #     # print('shape of xdata: ',xdata.shape) #correct and == [585,5,1]
    #     hidden = copy.deepcopy(self.hidden)
    #     if self.rnn_net == 'LSTM':
    #         out, (h_out, _)= self.rnn(xdata, hidden)
    #     elif self.rnn_net == 'RNN':
    #         out, h_out = self.rnn(xdata,hidden)
    #     print("shape of output: ",out.shape,"\n shape of h_out: ",h_out.shape)
    #     h_out = h_out.view(-1, self.hidden_size)
    #     out = self.fc(h_out)
    #     return out                   

    def forward(self, x):
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
        # print("shape of outmain", np.squeeze(outmain,axis = -1).shape)
        # print("shape of x",x.shape)
        return copy.deepcopy(np.squeeze(outmain, axis = -1))

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
        self.loadparameters(w)
        criterion = torch.nn.MSELoss()
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),lr = self.lrate)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),lr = self.lrate)
        for i,sample in enumerate(x):
            sample = torch.FloatTensor(sample)
            sample = sample.view(1,self.input_seq_len,self.input_size)
            hidden = copy.deepcopy(self.hidden)
            optimizer.zero_grad()
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
