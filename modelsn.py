import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utilsn import WeightNorm, DotProduct1,DotProduct2
import torch
import numpy as np
np.random.seed(1)

class Net_nonlinear(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net_nonlinear, self).__init__()
        self.hidden0_1 = DotProduct1(n_feature)  # 
        self.hidden0_2 = DotProduct2(n_feature)  # 
        self.hidden1_1 = nn.Linear(n_feature, n_hidden1)  # Hidden layer 1
        self.hidden1_2 = nn.Linear(n_feature, n_hidden1)  # Hidden layer 1
        self.hidden2_1 = nn.Linear(n_hidden1, n_hidden2)  # Hidden layer 2
        self.hidden2_2 = nn.Linear(n_hidden1, n_hidden2)  # Hidden layer 2
        self.out1 = nn.Linear(n_hidden2, n_output)
        self.out2 = nn.Linear(n_hidden2, n_output)


    def forward(self, x):
        net1 = self.hidden0_1(x)
        net1 = F.relu(self.hidden1_1(net1))
        net1 = F.relu(self.hidden2_1(net1))
        net1 = self.out1(net1)

        net2 = self.hidden0_2(x)
        net2 = F.relu(self.hidden1_2(net2))
        net2 = F.relu(self.hidden2_2(net2))
        net2 = self.out2(net2)
        return net1, net2
class Net_linear(nn.Module):
    def __init__(self, n_feature, n_hidden1,n_output):
        super(Net_linear, self).__init__()
        self.hidden0_1 = DotProduct1(n_feature)  # 
        self.hidden0_2 = DotProduct2(n_feature)  # 
        self.hidden1_1 = nn.Linear(n_feature, n_hidden1)  # Hidden layer 1
        self.hidden1_2 = nn.Linear(n_feature, n_hidden1)  # Hidden layer 1
        self.out1 = nn.Linear(n_hidden1, n_output)
        self.out2 = nn.Linear(n_hidden1, n_output)

    def forward(self, x):
        net1 = self.hidden0_1(x)
        net1 = F.relu(self.hidden1_1(net1))
        net1 = self.out1(net1)

        net2 = self.hidden0_2(x)
        net2 = F.relu(self.hidden1_2(net2))
        net2 = self.out2(net2)
        return net1, net2



class Vor_model(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Vor_model, self).__init__()
        self.hidden0 = DotProduct1(n_feature)  #  
        self.hidden1 = nn.Linear(n_feature, n_hidden1)  # Hidden layer 1
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)  # Hidden layer 2
        self.out1 = nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        net1 = self.hidden0(x)
        net1 = F.relu(self.hidden1(net1))
        net1 = F.relu(self.hidden2(net1))
        # net1 = F.relu(self.hidden3(net1))
        net1 = self.out1(net1)
        return net1 
    
class DNN_model(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(DNN_model, self).__init__() 
        self.hidden1 = nn.Linear(n_feature, n_hidden1)  # Hidden layer 1
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)  # Hidden layer 2
        self.out1 = nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        net1 = F.relu(self.hidden1(x))
        net1 = F.relu(self.hidden2(net1))
        net1 = self.out1(net1)
        return net1