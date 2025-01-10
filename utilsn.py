import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


np.random.seed(1)
### Weight Normalization ###
############################
# Weight Normalization for Neural Network models
class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'
    
    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()
    
    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            
            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            
            # remove w from parameter list
            del self.module._parameters[name_w]
            
            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)
    
    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)
    
    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

#########################################
### Hadamard product, for first layer ###
#########################################
class DotProduct2(torch.nn.Module):
    def __init__(self, in_features):
        super(DotProduct2, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight.data.normal_(0, stdv)
    def forward(self, input):
        output_np = torch.mul(input, self.weight.expand_as(input))
        return output_np
    def __ref__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'
class DotProduct1(torch.nn.Module):
    def __init__(self, in_features):
        super(DotProduct1, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.weight = Parameter(torch.Tensor(in_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        #self.weight.data.normal_(0, stdv)
    def forward(self, input):
        output_np = torch.mul(input, self.weight.expand_as(input))
        return output_np
    def __ref__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'

