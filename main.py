from modelsn import Net_nonlinear,Net_linear
import torch
from torch import nn
import torch.optim as optim
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def initialize_model(p,n_hidden1,n_hidden2,learning_rate):
    model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2,n_output=1).to(device=device)
    best_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2,n_output=1).to(device=device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=0.0025)
    optimizer0_1 = torch.optim.Adam(model.hidden0_1.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizer0_2 = torch.optim.Adam(model.hidden0_2.parameters(), lr=learning_rate, weight_decay=0.0005)
    return model,best_model,optimizer,optimizer0_1,optimizer0_2
def initialize_linear(p,n_hidden1,learning_rate):
    model = Net_linear(n_feature=p, n_hidden1=n_hidden1,n_output=1).to(device=device)
    best_model = Net_linear(n_feature=p, n_hidden1=n_hidden1,n_output=1).to(device=device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=0.0025)
    optimizer0_1 = torch.optim.Adam(model.hidden0_1.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizer0_2 = torch.optim.Adam(model.hidden0_2.parameters(), lr=learning_rate, weight_decay=0.0005)
    return model,best_model,optimizer,optimizer0_1,optimizer0_2
def metric(correct_set,best_supp,seed, f, alpha,beta,n,p,pra='u'):
    s=len(correct_set)
    setA=set(correct_set)
    setB = set(best_supp)
    intersection = setA.intersection(setB)
    num_common_elements = len(intersection)
    TPR=num_common_elements/s
    FPR=(len(setB)-num_common_elements)/(p-s)
    return(TPR,FPR)
