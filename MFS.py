import torch
import numpy as np
from datetime import datetime
from modelsn import Net_nonlinear
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def total_loss(data, model, eta,ll):
    Z, T, delta, tau, d, Rj, idx = data[:]
    k, n, p = len(tau), *Z.shape
    out1,out2=model(Z)
    forward1=out1[:,0]
    forward2=out2[:,0]
    pi=torch.sigmoid(forward1)
    with torch.no_grad():
        osum = torch.stack([torch.sum(eta[Rj[j]] * torch.exp(forward2[Rj[j]])) for j in range(k)])
        threshold1 = torch.quantile(osum, 0)
        threshold2 = torch.quantile(osum, 0.95)
        osum = torch.clamp(osum, min=threshold1, max=threshold2)
        h0 = d / osum
        sumh0 = torch.stack([torch.sum(h0[idx[i]]) for i in range(n)])
        S0 = torch.exp(-sumh0)
        St = S0.pow(torch.exp(forward2))
    #with torch.no_grad():
        eta = (delta + (1 - delta) * pi * St / (1 - pi + pi * St)).clamp(min=0.001)
    los1 = -torch.sum(eta * torch.log(pi) + (1 - eta) * torch.log(1 - pi)) / n
    mitv = torch.stack([(eta[Rj[j]] * torch.exp(forward2[Rj[j]])).sum() for j in range(k)])
    los2 = (torch.sum(d * torch.log(mitv)) - torch.sum(forward2[delta==1])) / n
    phi1=model.hidden0_1.weight
    phi2=model.hidden0_2.weight
    phi1=torch.sigmoid(100*phi1**2)
    phi2=torch.sigmoid(100*phi2**2)
    loss=los1+los2+ll*torch.sum((phi1-phi2)**2)/p
    return eta,loss

def FS_epoch(model, s1,s2, supp_x, data, optimizer, optimizer0_1, optimizer0_2,eta,Ts,step=1,ll=20):
    Z, T, delta, tau, d, Rj, idx=data[:]
    k,n,p=len(tau),*Z.shape
    LOSS = []
    eta,loss=total_loss(data,model,eta,ll)
    #optimizer.zero_grad()
    loss.backward(retain_graph=True)
    #optimizer.step()
    z1 = model.hidden0_1.weight.grad.data
    z1_sort, z1_indices = torch.sort(-torch.abs(z1))
    Z1 = z1_indices[:2*s1].cpu().numpy()
    Z2 = z1_indices[:2*s2].cpu().numpy()
    
    supp_x1,supp_x2=supp_x[0],supp_x[1]
    T1=set(Z1).union(supp_x1)
    T2=set(Z2).union(supp_x2)
    TC1 = np.setdiff1d(np.arange(p), list(T1))
    TC2 = np.setdiff1d(np.arange(p), list(T2))
    for j in range(Ts):
        tmp1 = model.hidden0_1.weight.data
        tmp2 = model.hidden0_2.weight.data
        for _ in range(step):
            eta,loss=total_loss(data,model,eta,ll)
            
            LOSS.append(loss.data.cpu().numpy().tolist())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if (len(TC1|len(TC2) > 0) > 0):
                model.hidden0_1.weight.data[TC1] =tmp1[TC1]
                model.hidden0_2.weight.data[TC2] =tmp2[TC2]
        for _ in range(step):
            eta,loss=total_loss(data,model,eta,ll)
            
            LOSS.append(loss.data.cpu().numpy().tolist())
            optimizer0_1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer0_1.step()
            if (len(TC1) > 0|len(TC2) > 0):
                model.hidden0_1.weight.data[TC1] =tmp1[TC1]
                model.hidden0_2.weight.data[TC2] =tmp2[TC2]
                
        for _ in range(step):
            eta,loss=total_loss(data,model,eta,ll)
            LOSS.append(loss.data.cpu().numpy().tolist())
            optimizer0_2.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer0_2.step()
            if (len(TC1) >0|len(TC2) > 0):
                model.hidden0_2.weight.data[TC1] =tmp2[TC1]
                model.hidden0_1.weight.data[TC2] =tmp1[TC2]

    if (len(TC1) > 0|len(TC2) > 0):
        model.hidden0_1.weight.data[TC1] = 0
        model.hidden0_2.weight.data[TC2] = 0
### Find the w's with the largest magnitude
#net1
    w1 = model.hidden0_1.weight.data
    w1_sort, w1_indices = torch.sort(-torch.abs(w1))
    supp_x1 = w1_indices[:s1].cpu().numpy()
#net2
    w2 = model.hidden0_2.weight.data
    w2_sort, w2_indices = torch.sort(-torch.abs(w2))
    supp_x2 = w2_indices[:s2].cpu().numpy()
    print('L-part:',sorted(supp_x1),'C-part:',sorted(supp_x2))
    supp_x1_c=np.setdiff1d(range(p),supp_x1)
    supp_x2_c=np.setdiff1d(range(p),supp_x2)
    model.hidden0_1.weight.data[supp_x1_c]=0
    model.hidden0_2.weight.data[supp_x2_c]=0
    model.hidden0_1.weight.data.cpu().numpy()
    model.hidden0_2.weight.data.cpu().numpy()
    with torch.no_grad():
        eta[eta<0.001]=0.001
    return model,[supp_x1,supp_x2], LOSS,eta





def training_n(data_train, s1,s2, eta,epochs=10, n_hidden1=50, n_hidden2=10, learning_rate=0.0005, Ts=25, step=5,ll=20):
    Z, T, delta, tau, d, Rj, idx=data_train[:]
    n,p=Z.shape
    torch.manual_seed(1)
    model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2,n_output=1).to(device=device)
    best_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1).to(device=device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=0.0025)
    optimizer0_1 = torch.optim.Adam(model.hidden0_1.parameters(), lr=learning_rate, weight_decay=0.0005)
    optimizer0_2 = torch.optim.Adam(model.hidden0_2.parameters(), lr=learning_rate, weight_decay=0.0005)
    hist=[]
    SUPP1=[]
    SUPP2=[]
    LOSSES=[]
    supp_x1=supp_x2 = list(range(p)) # initial support
    supp_x=[supp_x1,supp_x2]
    SUPP1.append(supp_x1)
    SUPP2.append(supp_x2)
    for i in range(epochs):
        print('epoch:',i)
        # One DFS epoch
        model, supp_x, LOSS,eta=FS_epoch(model, s1,s2, supp_x,data_train, optimizer, optimizer0_1, optimizer0_2,eta, Ts, step,ll)
        # supp_x.sort()
        _,loss=total_loss (data_train,model,eta,ll)
        hist.append(loss.data.cpu().numpy().tolist())
        SUPP1.append(supp_x[0])
        SUPP2.append(supp_x[1])
        
        # Prevent divergence of optimization over support, save the current best model
        if hist[-1] == min(hist):
            best_model.load_state_dict(model.state_dict())
            best_supp = supp_x
            #print(best_supp)
        #Early stop criteria
        if ((len(SUPP1[-1])==len(SUPP1[-2])) & (len(SUPP2[-1])==len(SUPP2[-2]))):

            if((set(SUPP1[-1])==set(SUPP1[-2])) & (set(SUPP2[-1])==set(SUPP2[-2]))) :
                break
    print('L-part:',sorted(best_supp[0]),'C-part:',sorted(best_supp[1]))
    # TPR_train,TFR_train=TFPR(best_supp,s,p)
    _optimizer=torch.optim.Adam(list(best_model.parameters()), lr=learning_rate, weight_decay=0.0025)
    for _ in range(50):
        _,loss=total_loss (data_train,best_model,eta,ll)
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()
    print(loss)
    bic=(loss.data.cpu().numpy().tolist()) +0.3*(s1+s2)*np.log(n)/n
    return loss.data.cpu().numpy().tolist(),best_model,best_supp,bic
     
