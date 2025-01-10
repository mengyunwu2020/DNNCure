import torch 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def function_g(seed,f, Z, beta):
    torch.manual_seed(seed)
    n,p=Z.shape
    f1_value=f1(Z,f)
    pi = 1-torch.sigmoid(f1_value)#cure_rate
    pi=torch.clamp((pi-torch.mean(pi))/1.1+beta,min=0,max=1)
    # Generate U
    U = torch.rand(n, device=device)
    # Generate delta_c
    delta_c = torch.zeros(n, dtype=torch.float32, device=device)
    delta_c[U > pi] = 1
    # Separate indices for treatment and control groups
    ordt = torch.where(U > pi)[0]
    ordc = torch.where(U <= pi)[0]
    # Generate Tt
    Zt = Z[ordt]
    f2_value = f2(Zt,f)
    return ordc,ordt,pi, f2_value

def f1(Z,f):
    if f=='S4':
        f1_value=0.8*torch.sum(torch.sin(Z[:,0:10]),dim=1)+0.8*Z[:,0]*Z[:,1]+0.8*Z[:,8]*Z[:,9]
    elif f=='S3':
        f1_value=0.8*torch.sum(torch.sin(Z[:,0:10]),dim=1)+0.8*Z[:,8]*Z[:,9]
    elif f=='S2':
        f1_value=0.8*torch.sum(Z[:,0:5],dim=1)+0.8*torch.sum(torch.sin(Z[:,5:10]*1.5),dim=1)
    elif f=='S1':
        f1_value=0.8*torch.sum(Z[:,0:10],dim=1)
    elif f=='SS3':
        f1_value=0.8*torch.sum(torch.sin(Z[:,0:10]),dim=1)
    elif f=='SS2':
        f1_value=0.8*torch.sum(torch.sin(0.7*Z[:,0:10]),dim=1)
    elif f=='SS1':
        f1_value=0.8*torch.sum(torch.sin(Z[:,0:10]),dim=1)
    else:
        print('please give a f')
    return f1_value

def f2(Zt,f):
    if f=='S4':
        f2_value=0.8*torch.sum(torch.sin(Zt[:,0:10]),dim=1)+0.8*Zt[:,0]*Zt[:,1]+0.8*Zt[:,8]*Zt[:,9]
    elif f=='S3':
        f2_value=0.8*torch.sum(torch.sin(Zt[:,0:10]),dim=1)+0.8*Zt[:,8]*Zt[:,9]
    elif f=='S2':
        f2_value=0.8*torch.sum(Zt[:,0:5],dim=1)+0.8*torch.sum(torch.sin(Zt[:,5:10]*1.5),dim=1)
    elif f=='S1':
        f2_value=0.8*torch.sum(Zt[:,0:10],dim=1)
    elif f=='SS3':
        f2_value=0.5*torch.sum(torch.sin(Zt[:,0:10]),dim=1)
    elif f=='SS2':
        f2_value=0.8*torch.sum(torch.sin(Zt[:,0:7]),dim=1)+0.8*torch.sum(torch.sin(Zt[:,10:13]),dim=1)
    elif f=='SS1':
        f2_value=0.8*torch.sum(torch.sin(Zt[:,0:7]),dim=1)

    else :
        print('please give a f')
    return f2_value
