import torch
import numpy as np
from numpy.random import gamma
from numpy.random import exponential
from function import function_g
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rho = 0.5

# Generate covariance matrix

def generate_Z(seed,n,p):
    np.random.seed(seed)
    mean = np.zeros(p)
    covariance_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            covariance_matrix[i,j] = rho**abs(i-j)
    #covariance_matrix = np.eye(p)
    Z = torch.tensor(data=np.random.multivariate_normal(mean, covariance_matrix, n), 
                        dtype=torch.float32, 
                        device=device)
    return Z



def generate_data(device,seed, f, Z,n=1000, p=300, alpha=0.05, beta=0.3,cox=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    ordc,ordt,pi, f2_value =function_g(seed,f, Z, beta)
    Tt=torch.exp(f2_value)
    # Generate C
    qt=1.01-alpha
    temp = torch.quantile(Tt,qt)
    gamma_dist =torch.distributions.Gamma(temp,1)
    C = gamma_dist.sample((n,))
    C.to(device=device)
    # Generate Y
    Y = torch.zeros(n, dtype=torch.float32, device=device)
    Y[ordc] = 10000
    Y[ordt] = Tt
    # Generate delta
    ###uncure_point
    y_cure=torch.zeros(n, dtype=torch.float32, device=device)
    y_cure[ordt]=1
    delta = torch.zeros(n, dtype=torch.float32, device=device)
    T = torch.minimum(C, Y)
    delta[C >= Y] = 1
    tau = torch.unique(T[delta==1]).sort().values
    k = len(tau)
    d = torch.tensor([torch.sum(T == t) for t in tau], device=device)
    Rj = [torch.where(T >= t)[0] for t in tau]
    idx = [torch.where(tau <= t)[0] for t in T]#idx tau_j<=t
    return T, delta, tau, d, Rj, idx,y_cure
