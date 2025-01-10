import torch
from sklearn.metrics import roc_auc_score  

def cindex_AUC(T, Z, delta, model,y_cure,auc='False'):
    n = len(T)
    concordant = 0
    discordant = 0
    tied = 0
    f1,f2=model(Z)
    f1=f1[:,0]
    f2=f2[:,0]
    c1=torch.sigmoid(f1)
    y_true=y_cure.detach().cpu().numpy()
    y_scores=c1.detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_scores)
    for i in range(n-1):
        for j in range(i+1, n):
            if delta[i] == 1 and delta[j] == 1:
                if (T[i] < T[j] and f2[i] > f2[j]) or (T[i] > T[j] and f2[i] < f2[j]):
                    concordant += 1
                elif (T[i] < T[j] and f2[i] < f2[j]) or (T[i] > T[j] and f2[i] > f2[j]):
                    discordant += 1
                else:
                    tied += 1
    
    c_index = (concordant + 0.5 * tied) / (concordant + discordant + tied)
    return c_index,auc
