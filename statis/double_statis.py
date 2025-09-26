import numpy as np
import numpy.random as rnd
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn import preprocessing
from scipy.stats import energy_distance

import torch
from geomloss import SamplesLoss 

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.spatial.distance import cdist

from scipy.optimize import linear_sum_assignment

from statis.tools import compromis_V, compromis_W
from metrics.utility_metrics import wasserstein_dist

from scipy.linalg import orthogonal_procrustes

def compromise_table(V_compromis, W_compromis, df_true, rotation=True):
    scaler = preprocessing.StandardScaler().fit(df_true)
    df_true_scale= scaler.transform(df_true)

    U, D, Ut = np.linalg.svd(np.array(W_compromis))
    Q, S, Qt = np.linalg.svd(np.array(V_compromis))
     
    if np.shape(V_compromis)[0] ==np.shape(V_compromis)[1]:
        p=np.shape(V_compromis)[0]
    else:
        raise ValueError("No valid dimension for V_compromis")
   
    n = np.shape(W_compromis)[0]
    S = np.array([np.sqrt(l) for l in S])
    D = np.zeros((n, p))  # Matrice nulle de taille (n, p)
    np.fill_diagonal(D, S) 
    X_bar = U @D @Qt
   
    if rotation :
        print("rotation = true")

        R, _ = orthogonal_procrustes(X_bar, df_true_scale)
        X_bar = X_bar @ R
    else :
        print("rotation = false")

    X_bar = scaler.inverse_transform(X_bar)        

    return X_bar  



def aggregation_statis_double(DF, df_true, weight_method = "wasserstein", compromis_method = "eigen", delta_weight = True):
    K= len(DF)
    if weight_method =="energy":
        Loss =  SamplesLoss(weight_method)
        weight = np.array([1/Loss(torch.tensor(DF[i].values, dtype=torch.float32), torch.tensor(df_true.values, dtype=torch.float32) ).item() for i in range(K)])
    if weight_method =="wasserstein":
        weight = np.array([1/wasserstein_dist(DF[i],df_true, method = "pot") for i in range(K)] )

    delta = (1/np.sum(weight))*weight
    j = np.argmax(delta)

    V_compromis = compromis_V(DF, delta = delta,compromis_method = "eigen", delta_weight = True)
    W_compromis = compromis_W(DF, df_true=df_true, delta = delta,compromis_method = "eigen", delta_weight = True)

    #j = np.argmax(delta)
    #scaler = preprocessing.StandardScaler().fit(DF[j])

    X = compromise_table(V_compromis = V_compromis, W_compromis = W_compromis, df_true = df_true)
    return X                                                                                             



