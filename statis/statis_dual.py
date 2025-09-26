
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
from scipy.linalg import orthogonal_procrustes

from statis.tools import compromis_V, assign
from metrics.utility_metrics import wasserstein_dist, energy


def compromise_table(df,df_true, V_compromis, rotation):
    df, df_true = assign(df, df_true, scale= True)
    scaler = preprocessing.StandardScaler().fit(df)
    df= scaler.transform(df)

    scaler_true = preprocessing.StandardScaler().fit(df_true)
    df_true_scale= scaler.transform(df_true)




    U, D, Ut = np.linalg.svd(np.array(df)@np.array(df).T)
    
    print((D[:-1] >= D[1:]).all())

    Q, S, Qt = np.linalg.svd(np.array(V_compromis))

    print((S[:-1] >= S[1:]).all())
     
    if np.shape(V_compromis)[0] ==np.shape(V_compromis)[1]:
        p=np.shape(V_compromis)[0]
    else:
        raise ValueError("No valid dimension for V_compromis")
   
    n = len(df)
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


def aggregation_statis_dual(DF,df_true, weight_method = "wasserstein", compromis_method = "eigen", delta_weight = True, rotation = False):

    K= len(DF)

    if weight_method =="energy":
        weight = np.array([1/energy(DF[i],df_true, scale=True) for i in range(K)] )
    
    if weight_method =="wasserstein":
        weight = np.array([1/wasserstein_dist(DF[i],df_true, method="pot", scale=True) for i in range(K)] )
    
    weight = (1/np.sum(weight))*weight

    delta = (1/np.sum(weight))*weight
    
    V_compromis = compromis_V(DF, delta=delta, compromis_method = compromis_method, delta_weight=delta_weight)

    j = np.argmax(delta)
    X_bar = compromise_table(df=DF[j],df_true =df_true, V_compromis=V_compromis, rotation=rotation)

    return X_bar