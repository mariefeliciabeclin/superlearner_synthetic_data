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


def assign(df, df_true, scale= True):

    if scale:
    ## to scale
        scaler = preprocessing.StandardScaler().fit(df_true)
        df_true_norm = pd.DataFrame(scaler.transform(df_true))
        df_norm = pd.DataFrame(scaler.transform(df))
    else:
        df_true_norm = df_true
        df_norm = df

    ## Or not 


    distances = cdist(df_true_norm.values, df_norm.values, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(distances)
    matched_pairs = list(zip(row_ind, col_ind))

    rc = pd.DataFrame({"row_ind" : row_ind, "col_ind" : col_ind}).sort_values(by="row_ind")
    df= df.iloc[rc['col_ind']].reset_index(drop=True) 

    return df, df_true


def appair(df, df_true):
    matcher = Matcher(df, df_true)
    matcher.fit_scores(balance=True, nmodels=100)  # Calcul des scores de propension
    matcher.match(method="min", nmatches=1, threshold=0.001)


def compromis_V(DF, delta=None,compromis_method = "eigen", delta_weight = True):
    DF_scaled=[]
    V=[]
    p_df = np.shape(DF[0])[1]
    n_df = np.shape(DF[0])[0]
    K = len(DF)

    

    S = np.array([1/n_df]*n_df)
    D = np.zeros((n_df, n_df))  # Matrice nulle de taille (n, p)
    np.fill_diagonal(D, S)

   


    for df in DF:
        scaler = preprocessing.StandardScaler().fit(df)
        df_new = scaler.transform(df)
        DF_scaled.append(df_new)

 
        V.append((np.array(df_new).T)@(np.array(df_new)))

    omega = np.zeros((K,K))
   
    for i in range(K):
        for j in range(K):
            omega[i,j] = np.trace(V[i]@V[j])

    if delta is None:
        delta= np.array([1/K]*K)

    if delta_weight ==False :
        delta= np.array([1/K]*K)

    #omega = np.dot(np.array(omega),np.diag(np.array(delta)))

    omega = np.array(omega)@np.diag(np.array(delta))@np.array(omega)
    print(omega) 
    valp, vectp = np.linalg.eigh(omega)
    print(vectp.T)

    E = pd.DataFrame({"val":valp, "vect":vectp.T.tolist()}).sort_values(by="val", ascending=False)

   
#Extraire la plus grande valeur propre et son vecteur propre
  
    print(E['vect'])
    print(E['val'])
    tau = np.array(E['vect'].iloc[0])
    if tau[0] <0:
        tau=-1*tau

    print(tau)

 
    a = np.sum(tau)
    print('****')
    tau_1=(1/a)*tau
    print(tau_1)


       

    if compromis_method =="eigen":
        V_compromis = tau_1[0]*V[0]

        for i in range(K-1):
            V_compromis = V_compromis+tau_1[i+1]*V[i+1]
        return V_compromis

    elif compromis_method =="weight":
        V_compromis = delta[0]*V[0]
        for i in range(K-1):
            V_compromis = V_compromis+delta[i+1]*V[i+1]
        return V_compromis
    elif compromis_method =="no":
        V_compromis = (1/n_df)*V[0]
        for i in range(K-1):
            V_compromis = V_compromis+(1/n_df)*V[i+1]
        return V_compromis

def compromis_W(DF,df_true, delta=None, compromis_method = "eigen", delta_weight = True):
    DF_scaled=[]
    W=[]

    p_df = np.shape(DF[0])[1]
    n_df = np.shape(DF[0])[0]
    K = len(DF)


    S = [1/n_df]*n_df
    D = np.zeros((n_df, n_df))  # Matrice nulle de taille (n, p)
    np.fill_diagonal(D, S)

    for df in DF:
        df_new, df_true_bis = assign(df, df_true, scale= True)
        scaler = preprocessing.StandardScaler().fit(df_new)
        df_new = scaler.transform(df_new)
        DF_scaled.append(df_new)
        #df_new = assign(df, df_true, scale= True)
        W.append(np.array(df_new)@np.array(df_new).T)

    omega = np.zeros((K,K))
   
    for i in range(K):
        for j in range(K):
            omega[i,j] = np.trace(W[i]@W[j])

    if delta is None:
        delta= np.array([1/K]*K)

    if delta_weight ==False :
        delta= np.array([1/K]*K)
    
    omega = np.dot(np.array(omega),np.diag(np.array(delta)))
       
    valp, vectp = np.linalg.eigh(omega)

    E = pd.DataFrame({"val":valp, "vect":vectp.tolist()}).sort_values(by="val", ascending=False)

   
# Extraire la plus grande valeur propre et son vecteur propre
    indice_p = 0

    while indice_p+1 < len(E) and not np.all(np.array(E["vect"].iloc[indice_p]) >= 0):
        indice_p = indice_p+1


    indice_n = 0
    while indice_n +1< len(E) and not np.all(np.array(E["vect"].iloc[indice_n]) < 0):
        indice_n = indice_n+1


    if indice_n<indice_p : 
        tau_1 = np.array(E['vect'].iloc[indice_n])
       

    if indice_n>=indice_p : 
        tau_1 = np.array(E['vect'].iloc[indice_p])
    tau_1 = (1/np.sum(tau_1))*tau_1
    

    if compromis_method:
        W_compromis = tau_1[0]*W[0]

        for i in range(K-1):
            W_compromis = W_compromis+tau_1[i+1]*W[i+1]
        return W_compromis

    else:
        W_compromis = delta[0]*V[0]
        for i in range(K-1):
            w_compromis = W_compromis+delta[i+1]*w[i+1]
        return W_compromis
