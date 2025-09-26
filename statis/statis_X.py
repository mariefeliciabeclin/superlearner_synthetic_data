
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
from metrics.utility_metrics import wasserstein_dist



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



def statis(DF,df_true, delta, delta_weight = True):
    # procède à statis sur les X
    n_df=len(DF)
    omega = np.zeros((n_df, n_df))

    DF_scaled = []
    ## scale all dataframes
    for i in range(n_df):
        df, _ = assign(DF[i], df_true, scale= True)
        scaler = preprocessing.StandardScaler().fit(df)
        df_new = scaler.transform(df)
        DF_scaled.append(df_new)

    for i in range(n_df):
        for j in range(n_df):
            omega[i,j] = np.trace(np.dot(DF_scaled[i].T,DF_scaled[j]))

    
    
    if delta_weight==True:
        omega = np.dot(np.array(omega),np.diag(delta))

    #scaler = preprocessing.StandardScaler().fit(omega)
    #omega = scaler.transform(omega)
    

    omega = np.dot(np.array(omega),np.diag(np.array(delta)))
       
    valp, vectp = np.linalg.eigh(omega)

    E = pd.DataFrame({"val":valp, "vect":vectp.tolist()}).sort_values(by="val", ascending=False)

   
# Extraire la plus grande valeur propre et son vecteur propre
    indice_p = 0

    indice_p = 0
    while indice_p+1 < len(E) and not np.all(np.array(E["vect"].iloc[indice_p]) >= 0):
        indice_p = indice_p+1


    indice_n = 0
    while indice_n +1< len(E) and not np.all(np.array(E["vect"].iloc[indice_n]) < 0):
        indice_n = indice_n+1



    if indice_n<indice_p : 
        tau_1 = np.array(E['vect'].iloc[indice_n])


    if indice_p<=indice_n : 
        tau_1 = np.array(E['vect'].iloc[indice_p])
    
    tau_1 = (1/np.sum(tau_1))*tau_1



# Calculer le barycentre pondéré
## norm tau
    tau_1 = (1/np.sum(tau_1))*tau_1
    df_compromis =tau_1[0]*DF[0]
    
    for i in range(n_df-1):
        df_compromis = df_compromis+tau_1[i+1]*DF[i+1]
    return df_compromis


def aggregation_statis_X(DF, df_true, weight_method, compromis_method="eigen", delta_weight = True):
    #Compute the compromis according to "compromis_method" 
    # the statis method (weighted with the first eigen vector in the SVD) 
    # or with other weigth define with the "weight_method"

    K= len(DF)
    if weight_method =="energy":
        Loss =  SamplesLoss(weight_method)
        weight = np.array([1/Loss(torch.tensor(DF[i].values, dtype=torch.float32), torch.tensor(df_true.values, dtype=torch.float32) ).item() for i in range(K)])
    if weight_method == "wasserstein":
        weight = np.array([1/wasserstein_dist(DF[i],df_true, method="pot") for i in range(K)] )
    
    weight = (1/np.sum(weight))*weight

    if compromis_method == "eigen":
        return statis(DF=DF,df_true=df_true, delta=weight, delta_weight = True)

    elif compromis_method == "weight" :
        weight = (1/np.sum(weight))*weight
        df_compromis =weight[0]*DF[0]
        for i in range(K-1):
            df, _ = assign(DF[i], df_true, scale= True)
            df_compromis = df_compromis+weight[i+1]*df
        return df_compromis
    else :
        df_compromis = (1/K)*DF[0]
        for i in range(K-1):
            df, _ = assign(DF[i], df_true, scale= True)
            df_compromis = df_compromis+(1/K)*df
        return df_compromis


