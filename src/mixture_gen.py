import numpy as np
import pandas as pd

import torch
from geomloss import SamplesLoss 

from metrics.utility_metrics import wasserstein_dist, energy

def weighting( DF, df_true,weight_method, weight_type = "inverse" ):
    K = len(DF)
    if weight_type == "inverse":
        if weight_method =="energy":
            print([1/energy(DF[i],df_true, scale=True) for i in range(K)])
            weights= np.array([1/energy(DF[i],df_true, scale=True) for i in range(K)])
        if weight_method == "wasserstein":
            weights = np.array([1/wasserstein_dist(DF[i],df_true, method="pot") for i in range(K)] )
    else :
        if weight_method =="energy":
            weights= np.array([1/energy(DF[i],df_true, scale=True) for i in range(K)])
        if weight_method == "wasserstein":
            weights= np.array([1/wasserstein_dist(DF[i],df_true, method="pot") for i in range(K)] )
    return (1/np.sum(weights))*weights
        

class Mixture:
    def __init__(self, weight_method, weight_type = 'inverse'):
        self.weight_method = weight_method
        self.weight_type = weight_type

    def gen(self, DF, df_true):

        n_df = len(DF[0])
        weights = weighting( DF=DF, df_true=df_true,weight_method = self.weight_method, weight_type = self.weight_type )
        #weights = [int(round(w*n_df, 0)) for w in weights]
        S= []

        
        #Z = np.random.choice(len(weights),n_df, p=weights, replace=True)
        #unique, counts = numpy.unique(a, return_counts=True)

        #dict(zip(unique, counts))
        for i in range(n_df):
            z = np.random.choice(len(weights), p=weights)
            S.append(DF[z].sample(n=1, replace=True))
        return pd.concat(S,axis= 0)

        