import numpy as np
import numpy.random as rnd
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from scipy.spatial.distance import cdist


def second_smallest(lst):
    first, second = lst[0], lst[1]

    if first> second:
        first, second = second, first

    for num in lst[2:]:
        if num < first:  
            second, first = first, num  # Mettre à jour les 2 plus petits
        elif num < second or second == first:  
            second = num  # Mettre à jour seulement le deuxième plus petit
    return first, second





   

def NNDR(df_synthetic, df_true,all=False ):
    
    scaler = preprocessing.StandardScaler().fit(df_true)
    df_true_norm = pd.DataFrame(scaler.transform(df_true))
    df_sythetic_norm = pd.DataFrame(scaler.transform(df_synthetic))

    distances = cdist(df_true_norm.values, df_sythetic_norm.values, metric="euclidean")
    nndr =[]
    dcr =[]
    for i in range(len(df_true_norm)):
        first, second = second_smallest(distances[i])
        nndr.append(first/second)
        dcr.append(first)
    if all==True :
        return ([np.percentile(dcr,5),np.percentile(dcr,10), np.percentile(dcr,25), np.percentile(dcr,50), np.percentile(dcr,75),np.percentile(dcr,90), np.percentile(dcr,95) ], 
        [np.percentile(nndr,5),np.percentile(nndr,10), np.percentile(nndr,25), np.percentile(nndr,50), np.percentile(nndr,75),np.percentile(nndr,90), np.percentile(nndr,95)]) 
    else :
        return np.percentile(dcr,5),np.percentile(nndr,5)





