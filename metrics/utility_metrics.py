import numpy as np
import numpy.random as rnd
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn import preprocessing
from scipy.stats import energy_distance


import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.spatial.distance import cdist

from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import numpy as np
import ot
import cvxpy as cp
import dcor

print("on est ici 1")


from rpy2.robjects.packages import importr



def wasserstein_dist(df, df_true, method="pot", scale = True):
    if scale :
        scaler = preprocessing.StandardScaler().fit(df_true)
        df_true = pd.DataFrame(scaler.transform(df_true))
        df = pd.DataFrame(scaler.transform(df))

    if method == "pot":
        X = np.array(df)
        Y = np.array(df_true)

        a = np.ones(len(X)) / len(X)  # Distribution de poids pour X
        b = np.ones(len(Y)) / len(Y)  # Distribution de poids pour Y
        # Matrice des coûts (ici, la distance euclidienne)
        M = ot.dist(X, Y, metric='euclidean')

    # Calcul de la distance de Wasserstein
        l = ot.emd2(a, b, M)
        #print(l)
        return float(l)

    if method == "cvx": 
        X = np.array(df)
        Y = np.array(df_true)
        n = len(df)
        a = np.ones(n) / n  # Distribution de poids pour X
        b = np.ones(n) / n # Distribution de poids pour Y
        # Matrice des coûts (ici, la distance euclidienne)
        C = ot.dist(X, Y, metric='euclidean')

        P = cp.Variable((len(X), len(Y)), nonneg=True)
        objective = cp.Minimize(cp.sum(cp.multiply(C, P)))
        constraints = [
        P >= 0,  # Non-négativité
        cp.sum(P, axis=0) == np.ones(n)/n, 
        cp.sum(P, axis=1) == np.ones(n)/n   ]

        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return float(result)


def energy(df, df_true, scale = True, r=True):

    if scale : 
        scaler = preprocessing.StandardScaler(with_std=True).fit(df_true)
        df_true = pd.DataFrame(scaler.transform(df_true))
        df = pd.DataFrame(scaler.transform(df))
        #Loss =  SamplesLoss("energy")
        #return float(Loss(torch.tensor(df.values, dtype=torch.float32), torch.tensor(df_true.values, dtype=torch.float32)))


    #if r :
    #    robjects.r('install.packages("energy", repos="https://cloud.r-project.org")')
    #    energy = importr("energy")
    #    df1_r = pandas2ri.py2rpy(df_true)
    #    df2_r = pandas2ri.py2rpy(df)
    #    energy_distance = energy.edist(df1_r, df2_r)
    #    return  energy_distance

    if r :
        X = df.to_numpy()
        print("compute energy")
        
        Y = df_true.to_numpy()
    
        d_xy = np.mean(cdist(X, Y, metric='euclidean'))
        d_xx = np.mean(cdist(X, X, metric='euclidean'))
        d_yy = np.mean(cdist(Y, Y, metric='euclidean'))
        print(d_xy)
        print(d_xx)
        print(d_yy)
        print(2 * d_xy - d_xx - d_yy)
        print("end energy")
        return 2 * d_xy - d_xx - d_yy



    else : 
        return dcor.distance_correlation(df, df_true)


    

def coverage(value, ci):
    if float(ci[0])<value:
        if float(ci[1])> value:
            return 1
        else:
            return 0
    else :
        return 0

def energy_r(df, df_true):
    return 0


def p_mse(df_synthetic,df_true,  method ='rf') : 
    """
    Args:
        df_true (np.array | pd.DataFrame): Real Data
        df_synthetic (np.array | pd.DataFrame): Synthetic Data
    """
    I = [1]*len(df_true)

    df_true = pd.DataFrame(df_true).reset_index(drop=True)

    df_true = pd.concat([pd.DataFrame(df_true), pd.DataFrame({'I' : I})], axis=1)

    I = [0]*len(df_synthetic)
    df_synthetic=df_synthetic.reset_index(drop=True)

    df_synthetic= pd.concat([pd.DataFrame(df_synthetic), pd.DataFrame({'I' : I}).reset_index(drop=True)], axis=1)

    df = pd.concat([df_true,df_synthetic], axis = 0)

    y, X = df["I"], df.drop(columns=['I'])
    

    if method == "rf":
        model = RandomForestClassifier(n_estimators=500)

    if method == "cart":
        model = RandomForestClassifier(n_estimators=500)

    if method == 'log':
        model = LogisticRegression(max_iter=3000)

    model.fit(X, y)
    p_hat = model.predict_proba(X)
    c = len(df_true)/(len(df_synthetic)+len(df_true))




    return np.mean([((p - c)**2) for p in p_hat])
