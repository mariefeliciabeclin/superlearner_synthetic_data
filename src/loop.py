
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model


def step(df, reg):
    n_col = len(df.columns)
    new_df = dict({})
    for i, col in enumerate(df.columns):
        Y = df[col]
        col_ = list(df.columns)
        col_.pop(i)
        X = df[col_]
        if reg =="RF":
            r = RandomForestRegressor(n_estimators=10)
        
        if reg =="norm":
            r = linear_model.LinearRegression()
        r = r.fit(X, Y)
        new_df[col] = r.predict(X)
    return pd.DataFrame(new_df)


def sampler(df, reg):
    df_step=df
    for i in range(2):
        df = step(df, reg=reg)
    return df


class Loop:
    def __init__(self, n_loops,sub_method):
        """Initialise le modèle loop"""
        self.n_loops = n_loops
        self.data =[]
        self.method = sub_method

    def fit(self, data):
        self.data = data
        self.n_sample = len(data)

    def generate(self, n_samples):
        """Génère des données synthétiques."""
        return sampler(self.data, reg= self.method)


