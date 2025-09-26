import numpy as np
import pandas as pd

from statis.statis_X import aggregation_statis_X
from statis.statis_dual import aggregation_statis_dual
from statis.double_statis import aggregation_statis_double




class Statis:
    def __init__(self, statis_method, weight_method, compromis_method = "eigen", delta_weight = False, rotation=True):
        self.statis_method = statis_method
        self.weight_method = weight_method
        self.compromis_methods = compromis_method
        self.delta_weight = delta_weight
        self.rotation = rotation
        print(rotation)

    def gen(self, DF, df_true):
        if self.statis_method == "X":
            return aggregation_statis_X(DF=DF, 
                                df_true=df_true, 
                                weight_method=self.weight_method, 
                                compromis_method=self.compromis_methods,
                                delta_weight = self.delta_weight)

        elif self.statis_method == "dual": 
            return aggregation_statis_dual(DF=DF, 
                                df_true=df_true, 
                                weight_method=self.weight_method, 
                                compromis_method=self.compromis_methods,
                                delta_weight = self.delta_weight,
                                rotation = self.rotation)
        elif self.statis_method == "double":
            return aggregation_statis_double(DF=DF, 
                                df_true=df_true, 
                                weight_method=self.weight_method, 
                                compromis_method=self.compromis_methods,
                                delta_weight = self.delta_weight)
        else :
            raise("Statis method not supported")



