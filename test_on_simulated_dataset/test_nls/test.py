import pandas as pd
import numpy as np




import pandas as pd

from rpy2.robjects import pandas2ri
from rpy2 import robjects as ro
import numpy as np
def regression_nls(df):
    
    pandas2ri.activate()

    data = pandas2ri.py2rpy(df)
    ro.globalenv['data'] = data
    try :
        ro.r('''
            nls_model <- nls(
            Y ~ a * exp(-b * X0 - c*X1 - d*X2 - e*X3),
            data = data,
            algorithm = "port",
            start = list(a = 2, b = 1, c = 1, d = 1, e = 0)
            )
        ''')

        coeff = ro.r('summary(nls_model)$coefficients[ , "Estimate"]')
        coeff = pd.DataFrame({"a" : [coeff[0]], "beta_0" : [coeff[1]] , "beta_1" : [coeff[2]], "beta_2" : [coeff[3]],  "beta_3" : [coeff[4]]})
        return coeff
    except :
        return pd.DataFrame({"a" : [np.nan], "beta_0" : [np.nan] , "beta_1" :[np.nan], "beta_2" : [np.nan],  "beta_3" : [np.nan]})
        


DATA_PATH = "synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated/simulated_data_nls_5000.csv"
df_true = pd.read_csv(DATA_PATH, index_col=0 )
df_true = pd.DataFrame(df_true, columns = ['X0', 'X1', 'X2', 'X3', 'Y'])
print(df_true)

params_true = regression_nls(df=df_true,)
print(params_true)