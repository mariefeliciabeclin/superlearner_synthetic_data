import numpy as np
import numpy.random as rnd
import pandas as pd

import seaborn as sns
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ajouter le répertoire racine au chemin
sys.path.append(str(Path(__file__).resolve().parent.parent))

n = 5000

mean = [7, 1.9, 6, 9]

N = [[0.5, 0.08, 2.5, 0.5], 
       [0.08, 0.25,-1.3, 0.5],
       [2.5, -1.3, 1, 3], 
       [0.5, 0.5, 3, 0.5] ]

cov = np.transpose(N)@N


print(cov)


X = rnd.multivariate_normal(mean=mean, cov=cov, size = n)
beta_real = [10, 6, 9, -2]

### Real regression

Y = np.dot(X,beta_real)+rnd.normal(0,0.02, n)


Y = Y.reshape(n, 1)
# Concatenation des deux tableaux pour obtenir un (500, 8)
T = np.hstack((X,Y))

#col_name = np.array(['X0', 'X1', 'X2', 'X3','Y'])
col_name = np.array(['X0', 'X1', 'X2', 'X3', 'Y'])
df_true = pd.DataFrame(T, columns=col_name)

X = rnd.multivariate_normal(mean=mean, cov=cov, size = n)
beta_real = [10, 6, 9, -2]


### Real regression
#Y = np.dot(X,beta_real)+rnd.normal(0,0.02, n)
Y = np.dot(X,beta_real)+rnd.normal(0,0.02, n)

Y = Y.reshape(n, 1)


# Concatenation des deux tableaux pour obtenir un (500, 8)
T = np.hstack((X,Y))
#col_name = np.array(['X0', 'X1', 'X2', 'X3','Y'])
col_name = np.array(['X0', 'X1', 'X2', 'X3', 'Y'])
df_true = pd.DataFrame(T, columns=col_name)

print(df_true)

df_true.to_csv("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulation_on_multivariate_gaussian/simulated_data_hightcorr_big.csv")

sns.pairplot(df_true, corner=True, diag_kind="kde",)

plt.savefig("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulation_on_multivariate_gaussian/pairplot_hightcorr_big.pdf")


fig, ax = plt.subplots(1,1)
df_corr = df_true.corr()
sns.heatmap(df_corr, xticklabels = df_corr.columns , 
                 yticklabels = df_corr.columns, cmap = 'coolwarm', ax=ax)

plt.savefig("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulation_on_multivariate_gaussian/correlation_hightcorr_big.pdf")




