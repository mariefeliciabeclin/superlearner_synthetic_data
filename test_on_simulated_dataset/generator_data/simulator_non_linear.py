import numpy as np
import numpy.random as rnd
import pandas as pd

import seaborn as sns
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ajouter le répertoire racine au chemin
sys.path.append(str(Path(__file__).resolve().parent.parent))

n = 500

mean = [7, 1.9, 6]

N = [[1.1, 0.08, 0.005,], 
       [0.08, 1.5,-0.03, ],
       [-0.005, -0.03, 1,], ]

cov = np.transpose(N)@N



print(cov)


X = rnd.multivariate_normal(mean=mean, cov=cov, size = n)
beta_real = [1, 6, 9, -2]
Z = np.array([X[i][1]+ (X[i][2]-6)**2 for i in range(500)])

Z = Z.reshape(n, 1)


# Concatenation des deux tableaux pour obtenir un (500, 8)
X = np.hstack((X,Z))

Y = np.dot(X,beta_real)+rnd.normal(0,0.2, n)
Y = Y.reshape(n, 1)
T = np.hstack((X,Y))

#col_name = np.array(['X0', 'X1', 'X2', 'X3','Y'])
col_name = np.array(['X0', 'X1', 'X2', 'X3', 'Y'])
df_true = pd.DataFrame(T, columns=col_name)

print(np.corrcoef(df_true.T))


print(df_true)

df_true.to_csv("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulation_on_multivariate_gaussian/simulated_data_nonlinear.csv")

sns.pairplot(df_true, corner=True, diag_kind="kde", )

plt.savefig("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulation_on_multivariate_gaussian/pairplot_nonlinear.pdf")


fig, ax = plt.subplots(1,1)
df_corr = df_true.corr()
sns.heatmap(df_corr, xticklabels = df_corr.columns , 
                 yticklabels = df_corr.columns, cmap = 'coolwarm', ax=ax)

plt.savefig("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulation_on_multivariate_gaussian/correlation_nonlinear.pdf")




