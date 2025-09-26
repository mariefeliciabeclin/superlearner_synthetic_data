import numpy as np
import numpy.random as rnd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path






def generate_hightcorr_data(seed=123,n=500,
                           mean=None,
                           cov_matrix=None,
                           beta_real=None,):
      np.random.seed(seed)

      if mean is None:
              mean = [7, 1.9, 6, 9]

      if cov_matrix is None:
              N = [[0.5, 0.08, 2.5, 0.5], 
            [0.08, 0.25,-1.3, 0.5],
            [2.5, -1.3, 1, 3], 
            [0.5, 0.5, 3, 0.5] ]
      cov_matrix = np.transpose(N) @ N

      if beta_real is None:
        beta_real = [10, 6, 9, -2]


    # Génération des données
      X = rnd.multivariate_normal(mean=mean, cov=cov_matrix, size=n)
      Y = np.dot(X, beta_real) + rnd.normal(0, 0.02, n)
      Y = Y.reshape(n, 1)

      T = np.hstack((X, Y))
      col_names = ['X0', 'X1', 'X2', 'X3', 'Y']
      df = pd.DataFrame(T, columns=col_names)
      return df



def generate_and_save_data(n=500,
                           mean=None,
                           cov_matrix=None,
                           beta_real=None,
                           output_csv_path="simulated_data.csv",
                           output_pairplot_path="pairplot.pdf",
                           output_corrplot_path="correlation.pdf"):
      df = generate_hightcorr_data(n=500, mean=mean, cov_matrix=cov_matrix, beta_real=beta_real)
    
      print("Correlation matrix:")
      print(np.corrcoef(df.T))

       # Sauvegarde CSV
      df.to_csv(output_csv_path, index=False)

    # Pairplot
      sns.pairplot(df, corner=True, diag_kind="kde")
      plt.savefig(output_pairplot_path)
      plt.close()

    # Correlation heatmap
      fig, ax = plt.subplots()
      df_corr = df.corr()
      sns.heatmap(df_corr, xticklabels=df_corr.columns, 
                yticklabels=df_corr.columns, cmap='coolwarm', ax=ax)
      plt.savefig(output_corrplot_path)
      plt.close()

def main():
      base_path = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulation_on_multivariate_gaussian") 
      generate_and_save_data(
      n=500,
      output_csv_path=base_path / "simulated_data_hightcorr.csv",
      output_pairplot_path=base_path / "pairplot_hightcorr.pdf",
      output_corrplot_path=base_path / "correlation_hightcorr.pdf"
    )

if __name__ == "__main__":
    main()
