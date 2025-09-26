import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def generate_nonlinear_data(seed=123, n=5000):
    """Génère un DataFrame avec des relations non linéaires entre X et Y."""
    np.random.seed(seed)

    x_0 = np.random.normal(loc=0, scale=1, size=n)
    x_1 = np.random.normal(loc=0.9, scale=0.51, size=n)
    x_2 = (x_1 - 0.9) ** 2 + np.random.normal(loc=0, scale=0.25, size=n)
    x_3 = np.random.chisquare(df=6, size=n) + np.exp(x_0)

    y_data =  * np.exp(-1.3 * x_0 - 2 * x_1 - 1.2 * x_2 - 0.03 * x_3) \
             + np.random.normal(loc=0, scale=0.02, size=n)
2.5
    df = pd.DataFrame({
        "X0": x_0, "X1": x_1, "X2": x_2, "X3": x_3, "Y": y_data
    })

    return df

def save_dataset_and_plots(df: pd.DataFrame,
                            output_csv_path: Path,
                            output_pairplot_path: Path,
                            output_corrplot_path: Path):
    """Sauvegarde le dataset en CSV, un pairplot et une heatmap des corrélations."""
    # Sauvegarde du DataFrame en CSV
    df.to_csv(output_csv_path, index=False)

    # Pairplot
    sns.pairplot(df, corner=True, diag_kind="kde")
    plt.savefig(output_pairplot_path)
    plt.close()

    # Heatmap des corrélations
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                cmap='coolwarm', ax=ax)
    plt.savefig(output_corrplot_path)
    plt.close()

def main(n=500):
    base_path = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated")

    df = generate_nonlinear_data(seed=123, n=n)
    
    save_dataset_and_plots(
        df,
        output_csv_path=base_path / f"simulated_data_nls_{n}.csv",
        output_pairplot_path=base_path / f"pairplot_nls_{n}.pdf",
        output_corrplot_path=base_path / f"correlation_nonlinear_{n}.pdf",
    )

if __name__ == "__main__":
    if len(sys.argv) == 0:
        n=500
    else:
        n = int(sys.argv[1])
    main(n=n)
