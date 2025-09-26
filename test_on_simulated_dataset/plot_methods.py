import numpy as np
import numpy.random as rnd
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn import preprocessing
from scipy.stats import energy_distance
import time as time

#import torch as torch
from geomloss import SamplesLoss 
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ajouter le répertoire racine au chemin
sys.path.append(str(Path(__file__).resolve().parent.parent))

from nonlinear_regression.nls import regression_nls
from tools.utils import bootstrap
from src.synthetic_gen import Generator

from src.statis_gen import Statis
from src.mixture_gen import Mixture

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.feature_selection import mutual_info_regression

from metrics.utility_metrics import wasserstein_dist, energy, p_mse
from metrics.privacy_metrics import NNDR

import seaborn as sns
from scipy.stats import pearsonr, ks_2samp

import argparse


def main(data, all):  
    if data == "hightcorr":
        df_true = pd.read_csv("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated/simulated_data_hightcorr.csv", index_col=0)
    elif data == "lowcorr":
        df_true = pd.read_csv("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated/simulated_data_lowcorr.csv", index_col=0)
    elif data == "nonlinear":
        df_true = pd.read_csv("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated/simulated_data_nonlinear.csv", index_col=0)
    elif data == "nls":
        df_true = pd.read_csv("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated/simulated_data_nls.csv", index_col=0)  
    if all:
        METHODS_SYNTHGEN = [ 'synthpop_norm', 'synthpop_cart',  'avatar_ncp=2','avatar_ncp=4', "synthcity_ctgan", "gaussian_copula", "tvae", "sdv_ctgan"]
        type = "all"
    else :
        METHODS_SYNTHGEN = [ 'synthpop_norm', 'synthpop_cart','avatar_ncp=4',"sdv_ctgan" ]
        type ="best"

    n=500
    SYNTHETIC_METHODS = [ "statis_X_energy_eigen", "statis_X_wass_eigen","statis_X_energy_weight", "statis_X_wass_weight", "statis_dual_energy_eigen", "statis_dual_wass_eigen","statis_dual_energy_weight", "statis_dual_wass_weight","statis_dual_no_weight_eigen", "statis_double_energy_eigen", "statis_double_wass_eigen","statis_double_energy_weight", "statis_double_wass_weight","statis_double_no_weight_eigen","mixture_wass", "mixture_energy" ]

    statis_param = dict({"statis dual_energy eigen" : {"statis_method": "dual", "weight_method" :"energy", "compromis_method" : "eigen",  "delta_weight" : True, "rotation":True}, 
                    'statis dual_wass eigen' :  {"statis_method": "dual","weight_method" :"wasserstein", "compromis_method" : "eigen",  "delta_weight" : True, "rotation":True},
                    "statis dual_energy weight": {"statis_method" : "dual", "weight_method" :"energy", "compromis_method" : "weight",  "delta_weight" : True, "rotation":True},
                    "statis dual_wass weight": {"statis_method" : "dual", "weight_method" :"wasserstein", "compromis_method" : "weight",  "delta_weight" : True, "rotation":True},
                    
                    "statis_double_energy_eigen" : {"statis_method": "double", "weight_method" :"energy", "compromis_method" : "eigen",  "delta_weight" : True, "rotation":False}, 
                    'statis_double_wass_eigen' :  {"statis_method": "double","weight_method" :"wasserstein", "compromis_method" : "eigen",  "delta_weight" : True, "rotation":False},
                    "statis_double_energy_weight": {"statis_method" : "double", "weight_method" :"energy", "compromis_method" : "weight",  "delta_weight" : True, "rotation":False},
                    "statis_double_wass_weight": {"statis_method" : "double", "weight_method" :"wasserstein", "compromis_method" : "weight",  "delta_weight" : True, "rotation":False},
                    "statis_double_no_weight_eigen": {"statis_method" : "double", "weight_method" :"energy", "compromis_method" :"eigen",  "delta_weight" : False, "rotation":False} ,                   
                    
                    "statis X_energy eigen" : {"statis_method": "X", "weight_method" :"energy", "compromis_method" : "eigen",  "delta_weight" : True}, 
                    'statis X_wass eigen' :  {"statis_method": "X","weight_method" :"wasserstein", "compromis_method" : "eigen",  "delta_weight" : True},
                    "statis X_energy weight": {"statis_method" : "X", "weight_method" :"energy", "compromis_method" : "weight",  "delta_weight" : True},
                    "statis X_wass weight": {"statis_method" : "X", "weight_method" :"wasserstein", "compromis_method" : "weight",  "delta_weight" : True},
                    })
    

    DF =[]

    if data=="nls":
        params_true = regression_nls(df=df_true)
    else:
        reg_linear = sm.OLS(df_true['Y'], df_true[['X0', 'X1', 'X2', 'X3']]).fit()
        params_true = pd.DataFrame(reg_linear.params).T
        params_true.columns = ["beta_0", "beta_1", "beta_2", "beta_3"]

    def corrfunc(x, y, label, color, ax=None):
        """Plot the correlation coefficient in the bottom left hand corner of a plot."""
        r, _ = pearsonr(x, y)
        ax = ax or plt.gca()
    
        y_offset = 0.9 if label == 'original' else 0.5
        ax.annotate(f'{label}:\n ρ = {r:.2f}', xy=(0.1, y_offset), xycoords=ax.transAxes, color=color, fontsize=10)



    inverse_weight_wass=dict({})
    inverse_weight_energy=dict({})

    for method in METHODS_SYNTHGEN : 
        
        if method=='synthpop_cart':
            data_synth_gen = Generator(method='synthpop',params_synthpop=dict({'sub_method': ['cart']}) ).get_generator()

        elif method=='synthpop_norm':
            data_synth_gen = Generator(method='synthpop',params_synthpop=dict({'sub_method': ['norm']}) ).get_generator()
        
    
        elif method=='avatar_ncp=2':
            print("ncp=2")
            data_synth_gen = Generator(method='avatar',params_avatar=dict({'ncp': 2}) ).get_generator()

        elif method=='avatar_ncp=4':
            print("ncp=4")
            data_synth_gen = Generator(method='avatar',params_avatar=dict({'ncp': 4}) ).get_generator()

        else :
            data_synth_gen = Generator(method=method).get_generator()
        
        data_synth_gen.fit(df_true)
        gen = data_synth_gen.generate(n)
        gen = pd.DataFrame(gen, columns = ['X0', 'X1', 'X2', 'X3', 'Y'])

        DF.append(gen)

        I = ["original"]*len(df_true)
        df_true_I = pd.concat([pd.DataFrame(df_true), pd.DataFrame({'source' : I})], axis=1)

        I = [str(method).replace('_', '\n')]*len(gen)
        gen_I =gen.reset_index(drop=True)

        df_synthetic = pd.concat([pd.DataFrame(gen_I), pd.DataFrame({'source' : I}).reset_index(drop=True)], axis=1)
        # Fusionne les deux DataFrames

        df_combined = pd.concat([df_true_I,df_synthetic], axis = 0)
    
        inverse_weight_wass[method]=[round(1/wasserstein_dist(gen, df_true, scale=True),3)]    
        inverse_weight_energy[method]=[round(1/energy(gen, df_true, scale=True),3)]   


        g=sns.pairplot(df_combined, hue="source", corner=True, diag_kind="kde",
             plot_kws=dict(marker=".", linewidth=1, alpha=0.2), )
        g.map_lower(corrfunc)
        num_vars = df_true.select_dtypes(include=np.number).columns
        for i, var in enumerate(num_vars):
            ax = g.diag_axes[i] 
        # Test KS sur la variable actuelle
            stat, p_value = ks_2samp(df_true[var], gen[var])

        # Annotation sur le graphe
            ax.annotate(f'KS p={p_value:.3f}', xy=(0.05, 0.9), xycoords='axes fraction',ha='left', fontsize=10, color='red',
                bbox=dict(facecolor='white', alpha=0.6))
    

        if data=="nls":
            params= regression_nls(df=gen)
        else:
            reg_linear = sm.OLS(gen['Y'], gen[['X0', 'X1', 'X2', 'X3']]).fit()
            params= pd.DataFrame(reg_linear.params).T
            params.columns = ["beta_0", "beta_1", "beta_2", "beta_3"]

     # corner=True = triangle infér,ieur

    # Nombre de variables = taille de la grille
    # Taille de la grille du pairplot
        n_ax = len(g.axes)

        fig = g.fig

    # 1️⃣ Premier texte — case (0, n_ax-1) => position = 0 * n_ax + (n_ax-1) + 1
        pos1 = n_ax  # car pos1 = 0 * n_ax + (n_ax-1) + 1 = n_ax
        ax_new = fig.add_subplot(n_ax, n_ax, pos1)

        ax_new.text(0, 0,
        r'$\hat{\beta}_0^{original}$ : '+str(round(params_true['beta_0'][0], 3))+'\n'
        + r'$\hat{\beta}_1^{original}$ : '+str(round(params_true['beta_1'][0], 3))+'\n'
        + r'$\hat{\beta}_2^{original}$ : '+str(round(params_true['beta_2'][0], 3))+'\n'
        + r'$\hat{\beta}_3^{original}$ : '+str(round(params_true['beta_3'][0], 3))+'\n',
        transform=ax_new.transAxes, ha='left', va='top', fontsize=15)

        ax_new.set_xticks([])
        ax_new.set_yticks([])
        ax_new.set_frame_on(False)

    # 2️⃣ Deuxième texte — case (1, n_ax-1) => position = 1 * n_ax + (n_ax-1) + 1
        pos2 = 1 * n_ax + (n_ax - 1) + 1
        ax_new_2 = fig.add_subplot(n_ax, n_ax, pos2)

        ax_new_2.text(0, 0,
            r'$\hat{\beta}_0$ : '+str(round(params['beta_0'][0], 3))+'\n'
            + r'$\hat{\beta}_1$ : '+str(round(params['beta_1'][0], 3))+'\n'
            + r'$\hat{\beta}_2$ : '+str(round(params['beta_2'][0], 3))+'\n'
            + r'$\hat{\beta}_3$ : '+str(round(params['beta_3'][0], 3))+'\n',
        transform=ax_new_2.transAxes, ha='left', va='top', fontsize=15)

        ax_new_2.set_xticks([])
        ax_new_2.set_yticks([])
        ax_new_2.set_frame_on(False)



        pos4 = 1 * n_ax + 3
        ax_new_4 = fig.add_subplot(n_ax, n_ax, pos4)
        ax_new_4.text(0,0, "Wasserstein distance " +str(round(wasserstein_dist(gen, df_true, scale=True),3)) +'\n'+"Energy distance " +str(round(energy(gen, df_true, scale=True),3))+'\n'+"Energy distance dcor " +str(round(energy(gen, df_true, scale=True, r=False),3)),
        transform=ax_new_4.transAxes, ha='left', va='bottom', fontsize=15)

        ax_new_4.set_xticks([])
        ax_new_4.set_yticks([])       
        ax_new_4.set_frame_on(False)

# Boucle sur les axes vides (triangle sbottomérieur)
    

       
        plt.savefig(str(method)+"_"+type+"_"+data+".pdf")

    inverse_weight_energy = pd.DataFrame(inverse_weight_energy)
    inverse_weight_wass= pd.DataFrame(inverse_weight_wass)


    inverse_weight_energy = inverse_weight_energy.div(inverse_weight_energy.sum(axis=1), axis=0)

    inverse_weight_wass = inverse_weight_wass.div(inverse_weight_wass.sum(axis=1), axis=0)

    inverse_weight_wass=inverse_weight_wass.round(2)
    inverse_weight_energy=inverse_weight_energy.round(2)
         
    print(inverse_weight_wass)   

    for key, value in statis_param.items():

        gen = pd.DataFrame(Statis(**value).gen(DF=DF, df_true= df_true), columns = ['X0', 'X1', 'X2', 'X3', 'Y'])

        I = ["original"]*len(df_true)
        df_true_I = pd.concat([pd.DataFrame(df_true), pd.DataFrame({'source' : I})], axis=1)

        I = [str(key).replace('_', '\n')]*len(gen)
        gen_I =gen.reset_index(drop=True)

        df_synthetic = pd.concat([pd.DataFrame(gen_I), pd.DataFrame({'source' : I}).reset_index(drop=True)], axis=1)
    # Fusionne les deux DataFramesk

        df_combined = pd.concat([df_true_I,df_synthetic], axis = 0)
    
        print(df_combined)
    # Pairplot avec sbottomerposition via 'hue'
        g = sns.pairplot(df_combined, hue="source", corner=True, diag_kind="kde",
             plot_kws=dict(marker=".", linewidth=1, alpha=0.6), )
        g.map_lower(corrfunc)

        num_vars = df_true.select_dtypes(include=np.number).columns
        for i, var in enumerate(num_vars):
            ax = g.diag_axes[i] 
        # Test KS sur la variable actuelle
            stat, p_value = ks_2samp(df_true[var], gen[var])

        # Annotation sur le graphe
            ax.annotate(f'KS p={p_value:.3f}', xy=(0.05, 0.9), xycoords='axes fraction',ha='left', fontsize=10, color='red',
                bbox=dict(facecolor='white', alpha=0.6))
    

        g.map_lower(corrfunc)
        
        if data=="nls":
            params= regression_nls(df=gen)
        else:
            reg_linear = sm.OLS(gen['Y'], gen[['X0', 'X1', 'X2', 'X3']]).fit()
            params= pd.DataFrame(reg_linear.params).T
            params.columns = ["beta_0", "beta_1", "beta_2", "beta_3"]

        n_ax = len(g.axes)

        fig = g.fig

# 1️⃣ Premier texte — case (0, n_ax-1) => position = 0 * n_ax + (n_ax-1) + 1
        pos1 = n_ax  # car pos1 = 0 * n_ax + (n_ax-1) + 1 = n_ax
        ax_new = fig.add_subplot(n_ax, n_ax, pos1)

        ax_new.text(0, 0,
        r'$\hat{\beta}_0^{original}$ : '+str(round(params_true['beta_0'][0], 3))+'\n'
        + r'$\hat{\beta}_1^{original}$ : '+str(round(params_true['beta_1'][0], 3))+'\n'
        + r'$\hat{\beta}_2^{original}$ : '+str(round(params_true['beta_2'][0], 3))+'\n'
        + r'$\hat{\beta}_3^{original}$ : '+str(round(params_true['beta_3'][0], 3))+'\n',
        transform=ax_new.transAxes, ha='left', va='top', fontsize=15)

        ax_new.set_xticks([])
        ax_new.set_yticks([])
        ax_new.set_frame_on(False)

# 2️⃣ Deuxième texte — case (1, n_ax-1) => position = 1 * n_ax + (n_ax-1) + 1
        pos2 = 1 * n_ax + (n_ax - 1) + 1
        ax_new_2 = fig.add_subplot(n_ax, n_ax, pos2)

        ax_new_2.text(0, 0,
            r'$\hat{\beta}_0$ : '+str(round(params['beta_0'][0], 3))+'\n'
            + r'$\hat{\beta}_1$ : '+str(round(params['beta_1'][0], 3))+'\n'
            + r'$\hat{\beta}_2$ : '+str(round(params['beta_2'][0], 3))+'\n'
            + r'$\hat{\beta}_3$ : '+str(round(params['beta_3'][0], 3))+'\n',
        transform=ax_new_2.transAxes, ha='left', va='top', fontsize=15)

        ax_new_2.set_xticks([])
        ax_new_2.set_yticks([])        
        ax_new_2.set_frame_on(False)

# 1️⃣ Premier texte — case (0, 1) => position = 0 * n_ax + (n_ax-1) + 1
        pos3 = 0 * n_ax + 5
        ax_new_3 = fig.add_subplot(n_ax, n_ax, pos3)

        if statis_param[key]["weight_method"] == "wasserstein":
            table = ax_new_3.table(cellText=inverse_weight_wass.head(1).values, colLabels=inverse_weight_wass.columns,
                  cellLoc="center", loc="upper right", 
                  rowLoc='center',
                 
                  )
    
        elif statis_param[key]["weight_method"] == "energy":
            table = ax_new_3.table(cellText=inverse_weight_energy.head(1).values, colLabels=inverse_weight_energy.columns,
                  cellLoc="center", loc="upper right", 
                  rowLoc='center',
                  
                  )

        table.scale(3, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        ax_new_3.axis('off')

        ax_new_2.set_xticks([])
        ax_new_2.set_yticks([])        
        ax_new_2.set_frame_on(False)

        pos4 = 1 * n_ax + 3
        ax_new_4 = fig.add_subplot(n_ax, n_ax, pos4)
        ax_new_4.text(0,0, "Wasserstein distance " +str(round(wasserstein_dist(gen, df_true, scale=True),3)) +'\n'+"Energy distance " +str(round(energy(gen, df_true, scale=True),3)) ,
        transform=ax_new_4.transAxes, ha='left', va='bottom', fontsize=15)

        ax_new_4.set_xticks([])
        ax_new_4.set_yticks([])        
        ax_new_4.set_frame_on(False)
    
        plt.show()
        plt.savefig(str(key)+"_"+type+"_"+data+".pdf")



    method = "mixture_wass"
    gen = Mixture(weight_method="wasserstein").gen(DF=DF, df_true = df_true)

    if data=="nls":
        params= regression_nls(df=gen)
    else:
        reg_linear = sm.OLS(gen['Y'], gen[['X0', 'X1', 'X2', 'X3']]).fit()
        params= pd.DataFrame(reg_linear.params).T
        params.columns = ["beta_0", "beta_1", "beta_2", "beta_3"]

    I = ["original"]*len(df_true)
    df_true_I = pd.concat([pd.DataFrame(df_true), pd.DataFrame({'source' : I})], axis=1)

    I = [str(key).replace('_', '\n')]*len(gen)
    gen_I =gen.reset_index(drop=True)



    df_synthetic = pd.concat([pd.DataFrame(gen_I), pd.DataFrame({'source' : I}).reset_index(drop=True)], axis=1)
    # Fusionne les deux DataFrames

    df_combined = pd.concat([df_true_I,df_synthetic], axis = 0)


    
    print(df_combined)
    # Pairplot avec sbottomerposition via 'hue'
    g=sns.pairplot(df_combined, hue="source", corner=True, diag_kind="kde",
             plot_kws=dict(marker=".", linewidth=1, alpha=0.2), )
    fig = g.fig
    g.map_lower(corrfunc)

    num_vars = df_true.select_dtypes(include=np.number).columns
    for i, var in enumerate(num_vars):
        ax = g.diag_axes[i] 
        # Test KS sur la variable actuelle
        stat, p_value = ks_2samp(df_true[var], gen[var])

        # Annotation sur le graphe
        ax.annotate(f'KS p={p_value:.3f}', xy=(0.05, 0.9), xycoords='axes fraction',ha='left', fontsize=10, color='red',
                bbox=dict(facecolor='white', alpha=0.6))

    pos1 = n_ax  # car pos1 = 0 * n_ax + (n_ax-1) + 1 = n_ax
    ax_new = fig.add_subplot(n_ax, n_ax, pos1)

    ax_new.text(0, 0,
        r'$\hat{\beta}_0^{original}$ : '+str(round(params_true['beta_0'][0], 3))+'\n'
        + r'$\hat{\beta}_1^{original}$ : '+str(round(params_true['beta_1'][0], 3))+'\n'
        + r'$\hat{\beta}_2^{original}$ : '+str(round(params_true['beta_2'][0], 3))+'\n'
        + r'$\hat{\beta}_3^{original}$ : '+str(round(params_true['beta_3'][0], 3))+'\n',
        transform=ax_new.transAxes, ha='left', va='top', fontsize=15)

    ax_new.set_xticks([])
    ax_new.set_yticks([])
    ax_new.set_frame_on(False)

# 2️⃣ Deuxième texte — case (1, n_ax-1) => position = 1 * n_ax + (n_ax-1) + 1
    pos2 = 1 * n_ax + (n_ax - 1) + 1
    ax_new_2 = fig.add_subplot(n_ax, n_ax, pos2)

    ax_new_2.text(0, 0,
            r'$\hat{\beta}_0$ : '+str(round(params['beta_0'][0], 3))+'\n'
            + r'$\hat{\beta}_1$ : '+str(round(params['beta_1'][0], 3))+'\n'
            + r'$\hat{\beta}_2$ : '+str(round(params['beta_2'][0], 3))+'\n'
            + r'$\hat{\beta}_3$ : '+str(round(params['beta_3'][0], 3))+'\n',
        transform=ax_new_2.transAxes, ha='left', va='top', fontsize=15)

    ax_new_2.set_xticks([])
    ax_new_2.set_yticks([])        
    ax_new_2.set_frame_on(False)


    pos3 = 0 * n_ax + 5
    ax_new_3 = fig.add_subplot(n_ax, n_ax, pos3)

    
    table = ax_new_3.table(cellText=inverse_weight_wass.head(1).values, colLabels=inverse_weight_wass.columns,
                  cellLoc="center", loc="upper right", 
                  rowLoc='center',
                 
                  )
    
    
    table.scale(3, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    ax_new_3.axis('off')

    pos4 = 1 * n_ax + 3
    ax_new_4 = fig.add_subplot(n_ax, n_ax, pos4)
    ax_new_4.text(0,0, "Wasserstein distance " +str(round(wasserstein_dist(gen, df_true, scale=True),3)) +'\n'+"Energy distance " +str(round(energy(gen, df_true, scale=True),3)) ,
    transform=ax_new_4.transAxes, ha='left', va='bottom', fontsize=15)

    ax_new_4.set_xticks([])
    ax_new_4.set_yticks([])        
    ax_new_4.set_frame_on(False)


    plt.show()
    plt.savefig(str(method)+"_"+data+".pdf")  

    method = "mixture_energy"

    df_mixture = Mixture(weight_method="energy").gen(DF=DF, df_true = df_true)

    if data=="nls":
        params= regression_nls(df=gen)
    else:
        reg_linear = sm.OLS(gen['Y'], gen[['X0', 'X1', 'X2', 'X3']]).fit()
        params= pd.DataFrame(reg_linear.params).T
        params.columns = ["beta_0", "beta_1", "beta_2", "beta_3"]

    I = ["original"]*len(df_true)
    df_true_I = pd.concat([pd.DataFrame(df_true), pd.DataFrame({'source' : I})], axis=1)

    I = [str(key).replace('_', '\n')]*len(gen)
    gen_I =gen.reset_index(drop=True)

    df_synthetic = pd.concat([pd.DataFrame(gen_I), pd.DataFrame({'source' : I}).reset_index(drop=True)], axis=1)
    # Fusionne les deux DataFrames

    df_combined = pd.concat([df_true_I,df_synthetic], axis = 0)
    
    print(df_combined)
    # Pairplot avec sbottomerposition via 'hue'
#g=sns.pairplot(df_combined, hue="source", corner=True, diag_kind="kde",
#             plot_kws=dict(marker=".", linewidth=1, alpha=0.6), )

    g=sns.pairplot(df_combined, hue="source", corner=True, diag_kind="kde",
            plot_kws=dict(marker=".", linewidth=1, alpha=0.6), )
    fig = g.fig
    num_vars = df_true.select_dtypes(include=np.number).columns
    for i, var in enumerate(num_vars):
        ax = g.diag_axes[i] 
        # Test KS sur la variable actuelle
        stat, p_value = ks_2samp(df_true[var], gen[var])

        # Annotation sur le graphe
        ax.annotate(f'KS p={p_value:.3f}', xy=(0.05, 0.9), xycoords='axes fraction',ha='left', fontsize=10, color='red',
                bbox=dict(facecolor='white', alpha=0.6))

    pos1 = n_ax  # car pos1 = 0 * n_ax + (n_ax-1) + 1 = n_ax
    ax_new = fig.add_subplot(n_ax, n_ax, pos1)

    ax_new.text(0, 0,
        r'$\hat{\beta}_0^{original}$ : '+str(round(params_true['beta_0'][0], 3))+'\n'
        + r'$\hat{\beta}_1^{original}$ : '+str(round(params_true['beta_1'][0], 3))+'\n'
        + r'$\hat{\beta}_2^{original}$ : '+str(round(params_true['beta_2'][0], 3))+'\n'
        + r'$\hat{\beta}_3^{original}$ : '+str(round(params_true['beta_3'][0], 3))+'\n',
        transform=ax_new.transAxes, ha='left', va='top', fontsize=15)

    ax_new.set_xticks([])
    ax_new.set_yticks([])
    ax_new.set_frame_on(False)

# 2️⃣ Deuxième texte — case (1, n_ax-1) => position = 1 * n_ax + (n_ax-1) + 1
    pos2 = 1 * n_ax + (n_ax - 1) + 1
    ax_new_2 = fig.add_subplot(n_ax, n_ax, pos2)

    ax_new_2.text(0, 0,
            r'$\hat{\beta}_0$ : '+str(round(params['beta_0'][0], 3))+'\n'
            + r'$\hat{\beta}_1$ : '+str(round(params['beta_1'][0], 3))+'\n'
            + r'$\hat{\beta}_2$ : '+str(round(params['beta_2'][0], 3))+'\n'
            + r'$\hat{\beta}_3$ : '+str(round(params['beta_3'][0], 3))+'\n',
        transform=ax_new_2.transAxes, ha='left', va='top', fontsize=15)

    ax_new_2.set_xticks([])
    ax_new_2.set_yticks([])        
    ax_new_2.set_frame_on(False)


    pos3 = 0 * n_ax + 5
    ax_new_3 = fig.add_subplot(n_ax, n_ax, pos3)

    
    table = ax_new_3.table(cellText=inverse_weight_energy.head(1).values, colLabels=inverse_weight_energy.columns,
                  cellLoc="center", loc="upper right", 
                  rowLoc='center',
                 
                  )
    

    table.scale(3, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    ax_new_3.axis('off')

    g.map_lower(corrfunc)

    pos4 = 1 * n_ax + 3
    ax_new_4 = fig.add_subplot(n_ax, n_ax, pos4)
    ax_new_4.text(0,0, "Wasserstein distance " +str(round(wasserstein_dist(gen, df_true, scale=True),3)) +'\n'+"Energy distance " +str(round(energy(gen, df_true, scale=True),3)) ,
    transform=ax_new_4.transAxes, ha='left', va='bottom', fontsize=15)

    ax_new_4.set_xticks([])
    ax_new_4.set_yticks([])        
    ax_new_4.set_frame_on(False)
    
    plt.savefig(str(method)+"_"+type+"_"+data+".pdf")
    print("save")
    
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(args.data, args.all)