import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


class SynthpopWrapper():
    def __init__(self, n_mi=1, sub_method =None):
        """Initialise le wrapper pour le package synthpop."""
        # Active la conversion automatique entre les dataframes Pandas et R
        pandas2ri.activate()
        # Charger le package synthpop
        ro.r('library(synthpop)')
        self.sub_method = sub_method
     

    def fit(self, data):
        """Prépare les données pour la synthèse en R."""
        # Convertir le dataframe Pandas en dataframe R
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        self.data = pandas2ri.py2rpy(data)

    def generate(self, n_samples=100):
        """Génère des données synthétiques."""
        # Passer le dataframe à R
        ro.globalenv['data'] = self.data
        
        # Construire la commande pour la synthèse
        if self.sub_method is not None:
            # Si des méthodes sont spécifiées, les appliquer
            methods = ', '.join([f"'{m}'" for m in self.sub_method])
            print(f'syn(data, method = c({methods}))')
            synth_data = ro.r(f'syn(data, method = c({methods}), minnumlevels=3)')
        else:
            # Utiliser les méthodes par défaut
            synth_data = ro.r('syn(data)')

        # Convertir le dataframe R de sortie en dataframe Pandas
        synthetic_data = pandas2ri.rpy2py_dataframe(synth_data.rx2('syn'))
        return synthetic_data.sample(n_samples).to_numpy()  # Échantillonner les données synthétiques
