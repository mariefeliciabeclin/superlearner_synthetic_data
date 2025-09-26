from ctgan import CTGAN
import numpy as np
import random
from tabgan.sampler import OriginalGenerator, GANGenerator

class CTGANWrapper:
    def __init__(self, epochs=100):
        """Initialise le modèle CTGAN avec les paramètres spécifiés."""
        self.epochs = epochs
        self.model = CTGAN(epochs=self.epochs)
        

    def fit(self, data):
        """Entraîne le modèle CTGAN sur les données réelles."""
        self.data = data 

    def generate(self, n_samples):
        """Génère des données synthétiques."""
        gen = self.model
        gen.fit(self.data)
        return gen.sample(n_samples)


from synthcity.plugins import Plugins


class CTGANSynthcity:
    def __init__(self, seed, sub_method = "ctgan"):
        self.sub_method = sub_method
        self.seed =seed
        Plugins(categories=["generic", "privacy"]).list()
    
    def fit(self, data):
        """Entraîne le modèle CTGAN sur les données réelles."""
        self.data = data

    def generate(self, n_samples):
        syn_model = Plugins().get("ctgan", random_state=self.seed)
        syn_model.fit(self.data)
        gen = syn_model.generate(count = n_samples)
        return gen.dataframe()

#from sdv.tabular import CTGAN

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

class CTGANsdv:
    def __init__(self, seed=None, sub_method="ctgan"):
        self.seed = seed
        self.sub_method = sub_method
        self.model = None

    def fit(self, data):
        """Entraîne le modèle CTGAN sur les données réelles."""
        self.data = data

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        metadata_true = Metadata.detect_from_dataframes({"df_true": self.data})
        self.model = CTGANSynthesizer(metadata_true, epochs=300)
        self.model.fit(self.data)  #i

    def generate(self, n_samples):
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de générer des données.")
        synthetic_data = self.model.sample(num_rows=n_samples)
        return synthetic_data

from sklearn.model_selection import train_test_split

class TabGAN:
    def __init__(self, seed=None, test_size=0.2):
        self.seed = seed
        self.test_size = test_size
        self.model = None
        self.train_data = None
        self.test_data = None

    def fit(self, data):
        """Sépare les données en train/test et entraîne le modèle GAN sur les données d'entraînement."""
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            print('seed tabgan')
            print(self.seed)

        # Séparation en données d'entraînement et de test
        self.train_data, self.test_data = train_test_split(
            data, test_size=self.test_size, random_state=self.seed, 
        )

        # Initialisation et entraînement du modèle GAN
      

    def generate(self, n_samples):
        """Génère des données synthétiques à partir des données d'entraînement."""

        self.model = GANGenerator(
            gen_x_times=0.26, cat_cols=None,
            bot_filter_quantile=0.001, top_filter_quantile=0.999, is_post_process=False,
            adversarial_model_params={
                "metrics": "AUC", "max_depth": 2, "max_bin": 100,
                "learning_rate": 0.02, "random_state": self.seed, "n_estimators": 100,
            },
            pregeneration_frac=1, only_generated_data=True,
            gen_params={"batch_size": 500, "patience": 25, "epochs": 500},
        )

        synthetic_data, target = self.model.generate_data_pipe(
            train_df=self.train_data, target=None,
            test_df=self.test_data, deep_copy=True,
            only_adversarial=False, use_adversarial=True
        )

        synthetic_data = synthetic_data.iloc[:n_samples].reset_index(drop=True)

        print(len(synthetic_data))
        return synthetic_data