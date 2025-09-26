from src.ctgan_wrapper import CTGANWrapper,CTGANSynthcity, CTGANsdv, TabGAN
from src.synthpop_wrapper import SynthpopWrapper
from src.tvae import TVAESynthcity
from src.gaussian_copula_wrapper import GaussianCopula
from src.avatar import Avatar
from random import randint



class Generator:
    def __init__(self, method: str, params_avatar: dict = None, params_ctgan: dict = None, params_synthpop : dict = None,
                params_loop: dict = None, seed=None):
        self.method = method
        self.seed =seed
        self.params_avatar = params_avatar if params_avatar else {}
        self.params_ctgan = params_ctgan if params_ctgan else {}
        self.params_synthpop = params_synthpop if params_synthpop else {}
        self.params_loop = params_loop if params_loop else {}

    def get_generator(self):
        """Retourne la classe de génération appropriée en fonction de la méthode."""
        if self.method == "avatar":
            # Passer les paramètres spécifiques d'avatar
            return Avatar(**self.params_avatar,seed=self.seed)
        
        elif self.method == 'synthpop':
            return SynthpopWrapper(**self.params_synthpop )
        
        elif self.method == "ctgan":
            # Passer les paramètres spécifiques de CTGAN
            return CTGANWrapper(**self.params_ctgan)

        elif self.method == "synthcity_ctgan":
            return CTGANSynthcity(**self.params_ctgan, seed=self.seed)

        elif self.method == "tabgan":
            return TabGAN(seed=self.seed)

        elif self.method == "sdv_ctgan":
            return CTGANsdv(seed = self.seed)

        elif self.method == "gaussian_copula":
            return GaussianCopula()

        elif self.method == "tvae":
            return TVAESynthcity()


        elif self.method == "loop":
            return Loop(**self.params_loop)
    
        else:
            raise ValueError(f"Method '{self.method}' not supported")
        

