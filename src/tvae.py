from synthcity.plugins import Plugins


class TVAESynthcity:
    def __init__(self, sub_method = "tvae"):
        self.data =[]
        self.sub_method = sub_method
        Plugins(categories=["generic", "privacy"]).list()
    
    def fit(self, data):
        """Entraîne le modèle CTGAN sur les données réelles."""
        self.data = data

    def generate(self, n_samples):
        syn_model = Plugins().get("tvae")
        syn_model.fit(self.data)
        gen = syn_model.generate(count = n_samples)
        return gen.dataframe()