from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

class GaussianCopula:
    
    def __init__(self, sub_method = "ctgan"):
        self.data =None

    def fit(self, data):
        """Entraîne le modèle CTGAN sur les données réelles."""
        self.data = data

    def generate(self, n_samples):
        metadata_true = Metadata.detect_from_dataframes({"df_true" : self.data})
        synthesizer = GaussianCopulaSynthesizer(metadata_true)
        synthesizer.fit(self.data)
        synthetic_data = synthesizer.sample(num_rows= n_samples)
        return synthetic_data



