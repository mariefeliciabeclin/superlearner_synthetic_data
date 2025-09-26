import numpy as np
import numpy.random as rand
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class Avatar():
    def __init__(self, ncp=4, k=12, n_mi=1, seed=None):
        self.ncp = ncp
        self.k = k
        self.n_mi = n_mi
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        
    def new_avatar_point(self, x_i, neighbour : list) : 
        """_summary_

        Args:
        x_i (_type_): _description_
        neighbor (list): _description_

        Returns:
            _type_: _description_
        """
  
        d = [ np.linalg.norm(x_j-x_i) for x_j in neighbour] 
        r = self.rng.exponential(scale=1, size=len(neighbour))
        sigma = self.rng.permutation(len(neighbour))
        P = [(1/2)**(sigma[i]+1) * (1/d[i])*r[i] for i in range( len(neighbour))]
        
        new_a = np.average(neighbour, axis =0,weights = P)
        
        return new_a
        
    def avatar(self, T):
     
        scaler = preprocessing.StandardScaler().fit(T)
        T_scale = scaler.transform(T)

        dim = np.shape(T)[1]
        
        
        pca = PCA(n_components=self.ncp)
        T_latent_reduced = pca.fit_transform(T_scale)

    
        clf = NearestNeighbors()
        clf = clf.fit(T_latent_reduced)
   
        neigh_dist, neigh_ind = clf.kneighbors(n_neighbors=self.k)


        new_data = np.array([
        self.new_avatar_point(T.iloc[i].values, T.iloc[neigh_ind[i]].values)
        for i in range(len(T))
            ])

        return pd.DataFrame(new_data, columns=T.columns)
        

    def fit(self, T):
        self.data = T
        print("seed avatar")
        print(self.seed)

    def generate(self, n_sample):
        if self.n_mi  == 1:
            return self.avatar(self.data)
        else :
            # voir si on ne peut pas utiliser pipe
            return [self.avatar(self.data) for i in range(self.n_mi)]


        















