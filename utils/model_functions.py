import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

class kmeans():
    def __init__(self, df,cat=False):
        
        self.X = df
        if cat:
            self.y = df.loc[:, 'Class']

        self.model =None
        
        
    def cluster_criterion(self,n_clusters):
        ine = []
        sil = []
        calinski_harabasz = []
        davies_bouldin = []
        
        n_range = range(2, n_clusters + 1)

        for i in tqdm(n_range):
            kmeans = KMeans(n_clusters = i, max_iter = 3000, random_state = 42, init='k-means++')
            kmeans.fit(self.X)
            ine.append(kmeans.inertia_)
            sil.append(silhouette_score(self.X, kmeans.predict(self.X)))
            calinski_harabasz.append(calinski_harabasz_score(self.X, kmeans.predict(self.X)))
            davies_bouldin.append(davies_bouldin_score(self.X, kmeans.predict(self.X)))
            
            print(f'Number of groups: {i} --- Inertia: {ine[i-2]} --- Silhouette coefficient: {sil[i-2]}')
            print(f'Number of groups: {i} --- Inertia: {ine[i-2]} --- calinski_harabasz: {calinski_harabasz[i-2]}')
            print(f'Number of groups: {i} --- Inertia: {ine[i-2]} --- davies_bouldin: {davies_bouldin[i-2]}')
            
        
        fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex=True, figsize = (26, 13))
        
        ax[0,0].plot(n_range, ine)
        ax[0,0].set_xticks(n_range)
        ax[0,0].set_ylabel("Inertia")
            
        ax[0,1].plot(n_range, sil)
        ax[0,1].set_xticks(n_range)
        ax[0,1].set_ylabel("Silhouette coefficient")
        

        ax[1,0].plot(n_range, calinski_harabasz)
        ax[1,0].set_xticks(n_range)
        ax[1,0].set_ylabel("Calinski harabasz score")
        ax[1,0].set_xlabel("Number of groups")
            
        ax[1,1].plot(n_range, davies_bouldin)
        ax[1,1].set_xticks(n_range)
        ax[1,1].set_xlabel("Number of groups")
        ax[1,1].set_ylabel("Davies bouldin score")

        plt.tight_layout()
        plt.show()
    
    def metrics(self,n_clusters):
        
        self.silhouette_score = silhouette_score(self.X, self.model.predict(self.X))
        self.calinski_score = calinski_harabasz_score(self.X, self.model.predict(self.X))
        self.davies = davies_bouldin_score(self.X, self.model.predict(self.X))
        
        print('Number of groups: {0} --- Silhouette coefficient: {1}'.format(n_clusters,self.silhouette_score))
        print('Number of groups: {0} --- Calinski harabasz score: {1}'.format(n_clusters,self.calinski_score))
        print('Number of groups: {0} --- Davies bouldin score: {1}'.format(n_clusters,self.davies))        
        
    def create_model(self, n_clusters):
        self.model = KMeans(n_clusters = n_clusters, max_iter = 5000, random_state = 42)
        self.model.fit(self.X)
        print(metrics(n_clusters))
        
    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))
  
    def load_model(self, path):
        kmeans = pickle.load(open(path, 'rb'))
        self.model = kmeans.model
    
    def clustering(self):
        if not self.model:
            print("Create a clustering model first")
            
        self.Y_ = self.model.predict(self.X)
        cluster_data = self.X.copy()
        cluster_data['cluster_kmeans'] = self.Y_
        
        return cluster_data
    
    
class pca():
    def __init__(self,U,threshold):
        self.U = U
        self.threshold = threshold
        
        self.model = PCA()
        self.model.fit(self.U)
        self.cumsum = np.cumsum(self.model.explained_variance_ratio_)
        self.d = np.argmax(self.cumsum >= self.threshold) + 1
        
        self.model_red = PCA(n_components=self.d)
        
        
    def save_model(self, path):
        pickle.dump(self.model_red, open(path, 'wb'))
        
    def load_model(self, path):
        self.model_red = pickle.load(open(path, 'rb'))
        
    def plot_pca(self):
      
        plt.plot(self.cumsum * 100,label='Train dataset')
        plt.xlabel('Number of principal components (PCs)')
        plt.ylabel('Explained cumulative variance (%)')
        
        plt.vlines(x=self.d, ymin=0, ymax=100*self.threshold,
                    label = '{0} principal components with {1}% explained variance'.format(self.d,100*self.threshold),
                    linestyles='dashed',
                    color='red')
        
        plt.legend(fontsize='16')
        plt.show()    
        