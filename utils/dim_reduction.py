from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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
        
            
    def plot_compare(self,X,T,U,U_reduced):
        fig, ax = plt.subplots(1,3,sharey=True,figsize=(14, 5))
        plt.suptitle('Dataset reconstruction with {0} principal components and {1}% explained variance'.format(self.d,100*self.threshold), y=1.01)

        vmin = U_reduced.min().min()
        vmax = U_reduced.max().max()
        
        surf1 = ax[0].pcolor(X, T, U, cmap=plt.get_cmap("seismic"),shading='auto',vmin=vmin, vmax=vmax)
        fig.colorbar(surf1, ax=ax[0], ticks=None)
        ax[0].set_title('Original data')
        ax[0].set_ylabel('Time')

        surf2 = ax[1].pcolor(X, T, U_reduced,cmap=plt.get_cmap("seismic"),shading='auto',vmin=vmin, vmax=vmax)
        fig.colorbar(surf2, ax=ax[1])
        ax[1].set_title('Reduced data')

        diff =  abs(U-U_reduced)
        surf3 = ax[2].pcolor(X, T, diff,cmap=plt.get_cmap("seismic"),shading='auto') 
        fig.colorbar(surf3, ax=ax[2])
        ax[2].set_title('Absolute difference')
        plt.tight_layout()
        
        return fig, ax, diff