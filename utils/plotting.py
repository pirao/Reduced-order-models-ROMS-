import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import ppscore as pps
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.ensemble import RandomForestRegressor
from utils.model_summary_functions import feature_importance, metrics
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, mean_absolute_error, r2_score

def plot_samples(df, n_cols=2,n_rows=2):
    fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(n_rows,n_cols,sharex=True,sharey=True,figsize=(24, 7))

    ax1.plot(df[df.columns[0]], label = df.columns[0])
    ax1.set_ylabel('Eigenvector ' + str(df.columns[0]))

    ax2.plot(df[df.columns[-2]], label = df.columns[-2])
    ax2.set_ylabel('Eigenvector ' + str(df.columns[-2]))

    ax4.plot(df[df.columns[1]], label = df.columns[1])
    ax4.set_ylabel('Autovector ' + str(df.columns[1]))
    ax4.set_xlabel('Number of timesteps')

    ax5.plot(df[df.columns[-1]], label = df.columns[-1])
    ax5.set_ylabel('Autovector ' + str(df.columns[-1]))
    ax5.set_xlabel('Number of timesteps')
    
    plt.tight_layout()
    plt.show()
    
def plot_autocorrelation(df,i=0,zoom=False,acf_lim=100,pacf_lim=20):
    fig, ax = plt.subplots(1,2,figsize=(25, 5))
    
    plot_acf(df[df.columns[i]],lags=len(df)-1,ax=ax[0]);
    ax[0].set_title('Autocorrelation - Eigenvector ' + str(i))
    ax[0].set_xlabel('Number of lags')
    ax[0].set_ylabel('Pearson coefficient')

    plot_pacf(df[df.columns[i]], lags=200,method='ywmle',alpha=.5,ax=ax[1])
    ax[1].set_title('Partial autocorrelation - Eigenvector ' + str(i))
    ax[1].set_xlabel('Number of lags')
    ax[1].set_ylabel('Pearson coefficient for residual')
    
    if zoom==True:
        ax[0].set_xlim([0,acf_lim])
        ax[1].set_xlim([0,pacf_lim])
        
    return    

def dual_heat_map(data,figsize=(25,15)):
    
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=figsize)
    sns.set(font_scale=1.1)

    corr_pearson=data.corr(method='pearson')
    corr_spearman=data.corr(method='spearman')

    mask = np.zeros_like(corr_pearson)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr_pearson,cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={"size":14},mask=mask,square=True,cbar=False,ax=ax[0],fmt='.2f')
    sns.heatmap(corr_spearman,cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={"size":14},mask=mask,square=True,cbar=False,ax=ax[1],fmt='.2f')

    ax[0].set_title('Pearson correlation')
    ax[1].set_title('Spearman correlation')
    plt.show()
    return
    
    
def scatterplot_pearson(df, x_vars, y_vars, cmap='viridis'):
    sns.set(font_scale=1.9)
    g = sns.PairGrid(df, hue="Class", x_vars=x_vars,y_vars=y_vars, palette=cmap, corner=False,height=7,aspect=1)
    g.map_diag(sns.histplot, color='.5')
    g.map_offdiag(sns.scatterplot, s=5,alpha=0.7)
    g.tight_layout()
    g.add_legend()
    return g


def pps_heat_map(data, figsize=(30, 15)):
    corr = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(
        columns='x', index='y', values='ppscore')
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, annot=True)
    plt.title('Power predictive score (PPS)')
    plt.show()
    return


def plot_selectkbest(X_train, X_test, y_train, y_test, step, model, score_func=f_regression):

    k_vs_score = []
    range = np.arange(2, X_train.shape[1]+1, step)

    for k in range:
        selector = SelectKBest(score_func=score_func, k=k)
        X_new2 = selector.fit_transform(X_train, y_train)
        X_val = selector.transform(X_test)

        mdl = model
        mdl.fit(X_new2, y_train)

        p = mdl.predict(X_val)
        print('k = {} '.format(k))
        score = metrics(y_test, p)
        k_vs_score.append(r2_score(y_test, p))

    plt.plot(range, k_vs_score)
    return
