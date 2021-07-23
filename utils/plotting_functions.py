import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import ppscore as pps
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.ensemble import RandomForestRegressor
from utils.model_summary_functions import feature_importance, metrics
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, mean_absolute_error, r2_score

def plot_sensor_group(plot_type,data,x,y_cols,hue,hue_order=None,order=None,figsize=(30,20),num_rows=2,num_cols=2,dodge=False):
    
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)

    if num_cols % 2  == 0:

        if plot_type=='lineplot':
            sns.lineplot(data=data,x=x, y=y_cols[0],hue=hue,hue_order=hue_order,ax=ax[0,0])
            sns.lineplot(data=data,x=x, y=y_cols[1],hue=hue,hue_order=hue_order,ax=ax[0,1])
            sns.lineplot(data=data,x=x, y=y_cols[2],hue=hue,hue_order=hue_order,ax=ax[1,0])
            sns.lineplot(data=data,x=x, y=y_cols[3],hue=hue,hue_order=hue_order,ax=ax[1,1])    

        if plot_type=='boxplot':
            sns.boxplot(data=data,x=x, y=y_cols[0], order=order,hue=hue,hue_order=hue_order,ax=ax[0,0])
            sns.boxplot(data=data,x=x, y=y_cols[1], order=order,hue=hue,hue_order=hue_order,ax=ax[0,1])
            sns.boxplot(data=data,x=x, y=y_cols[2], order=order,hue=hue,hue_order=hue_order,ax=ax[1,0])
            sns.boxplot(data=data,x=x, y=y_cols[3], order=order,hue=hue,hue_order=hue_order,ax=ax[1,1])  

    elif num_cols % 2  != 0:
        
        for i in range(num_rows):
            
            if plot_type=='lineplot':
                sns.lineplot(data=data,x=x, y=y_cols[i],hue=hue,hue_order=hue_order,ax=ax[i])
            
            if plot_type=='boxplot':
                sns.boxplot(data=data,x=x, y=y_cols[i], order=order,hue=hue,hue_order=hue_order,ax=ax[i],dodge = dodge)
    return

def dual_heat_map(data,figsize=(25,15),dual=True):
    
    sns.set(font_scale=1.1)
    corr_pearson=data.corr(method='pearson')
    corr_spearman=data.corr(method='spearman')

    mask = np.zeros_like(corr_pearson)
    mask[np.triu_indices_from(mask)] = True

    if dual:
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=figsize)
        sns.heatmap(corr_pearson,cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={"size":14},mask=mask,square=True,ax=ax[0],fmt='.2f',cbar=False)
        sns.heatmap(corr_spearman,cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={"size":14},mask=mask,square=True,ax=ax[1],fmt='.2f',cbar=False)
        ax[0].set_title('Pearson correlation')
        ax[1].set_title('Spearman correlation')
        plt.show()
        
    else:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=figsize)
        sns.heatmap(corr_pearson,cmap="coolwarm", linewidths=0.5, annot=True, annot_kws={"size":14},mask=mask,square=True,fmt='.2f',cbar=False)
        ax.set_title('Pearson correlation')
        plt.show()  
    
    return
    
    
def scatterplot_pearson(df, x_vars, y_vars, cmap='viridis',hue='Class'):
    sns.set(font_scale=1.9)
    g = sns.PairGrid(df, hue=hue, x_vars=x_vars,y_vars=y_vars, palette=cmap, corner=False,height=7,aspect=1)
    g.map_diag(sns.histplot, color='.5')
    g.map_offdiag(sns.scatterplot, s=5,alpha=0.7)
    g.tight_layout()
    g.add_legend()
    return g


def pps_heat_map(data, figsize=(30, 15)):
    corr = pps.matrix(data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, annot=True)
    plt.title('Power predictive score (PPS)')
    plt.show()
    return


def plot_selectkbest(X_train,X_test,y_train,y_test,step,model,score_func = f_regression):
    
    k_vs_score=[]
    range = np.arange(2,X_train.shape[1]+1,step)
    
    for k in range:
        selector = SelectKBest(score_func=score_func, k=k)
        X_new2 = selector.fit_transform(X_train,y_train)
        X_val = selector.transform(X_test)
        
        mdl = model
        mdl.fit(X_new2,y_train)
        
        p=mdl.predict(X_val)
        print('k = {} '.format(k))
        score = metrics(y_test,p)
        k_vs_score.append(r2_score(y_test,p))

    plt.plot(range,k_vs_score)  
    return 