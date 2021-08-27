from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error
from yellowbrick.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from yellowbrick.model_selection import feature_importances
import xgboost as xgb
import eli5
from tqdm import tqdm
from eli5.sklearn import PermutationImportance


def metrics(y_test, predict):

    mse = mean_squared_error(y_test, predict)
    mae = mean_absolute_error(y_test, predict)
    r2 = r2_score(y_test, predict)
    return print("MSE:{}".format(mse), "\nMAE:{}".format(mae), "\nR2:{}".format(r2))

def summary_plot(model,X_train,y_train,X_test,y_test,cv,train_sizes=np.linspace(0.1,1.0,5),lc=False):
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))
    plt.rc('legend',fontsize=12.5) 


    visualize_residuals = residuals_plot(model,X_train, y_train, X_test, y_test,show=False,ax=ax[0],title=' ');
    ax[0].tick_params(labelsize=13)
    ax[0].set_xlabel('Predicted value (mm)',fontsize=16)
    ax[0].set_ylabel('Residuals (mm)',fontsize=16)

    visualizer = prediction_error(model, X_test, y_test, show=False, ax=ax[1], title=' ')
    ax[1].tick_params(labelsize=13)
    ax[1].set_xlabel('Predicted value (mm)',fontsize=16)
    ax[1].set_ylabel('Real value (mm)',fontsize=16)

    if lc:
        print('Plotting learning curves')
        visual_LC = learning_curve(model, X_train, y_train,scoring='r2',cv=cv,ax=ax[2],title=' ',show=False,train_sizes=train_sizes,n_jobs=-1);
        ax[2].set_ylim([0.6, 1.05])
        ax[2].tick_params(labelsize=13)
        ax[2].set_xlabel('Number of training instances',fontsize=16)
        ax[2].set_ylabel(r'$R^2$' + ' metric',fontsize=16)
        
    plt.show()
    
    return


def metrics(y_test,predict):
    
    mse = mean_squared_error(y_test,predict)
    mae= mean_absolute_error(y_test,predict)
    r2= r2_score(y_test,predict)
    return print("MSE:{}".format(mse),"\nMAE:{}".format(mae),"\nR2:{}".format(r2))


def feature_importance(X_train, y_train, X_test, y_test, relative=True, topn=8):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 10), sharex=True)
    plt.rc('legend', fontsize=12.5)

    ################################################################
    # Lasso Regression
    ################################################################

    mod = Lasso(alpha=0.001)
    mod.fit(X_train, y_train)
    r2 = r2_score(y_test, mod.predict(X_test))

    viz1 = FeatureImportances(Lasso(
        alpha=0.001), relative=relative, topn=topn, title=' ', ax=ax[0, 0], absolute=True)
    viz1.fit(X_train, y_train)
    ax[0, 0].tick_params(labelsize=13)
    ax[0, 0].set_title('Lasso Regression - R^2 = {}'.format(r2), fontsize=16)

    ################################################################

    mod = RandomForestRegressor(random_state=0)
    mod.fit(X_train, y_train)
    r2 = r2_score(y_test, mod.predict(X_test))

    rfr = RandomForestRegressor(random_state=0)
    viz2 = FeatureImportances(rfr, relative=relative,
                              topn=topn, title=' ', ax=ax[0, 1])
    viz2.fit(X_train, y_train)
    ax[0, 1].tick_params(labelsize=13)
    ax[0, 1].set_title(
        'RandomForestRegressor - R^2 = {}'.format(r2), fontsize=16)

    ################################################################

    mod = AdaBoostRegressor(random_state=0)
    mod.fit(X_train, y_train)
    r2 = r2_score(y_test, mod.predict(X_test))

    abr = AdaBoostRegressor()
    viz3 = FeatureImportances(abr, relative=relative,
                              topn=topn, title=' ', ax=ax[1, 0])
    viz3.fit(X_train, y_train)
    ax[1, 0].tick_params(labelsize=13)
    ax[1, 0].set_xlabel('Relative feature importance', fontsize=16)
    ax[1, 0].set_title('AdaBoostRegressor - R^2 = {}'.format(r2), fontsize=16)

    ################################################################

    mod = GradientBoostingRegressor(random_state=0)
    mod.fit(X_train, y_train)
    r2 = r2_score(y_test, mod.predict(X_test))

    gbr = GradientBoostingRegressor(random_state=0)
    viz4 = FeatureImportances(
        gbr, relative=relative, topn=topn, title='GradientBoostingRegressor', ax=ax[1, 1])
    viz4.fit(X_train, y_train)
    ax[1, 1].tick_params(labelsize=13)
    ax[1, 1].set_xlabel('Relative feature importance', fontsize=16)
    ax[1, 1].set_title(
        'GradientBoostingRegressor - R^2 = {}'.format(r2), fontsize=16)

    plt.tight_layout()
    plt.show()

    return


class multivariate_importance():
    def __init__(self, X_train, X_test, y_train, y_test, nmodels=6):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.nmodels = nmodels

        mod1 = Lasso()
        mod2 = RandomForestRegressor(random_state=0, n_jobs=-1)
        mod3 = AdaBoostRegressor(random_state=0)
        mod4 = GradientBoostingRegressor(random_state=0)
        mod5 = ExtraTreesRegressor(random_state=0, n_jobs=-1)
        mod6 = xgb.XGBRegressor(
            seed=123, gpu_id=0, tree_method='gpu_hist', random_state=0, n_jobs=-1)

        self.mod_list = [mod1, mod2,
                         mod3, mod4,
                         mod5, mod6]

        self.mod_list = self.mod_list[0:self.nmodels]

        self.model_r2 = None

        print('All models for determining feature importance')
        print(self.mod_list)
        print('')

    def train_models(self):

        model_r2 = []
        for model in tqdm(self.mod_list):
            model.fit(self.X_train, self.y_train)
            model_r2.append(
                np.round(r2_score(self.y_test, model.predict(self.X_test)), 4))

        self.model_r2 = model_r2

        return model_r2

    def permutation_importance(self, model_index=1):

        self.mod_list[model_index].fit(self.X_train, self.y_train)
        perm = PermutationImportance(self.mod_list[model_index], random_state=1).fit(
            self.X_train, self.y_train)
        return eli5.show_weights(perm, feature_names=X_train.columns.tolist())

    def plot(self, relative=True, topn=8, absolute=True, plot_R2=True):

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 18))

        if self.model_r2 == None:
            print('Obtaining R2 score for all 6 models')
            multivariate_importance.train_models(self)
            print('R2 score calculated')

        print('Obtaining feature importance - 0%')
        viz1 = FeatureImportances(
            self.mod_list[0], relative=relative, topn=topn, ax=ax[0, 0], absolute=absolute)
        viz1.fit(self.X_train, self.y_train)
        ax[0, 0].tick_params(labelsize=13)

        viz2 = FeatureImportances(
            self.mod_list[1], relative=relative, topn=topn, ax=ax[0, 1], absolute=absolute)
        viz2.fit(self.X_train, self.y_train)
        ax[0, 1].tick_params(labelsize=13)

        viz3 = FeatureImportances(
            self.mod_list[2], relative=relative, topn=topn, ax=ax[0, 2], absolute=absolute)
        viz3.fit(self.X_train, self.y_train)
        ax[0, 2].tick_params(labelsize=13)
        print('Obtaining feature importance - 50%')
        viz4 = FeatureImportances(
            self.mod_list[3], relative=relative, topn=topn, ax=ax[1, 0], absolute=absolute)
        viz4.fit(self.X_train, self.y_train)
        ax[1, 0].tick_params(labelsize=13)

        viz5 = FeatureImportances(
            self.mod_list[4], relative=relative, topn=topn, ax=ax[1, 1], absolute=absolute)
        viz5.fit(self.X_train, self.y_train)
        ax[1, 1].tick_params(labelsize=13)

        viz6 = FeatureImportances(
            self.mod_list[5], relative=relative, topn=topn, ax=ax[1, 2], absolute=absolute)
        viz6.fit(self.X_train, self.y_train)
        ax[1, 2].tick_params(labelsize=13)
        print('Obtaining feature importance - 100%')

        if plot_R2:

            ax[0, 0].set_title(
                'Lasso Regression - $R^2$ = {}'.format(self.model_r2[0]), fontsize=16)
            ax[0, 1].set_title(
                'RandomForestRegressor - $R^2$ = {}'.format(self.model_r2[1]), fontsize=16)
            ax[0, 2].set_title(
                'AdaBoostRegressor - $R^2$ = {}'.format(self.model_r2[2]), fontsize=16)
            ax[1, 0].set_title(
                'GradientBoostingRegressor - $R^2$ = {}'.format(self.model_r2[3]), fontsize=16)
            ax[1, 1].set_title(
                'ExtraTreesRegressor - $R^2$ = {}'.format(self.model_r2[4]), fontsize=16)
            ax[1, 2].set_title(
                'XGBoost - $R^2$ = {}'.format(self.model_r2[5]), fontsize=16)

            ax[0, 0].set_xlabel('Coefficient value', fontsize=14)
            ax[0, 1].set_xlabel('Coefficient value', fontsize=14)
            ax[0, 2].set_xlabel('Coefficient value', fontsize=14)
            ax[1, 0].set_xlabel('Coefficient value', fontsize=14)
            ax[1, 1].set_xlabel('Coefficient value', fontsize=14)
            ax[1, 2].set_xlabel('Coefficient value', fontsize=14)

        plt.tight_layout()
        return
