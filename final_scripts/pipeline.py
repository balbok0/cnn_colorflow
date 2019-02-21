import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../utilities")
sys.path.append("../visualization")

import constants
from data import get_train_test, preprocess
from train import train_model
from Pearson import plot_pearson
from metrics import plot_sic, plot_roc

datasets = ['h_qq', 'h_gg', 'cp_qq', 'qx_qg', 's8_gg', 'zp_qq']
usePrev = True
n = 150000
def main():
    for i in range(6):
        for j in range(6):
            if j >= i:
                continue
            # if j != 0 or i != 1: # for debugging, do only 1 run
            #     break

            sig = datasets[i]
            bg = datasets[j]

            constants.SIG_H5 = os.path.join(constants.DATA_DIR, sig + '.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, bg + '.h5')

            model_dir = 'best_model'
            fig_dir = 'final_figures'
            model_name = sig + ' vs ' + bg

            constants.MODEL_NAME= model_name + '_model'

            X_train, X_test, y_train, y_test, \
            weights_train, weights_test, sig_metadata, \
            bg_metadata, _ = get_train_test(n=n)

            model = train(X_train, X_test, y_train, \
                y_test, weights_train, model_name)
        
            makeImage(np.mean(X_train[y_train==1.0], axis=0), 'Average_' + sig)
            makeImage(np.mean(X_train[y_train==0.0], axis=0), 'Average_' + bg)

            obs_test = calcObs(X_test)
            obs_train = calcObs(X_train)

            obs_model = adaboost(obs_train, y_train)
            
            plot_sic(model_name, 'final_curves/sic_'+model_name, X_test, y_test, model, use2 = True, X_test2 = obs_test, model2 = obs_model)
            plot_roc(model_name, 'final_curves/roc_'+model_name, X_test, y_test, model, use2 = True, X_test2 = obs_test, model2 = obs_model)

            if j == 0 and i == 1:
                plot_pearson('../best_model/', 'final_curves/pearsons/', model_name, show_obs=True, provide_data=True, X_test=X_test, y_test=y_test, model=model)

def train(X_train, X_test, y_train, \
                y_test, weights_train, name):
    if usePrev and os.path.isfile('../best_model/' + name + '_model'):
        from keras.models import load_model
        model = load_model('../best_model/' + name + '_model')
    else:
        model = train_model(X_train, X_test, y_train, \
                y_test, weights_train, '../best_model', epochs=200)

    return model

def makeImage(arr, title):
    plt.clf()
    plt.imshow(np.log(arr+0.0001).reshape(constants.DATA_SIZE, constants.DATA_SIZE), interpolation='none', cmap='GnBu')
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title(title, fontsize=14)
    plt.colorbar()
    plt.savefig('final_curves/'+title)

def calcObs(X):
    n = X.shape[0]
    obs = np.zeros((n, 4))
    for i in range(constants.DATA_SIZE):
        for j in range(constants.DATA_SIZE):
            if i == 32 and j == 32:
                obs[:, 0] = X[:, i, j, 0]
            elif pow(i-32, 2) + pow(j-32, 2) < 36:
                obs[:, 1] = obs[:, 1] + X[:, i, j, 0]
            elif pow(i-32, 2) + pow(j-43, 2) < 25:
                obs[:, 2] = obs[:, 2] + X[:, i, j, 0]
            elif pow(i-32, 2.0)/pow(5.0, 2.0) + pow(j-53, 2.0)/pow(12.0, 2.0) < 1.0:
                obs[:, 3] = obs[:, 3] + X[:, i, j, 0]

    return obs

def adaboost(X, y):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200)
    bdt.fit(X, y)
    return bdt

        
if __name__ == '__main__':
  main()
        

