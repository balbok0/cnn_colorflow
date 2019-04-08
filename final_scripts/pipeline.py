import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


sys.path.append("../utilities")
sys.path.append("../visualization")

import constants
from data import get_train_test, preprocess
from train import train_model
from Pearson import plot_pearson
from metrics import plot_sic, plot_roc, n_pass_hyp
from combine_plots import cp_main

def train(X_train, X_test, y_train, \
                y_test, weights_train, name, usePrev=True):
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
    plt.savefig('final_curves/'+title+'.png')
    plt.savefig('final_curves/'+title+'.pdf')

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

def do_train(datasets, ischarged, usePrev = True, n = 150000):
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

            model_name = sig + '_vs_' + bg
            constants.MODEL_NAME= model_name + '_model'

            X_train, X_test, y_train, y_test, \
            weights_train, weights_test, sig_metadata, \
            bg_metadata, _ = get_train_test(n=n)

            model = train(X_train, X_test, y_train, \
                y_test, weights_train, model_name, usePrev=usePrev)
        
            makeImage(np.mean(X_train[y_train==1.0], axis=0), 'Average_' + sig)
            makeImage(np.mean(X_train[y_train==0.0], axis=0), 'Average_' + bg)

            plot_pearson('../best_model/', 'final_curves/pearsons/', model_name, show_obs=True, provide_data=True, X_test=X_test, y_test=y_test, model=model)

# do hypothesis tests
def do_hyptest(datasets, ischarged):
    n_hyp_tbl = np.zeros((len(datasets), len(datasets))) - 1
    n=1000
    for i in range(6):
        for j in range(6):
            if j >= i:
                continue
            sig = datasets[i]
            bg = datasets[j]

            constants.SIG_H5 = os.path.join(constants.DATA_DIR, sig + '.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, bg + '.h5')

            model_name = sig + '_vs_' + bg
            constants.MODEL_NAME= model_name + '_model'

            X_train, X_test, y_train, y_test, \
            weights_train, weights_test, sig_metadata, \
            bg_metadata, _ = get_train_test(n=n)

            model = train(X_train, X_test, y_train, \
                y_test, weights_train, model_name)

            n_hyp_tbl[i, j] = n_pass_hyp(X_test, y_test, model, flip=0)
            n_hyp_tbl[j, i] = n_pass_hyp(X_test, y_test, model, flip=1)
            
            print(n_hyp_tbl)

def do_perf_plots(datasets, n = 150000):
    def hist(x, title, log = False):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(x, bins = 20, histtype=u'step')  
        if log:
            ax.set_yscale('log')
            plt.title(title + ' (log)', fontsize=14)
        else:
            plt.title(title, fontsize=14)
        plt.savefig('final_curves/hists/'+title+'.png')
        plt.savefig('final_curves/hists/'+title+'.pdf')
        plt.close(fig)

    for i in range(6):
        for j in range(6):
            if j >= i:
                continue
            sig = datasets[i]
            bg = datasets[j]

            constants.SIG_H5 = os.path.join(constants.DATA_DIR, sig + '.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, bg + '.h5')

            model_name = sig + '_vs_' + bg
            constants.MODEL_NAME= model_name + '_model'

            X_train, X_test, y_train, y_test, \
            weights_train, weights_test, sig_metadata, \
            bg_metadata, _ = get_train_test(n=n)

            model = train(X_train, X_test, y_train, \
                y_test, weights_train, model_name, usePrev=True)

            obs_train = calcObs(X_train)
            sig_obs = obs_train[y_train == 1]
            bg_obs = obs_train[y_train == 0]

            name = model_name + '_'
            hist([sig_metadata.iloc[:, 0], bg_metadata.iloc[:, 0]], name+'pull1')
            hist([sig_metadata.iloc[:, 1], bg_metadata.iloc[:, 1]], name+'pull2')
            hist([sig_obs[:, 0], bg_obs[:, 0]], name+'obs1')
            hist([sig_obs[:, 1], bg_obs[:, 1]], name+'obs2')
            hist([sig_obs[:, 2], bg_obs[:, 2]], name+'obs3', log=True)
            hist([sig_obs[:, 3], bg_obs[:, 3]], name+'obs4', log=True)

            obs_test = calcObs(X_test)
            obs_model = adaboost(obs_train, y_train)

            pull_X = np.asarray([np.concatenate(sig_metadata.iloc[:, 0], bg_metadata.iloc[:, 0]), \
                np.concatenate(sig_metadata.iloc[:, 1], bg_metadata.iloc[:, 1])])
            pull_y = np.concatenate(np.ones([sig_metadata.iloc[:, 0].size[0]]), \
                np.ones([bg_metadata.iloc[:, 0]]))
            pull_train, pull_test, y_train_pull, y_test_pull = train_test_split(pull_X, pull_y, train_size=0.8)
            pull_model = adaboost(pull_train, y_train_pull)
            
            plot_sic(model_name, 'final_curves/sic_'+model_name, X_test, y_test, model, use2 = True, X_test2 = obs_test, model2 = obs_model)
            plot_roc(model_name, 'final_curves/roc_'+model_name, X_test, y_test, model, use2 = True, X_test2 = obs_test, model2 = obs_model)


def main(train=True, hyptest=True, histos=True):
    datasets_c = ['h_qq_rot_charged', 'h_gg_rot_charged', 'cp_qq_rot_charged', 'qx_qg_rot_charged', 's8_gg_rot_charged', 'zp_qq_rot_charged']
    datasets_s = ['h_qq', 'h_gg', 'cp_qq', 'qx_qg', 's8_gg', 'zp_qq']

    if train:
        do_train(datasets_s, False)
        do_train(datasets_c, True)

    if hyptest:
        do_hyptest(datasets_s, False)
        do_hyptest(datasets_c, True)

    if histos:
        do_perf_plots(datasets_s)
        do_perf_plots(datasets_c)


if __name__ == '__main__':
  main()
        

