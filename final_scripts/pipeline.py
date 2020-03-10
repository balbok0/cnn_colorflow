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
from data import get_train_test
from train import train_model
from Pearson import plot_pearson
from metrics import plot_n_roc_sic, n_pass_hyp
from combine_plots import cp_main

def setConstants(sig, bg):
    constants.SIG_H5 = os.path.join(constants.DATA_DIR, sig + '.h5')
    constants.BG_H5 = os.path.join(constants.DATA_DIR, bg + '.h5')

    model_name = sig + '_vs_' + bg
    constants.MODEL_NAME= model_name + '_model'

    return model_name

def train(X_train, X_test, y_train, \
                y_test, weights_train, name, usePrev=True):
    from keras.models import load_model
    if (not usePrev) or not os.path.isfile('../best_model/' + name + '_model'):
        train_model(X_train, X_test, y_train, \
                y_test, weights_train, '../best_model', epochs=200)
    model = load_model('../best_model/' + name + '_model')

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

def calcObs(X, num=10):
    n = X.shape[0]
    radiuses = np.linspace(1, constants.DATA_SIZE, num=num)

    def isInTelescope(i, j, r):
        dist = (i-32)**2 + (j-32)**2
        if r > 1:
            rad_prev = radiuses[r-1]
        else:
            rad_prev = 0
        return ((dist <= radiuses[r]**2) and (dist > rad_prev**2))
    
    obs = np.zeros((n, num))
    for r in range(num):
        for i in range(constants.DATA_SIZE):
            for j in range(constants.DATA_SIZE):
                if isInTelescope(i, j, r):
                    obs[:, r] = obs[:, r] + X[:, i, j, 0]
    return obs


def adaboost(X, y):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200)
    bdt.fit(X, y)
    return bdt

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

def pipeline(datasets, ischarged, usePrev = True, skip = False, n = 150000):
    n_hyp_tbl = np.zeros((len(datasets), len(datasets))) - 1
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            if j >= i:
                continue

            sig = datasets[i]
            bg = datasets[j]

            model_name = setConstants(sig, bg)

            if skip \
               and os.path.exists('y_vals/y_pull_hat_'+model_name+'.npy') \
               and os.path.exists('final_curves/pearsons/truths/'+model_name+'_pearson_truth.png.npy') \
               and os.path.exists('../best_model/' + sig + '_vs_' + bg + '_model'):
                print('Skipped '  + sig + ' vs ' + bg + '!')
                continue


            X_train, X_test, y_train, y_test, \
            weights_train, weights_test, sig_metadata, \
            bg_metadata = get_train_test(n=n)

            model = train(X_train, X_test, y_train, \
                y_test, weights_train, model_name, usePrev=usePrev)
        
            makeImage(np.mean(X_train[y_train==1.0], axis=0), 'Average_' + sig)
            makeImage(np.mean(X_train[y_train==0.0], axis=0), 'Average_' + bg)

            plot_pearson('../best_model/', 'final_curves/pearsons/', model_name, show_obs=True, provide_data=True, X_test=X_test, y_test=y_test, model=model)

            obs_train = calcObs(X_train)
            sig_obs = obs_train[y_train == 1]
            bg_obs = obs_train[y_train == 0]

            name = model_name + '_'
            hist([sig_metadata.iloc[:, 0], bg_metadata.iloc[:, 0]], name+'pull1')
            hist([sig_metadata.iloc[:, 1], bg_metadata.iloc[:, 1]], name+'pull2')
            for k in range(10):
                hist([sig_obs[:, k], bg_obs[:, k]], name+'obs'+str(k+1))
                np.save('final_curves/tjets/' + sig + '_obs' + str(k+1), sig_obs[:, k])
                np.save('final_curves/tjets/' + bg + '_obs' + str(k+1), bg_obs[:, k])

            hist([sig_obs[:, 1], bg_obs[:, 1]], name+'obs2')
            hist([sig_obs[:, 2], bg_obs[:, 2]], name+'obs3', log=True)
            hist([sig_obs[:, 3], bg_obs[:, 3]], name+'obs4', log=True)

            obs_test = calcObs(X_test)
            obs_model = adaboost(obs_train, y_train)

            pull1 = np.concatenate((sig_metadata.iloc[:, 0], bg_metadata.iloc[:, 0]))
            pull2 = np.concatenate((sig_metadata.iloc[:, 1], bg_metadata.iloc[:, 1]))
            pull_X = np.concatenate((pull1.reshape(pull1.shape[0], 1), pull2.reshape(pull2.shape[0], 1)), axis=1)
            pull_y = np.concatenate((np.ones(len(sig_metadata.iloc[:, 0])), np.zeros(len(bg_metadata.iloc[:, 0]))))
            pull_train, pull_test, y_train_pull, y_test_pull = train_test_split(pull_X, pull_y, train_size=0.8)
            pull_model = adaboost(pull_train, y_train_pull)
            
            X_tests = [X_test, obs_test, pull_test]
            y_tests = [y_test, y_test, y_test_pull]
            models = [model, obs_model, pull_model]
            model_types = [True, False, False]
            labels = ['CNN', 'OBS', 'Pull']
            plot_n_roc_sic(model_name, 'final_curves/sic_'+model_name, X_tests, y_tests, models, model_types, labels, True)
            plot_n_roc_sic(model_name, 'final_curves/roc_'+model_name, X_tests, y_tests, models, model_types, labels, False)

            n_hyp_tbl[i, j] = n_pass_hyp(X_test[:1000, ...], y_test[:1000], model, flip=0)
            n_hyp_tbl[j, i] = n_pass_hyp(X_test[:1000, ...], y_test[:1000], model, flip=1)

            # save all y's
            np.save('y_vals/y_nn_test_'+model_name, y_test)
            y_hat = model.predict(X_test)
            np.save('y_vals/y_nn_hat_'+model_name, y_hat)

            np.save('y_vals/y_obs_test_'+model_name, y_test)
            obs_hat = obs_model.predict_proba(obs_test)
            np.save('y_vals/y_obs_hat_'+model_name, obs_hat[:, 1])
            
            np.save('y_vals/y_pull_test_'+model_name, y_test_pull)
            pull_hat = pull_model.predict_proba(pull_test)
            np.save('y_vals/y_pull_hat_'+model_name, pull_hat[:, 1])

    print(n_hyp_tbl)

def main(combine = False, n = 150000):
    datasets_c = ['h_qq_rot_charged', 'h_gg_rot_charged', 'cp_qq_rot_charged', 'qx_qg_rot_charged', 's8_gg_rot_charged', 'zp_qq_rot_charged',  'six_jj_rot_charged', 'x2_jj_rot_charged']
    datasets_s = ['h_qq', 'h_gg', 'cp_qq', 'qx_qg', 's8_gg', 'zp_qq', 'six_jj', 'x2_jj']
    datasets_untrimmed = ['cp_qq_untrim_standard', 'zp_qq_untrim_standard']
    datasets_untrimmed_charged = ['cp_qq_untrim_charged', 'zp_qq_untrim_charged']

    #pipeline(datasets_s, False, n=n, skip=False)
    #pipeline(datasets_c, True, n=n, skip=False)
    pipeline(datasets_untrimmed, False, n=n, skip=False)
    pipeline(datasets_untrimmed_charged, True, n=n, skip=False)
        
    if combine:
        cp_main()

if __name__ == '__main__':
  main()
        

