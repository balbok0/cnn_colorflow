import numpy as np
import sys
import os
from matplotlib import pyplot as plt

sys.path.append("../utilities")
sys.path.append("../visualization")

import constants
from data import get_train_test, preprocess
from train import train_model
from Pearson import plot_pearson
from metrics import plot_sic, plot_roc

datasets = ['h_qq', 'h_gg', 'cp_qq', 'qx_qg', 's8_gg', 'zp_qq']
usePrev = False
def main():
    for i in range(6):
        for j in range(6):
            if j >= i:
                continue
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
            bg_metadata, _ = get_train_test(n=150000)

            train(X_train, X_test, y_train, \
                y_test, weights_train, model_name)
        
            makeImage(np.mean(X_train[y_train==1.0], axis=0), 'Average_' + sig)
            makeImage(np.mean(X_train[y_train==0.0], axis=0), 'Average_' + bg)

def train(X_train, X_test, y_train, \
                y_test, weights_train, name):
    if usePrev and os.path.isfile('../best_model/' + name + '_model'):
        from keras.models import load_model
        model = load_model('../best_model/' + name + '_model')
    else:
        model = train_model(X_train, X_test, y_train, \
                y_test, weights_train, '../best_model', epochs=200)

    plot_sic(name, 'final_curves/sic_'+name, X_test, y_test, model)
    plot_roc(name, 'final_curves/roc_'+name, X_test, y_test, model)

def makeImage(arr, title):
    plt.clf()
    plt.imshow(np.log(arr+0.0001).reshape(constants.DATA_SIZE, constants.DATA_SIZE), interpolation='none', cmap='GnBu')
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title(title, fontsize=14)
    plt.colorbar()
    plt.savefig('final_curves/'+title)


        
if __name__ == '__main__':
  main()
        

