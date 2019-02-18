import numpy as np
import sys
import os
from matplotlib import pyplot as plt

sys.path.append("../utilities")
sys.path.append("../visualization")

import constants
from data import get_train_test, preprocess
from train import train_model
from metrics import plot_sic, plot_roc
from Pearson import plot_pearson

usePrev = True
def main():
    for cmp in range(4):
        if cmp == 0:
            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'h_qq_charged_rot_nocut.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'h_gg_charged_rot_nocut.h5')
            sample = 'charged'
            cmps = ' qq vs gg'
        elif cmp == 1:
            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'h_qq_standard_rot_nocut.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'h_gg_standard_rot_nocut.h5')
            sample = 'standard'
            cmps = ' qq vs gg'
        elif cmp == 3:
            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'h_qq_standard_rot_nocut.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'h_qq_charged_rot_nocut.h5')
            sample = 'quarks'
            cmps =  ' charged v standard'
        else:
            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'h_gg_standard_rot_nocut.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'h_gg_charged_rot_nocut.h5')
            sample = 'gluons' 
            cmps = ' charged v standard'

        constants.MODEL_NAME= sample + '_model'
        X_train, X_test, y_train, y_test, \
        weights_train, weights_test, sig_metadata, \
        bg_metadata, _ = get_train_test(n=150000) #same_file=True)

        train(X_train, X_test, y_train, \
                y_test, weights_train, sample, cmps)
        
        makeImage(np.mean(X_train[y_train==1.0], axis=0), 'Average_' + sample + '_quark')
        makeImage(np.mean(X_train[y_train==0.0], axis=0), 'Average_' + sample + '_gluon')

def train(X_train, X_test, y_train, \
                y_test, weights_train, name, cmps):
    if usePrev:
        from keras.models import load_model
        model = load_model('../best_model/' + name + '_model')
    else:
        model = train_model(X_train, X_test, y_train, \
                y_test, weights_train, '../best_model', epochs=200)

    plot_sic(name + cmps, 'charged_curves/sic_'+name, X_test, y_test, model)
    plot_roc(name + cmps, 'charged_curves/roc_'+name, X_test, y_test, model)

def makeImage(arr, title):
    plt.clf()
    plt.imshow(np.log(arr+0.0001).reshape(constants.DATA_SIZE, constants.DATA_SIZE), interpolation='none', cmap='GnBu')
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title(title, fontsize=14)
    plt.colorbar()
    plt.savefig('charged_curves/'+title)

if __name__ == '__main__':
  main()