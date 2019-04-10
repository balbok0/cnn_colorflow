import numpy as np
import sys
import os
from keras.models import load_model

sys.path.append("../utilities")
import constants
from data import get_train_test
from metrics import plot_n_roc_sic

datasets_c = ['h_qq_rot_charged', 'h_gg_rot_charged', 'cp_qq_rot_charged', 'qx_qg_rot_charged', 's8_gg_rot_charged', 'zp_qq_rot_charged']
datasets_s = ['h_qq', 'h_gg', 'cp_qq', 'qx_qg', 's8_gg', 'zp_qq']

def comp_all(i, datasets = datasets_s, n = 150000):
    name = 'all_' + datasets[i] + '_comps'
    X_tests = []
    y_yests = []
    models = []
    model_types = []
    labels = []

    sig = datasets[i]
    for j in range(6):
        if j == i:
            continue
        bg = datasets[j]

        constants.SIG_H5 = os.path.join(constants.DATA_DIR, sig + '.h5')
        constants.BG_H5 = os.path.join(constants.DATA_DIR, bg + '.h5')

        X_train, X_test, y_train, y_test, \
        _, _, sig_metadata, \
        bg_metadata, _ = get_train_test(n=n)

        if os.path.isfile('../best_model/' + sig + '_vs_' + bg + '_model'):
          model_name = sig + '_vs_' + bg
        else:
          model_name = bg + '_vs_' + sig
        model = load_model('../best_model/' + model_name + '_model')
        X_tests.append(X_test)
        y_yests.append(y_test)
        models.append(model)
        model_types.append(True)
        labels.append(model_name)
        
    plot_n_roc_sic(name, 'final_curves/sic_'+name, X_tests, y_yests, models, model_types, labels, True, fontfac=0.5)
    plot_n_roc_sic(name, 'final_curves/roc_'+name, X_tests, y_yests, models, model_types, labels, False, fontfac=0.5)

if __name__ == '__main__':
  for i in range(len(datasets_s)):
    comp_all(i)
        