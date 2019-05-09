import numpy as np
import sys
import os

sys.path.append("../utilities")
sys.path.append("../visualization")

import constants
from data import get_train_test
from keras.models import load_model
from metrics import plot_n_roc_sic

def sen_stud(datasets, ischarged):
    for i in range(4):
        for j in range(4):
            if j >= i:
                continue

            sig = datasets[i]
            bg = datasets[j]

            if ischarged:
                constants.SIG_H5 = os.path.join(constants.DATA_DIR, sig + '_rot_charged.h5')
                constants.BG_H5 = os.path.join(constants.DATA_DIR, bg + '_rot_charged.h5')
                charge = 'charged'
            else:
                constants.SIG_H5 = os.path.join(constants.DATA_DIR, sig + '.h5')
                constants.BG_H5 = os.path.join(constants.DATA_DIR, bg + '.h5')
                charge = 'standard'

            if ischarged:
                model_name = sig + '_vs_' + bg
            else:
                model_name = sig + '_rot_charged_vs_' + bg + '_rot_charged'
            constants.MODEL_NAME= model_name + '_model'

            _, X_test_14, _, y_test_14, \
            _, _, _, _ = get_train_test(n=150000)

            if not "qx_qg" in model_name:
                constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + sig + '_col_1_' + charge + '.h5')
                constants.BG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + bg + '_col_1_' + charge + '.h5')
                _, X_test_1, _, y_test_1, \
                _, _, _, _ = get_train_test(n=30000, train_size=0)

            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + sig + '_col_2_' + charge + '.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + bg + '_col_2_' + charge + '.h5')
            _, X_test_2, _, y_test_2, \
            _, _, _, _ = get_train_test(n=30000, train_size=0)
            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + sig + '_pp_21_' + charge + '.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + bg + '_pp_21_' + charge + '.h5')
            _, X_test_21, _, y_test_21, \
            _, _, _, _ = get_train_test(n=30000, train_size=0)
            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + sig + '_pp_25_' + charge + '.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + bg + '_pp_25_' + charge + '.h5')
            _, X_test_25, _, y_test_25, \
            _, _, _, _ = get_train_test(n=30000, train_size=0)
            constants.SIG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + sig + '_pp_26_' + charge + '.h5')
            constants.BG_H5 = os.path.join(constants.DATA_DIR, 'sensitivity_study/' + bg + '_pp_26_' + charge + '.h5')
            _, X_test_26, _, y_test_26, \
            _, _, _, _ = get_train_test(n=30000, train_size=0)

            model = load_model('../best_model/' + model_name + '_model')

            if not "qx_qg" in model_name:
                X_tests = [X_test_1, X_test_2, X_test_14, X_test_21, X_test_25, X_test_26]
                y_tests =  [y_test_1, y_test_2, y_test_14,  y_test_21, y_test_25, y_test_26]
                models = [model, model, model, model, model, model]
                model_types = [True, True, True, True, True, True]
                labels = ['Color 1', 'Color 2', 'pp 14', 'pp 21', 'pp 25', 'pp 26']
            else:
                X_tests = [X_test_2, X_test_14, X_test_21, X_test_25, X_test_26]
                y_tests =  [y_test_2, y_test_14,  y_test_21, y_test_25, y_test_26]
                models = [model, model, model, model, model]
                model_types = [True, True, True, True, True]
                labels = ['Color 2', 'pp 14', 'pp 21', 'pp 25', 'pp 26']
            
            plot_n_roc_sic(model_name, 'final_curves/sensitivity_study/sic_sens_'+model_name, X_tests, y_tests, models, model_types, labels, True)
            plot_n_roc_sic(model_name, 'final_curves/sensitivity_study/roc_sens_'+model_name, X_tests, y_tests, models, model_types, labels, False)

def main():
    datasets = ['h_qq', 'h_gg', 'cp_qq', 'qx_qg', 's8_gg', 'zp_qq']
    sen_stud(datasets, True)
    sen_stud(datasets, False)

if __name__ == '__main__':
  main()