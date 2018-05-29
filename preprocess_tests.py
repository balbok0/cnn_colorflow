import os
from keras.models import load_model
import numpy as np
from numpy import linalg as LA

import train
import data
import metrics
import constants
if constants.THEANO:
    from keras import backend as K
    K.set_image_dim_ordering('th')

def printdata(name, X_train, X_test, y_train, y_test, weights, direc, recalc=False):
    fname = direc + '/results_' + name + '.txt'
    if os.path.isfile(fname) and not recalc:
        return
    model = train.train_model_save(X_train, X_test, y_train, y_test, weights, direc, name + '.h5', epochs=50)
    f = open(fname, 'w+')
    f.write(name + ': ' + str(metrics.fixed_efficiency(X_test, y_test, model)) + '\n')
    f.close() #save out results

def safer(arr):
    eps = 1e-10
    arr[np.abs(arr)<eps]=eps
    return arr

def cap(arr):
    top = 10
    arr[arr>top]=top
    arr[arr<-top]=-top
    return arr

def safelog(arr):
    arr[arr==0]=1e-10
    arr = cap(arr)
    arr = safer(arr)
    return arr

def safenorm(arr):
    eps = 1
    top = 10
    arr[arr<eps]=eps
    arr[arr>top]=top
    return arr

def safedev(arr):
    eps = 1
    top = 100
    arr[arr<eps]=eps
    arr[arr>top]=top
    return arr

def safeminmax(arr):
    eps = 1e-1
    top = 1000
    arr[arr<eps]=eps
    arr[arr>top]=top
    return arr

def minMax(arr):
    return (arr - np.min(arr, axis=0)) / safeminmax(((np.max(arr, axis=0) - np.min(arr, axis=0))/2)) - 1

def preprocess_tests(preprocess_tests_dir = '../preprocess_tests'):
    constants.DATA_NPY = constants.ROTATED_NPY
    X_train_rot, X_test_rot, y_train_rot, y_test_rot, weights_train_rot, _ = data.get_train_test()
    constants.DATA_NPY = constants.NROTATED_NPY
    X_train_nrot, X_test_nrot, y_train_nrot, y_test_nrot, weights_train_nrot, _ = data.get_train_test()

    printdata('rotated', X_train_rot, X_test_rot, y_train_rot, y_test_rot, weights_train_rot, preprocess_tests_dir)
    printdata('n_rotated', X_train_nrot, X_test_nrot, y_train_nrot, y_test_nrot, weights_train_nrot, preprocess_tests_dir)

    X_train, X_test, y_train, y_test, weights_train = X_train_rot, X_test_rot, y_train_rot, y_test_rot, weights_train_rot #use winner of previous test
    X_train = safer(X_train)
    X_test = safer(X_test)
    X_train_log = safelog(np.log(X_train))
    X_test_log = safelog(np.log(X_test))

    X_train_log_norm1 = X_train_log/safenorm(LA.norm(X_train_log, ord=1, axis=0))
    X_test_log_norm1 = X_test_log/safenorm(LA.norm(X_test_log, ord=1, axis=0))
    X_train_norm1 = X_train/safenorm(LA.norm(X_train, 1, axis=0))
    X_test_norm1 = X_test/safenorm(LA.norm(X_test, 1, axis=0))

    X_train_log_norm2 = X_train_log/safenorm(LA.norm(X_train_log, 2, axis=0))
    X_test_log_norm2 = X_test_log/safenorm(LA.norm(X_test_log, 2, axis=0))
    X_train_norm2 =  X_train/safenorm(LA.norm(X_train, 2, axis=0))
    X_test_norm2 = X_test/safenorm(LA.norm(X_test, 2, axis=0))

    X_train_log_std = X_train_log/safedev(np.std(X_train_log, axis = 0)) - np.mean(X_train_log, axis = 0)
    X_test_log_std = X_test_log/safedev(np.std(X_test_log, axis = 0)) - np.mean(X_test_log, axis = 0)
    X_train_std =  X_train/safedev(np.std(X_train, axis = 0)) - np.mean(X_train, axis = 0)
    X_test_std = X_test/safedev(np.std(X_test, axis = 0)) - np.mean(X_test, axis = 0)

    X_train_log_mm = X_train_log/minMax(X_train_log)
    X_test_log_mm = X_test_log/minMax(X_test_log)
    X_train_mm =  X_train/minMax(X_train)
    X_test_mm = X_test/minMax(X_test)

    printdata('norm1_log', X_train_log_norm1, X_test_log_norm1, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('norm1', X_train_norm1, X_test_norm1, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('norm2_log', X_train_log_norm2, X_test_log_norm2, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('norm2', X_train_norm2, X_test_norm2, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('std_log', X_train_log_std, X_test_log_std, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('std', X_train_std, X_test_std, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('mm_log', X_train_log_mm, X_test_log_mm, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('mm', X_train_mm, X_test_mm, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('log', X_train_log, X_test_log, y_train, y_test, weights_train, preprocess_tests_dir)

    f.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate a learning curve.')
    parser.add_argument('--save', default='../preprocess_tests', help='The directory in which models and the curve will be saved.')
    args = parser.parse_args()
    
    preprocess_tests(args.save)

if __name__ == '__main__':
  main()