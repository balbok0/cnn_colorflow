import os
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

import train
import data
import metrics
import constants
if constants.THEANO:
    from keras import backend as K
    K.set_image_dim_ordering('th')

def printAverageImage(dat, file, name):
    plt.clf()
    plt.imshow((np.sum(dat, axis=0).reshape(25, 25))/dat.shape[0], interpolation="none", cmap='GnBu')
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title('Average Image ' + name, fontsize=19)
    plt.colorbar()
    plt.savefig(file)

def printdata(name, X_train, X_test, y_train, y_test, weights, direc, recalc=False):
    printAverageImage(X_train, direc + '/image_' + name + '.png', name)
    fname = direc + '/results_' + name + '.txt'
    if os.path.isfile(fname) and not recalc:
        return
    model = train.train_model_save(X_train, X_test, y_train, y_test, weights, direc, name + '.h5', epochs=200)
    f = open(fname, 'w+')
    f.write(name + ': ' + str(metrics.fixed_efficiency(X_test, y_test, model)) + '\n')
    f.close() #save out results

def shape(arr):
    return arr.reshape(arr.shape[0], 625)
def deshape(arr):
    return arr.reshape(1, 25, 25)

def minMax(arr):
    shape(arr)
    arr = (np.max(arr, axis=0) - np.min(arr, axis=0))/2
    arr[arr==0] = 1 #only occurs in pixels that are always 0 anyways
    return deshape(arr)

def safeLog(arr):
    eps = -6
    mask = (arr==0)
    arr = np.log(arr)
    arr[mask] = eps
    arr[arr<eps] = eps
    return 1+arr/6 #put back into reasonable range

#had problems with other norms
def myNorm(arr, ord):
    size = arr.shape[0]
    return np.power(np.sum(np.power(np.abs(arr), ord), axis=0), 1.0/float(size))

def safeNorm(arr, ord): #implement own norm, this one sucks
    arr = shape(arr)
    arr = myNorm(arr, ord)
    arr[arr==0] = 1 #only occurs in pixels that are always 0 anyways
    return deshape(arr)

def safeStd(arr):
    arr = shape(arr)
    arr = np.std(arr, axis = 0) #keep in reasonable range
    arr[arr==0] = 1 #only occurs in pixels that are always 0 anyways
    return deshape(arr)

def safeMean(arr):
    arr = shape(arr)
    size = arr.shape[0]
    arr = np.sum(arr, axis = 0)/size
    return deshape(arr)

def preprocess_tests(preprocess_tests_dir = '../preprocess_tests'):
    constants.DATA_NPY = constants.ROTATED_NPY
    X_train_rot, X_test_rot, y_train_rot, y_test_rot, weights_train_rot, _ = data.get_train_test()
    constants.DATA_NPY = constants.NROTATED_NPY
    X_train_nrot, X_test_nrot, y_train_nrot, y_test_nrot, weights_train_nrot, _ = data.get_train_test()

    printdata('rotated', X_train_rot, X_test_rot, y_train_rot, y_test_rot, weights_train_rot, preprocess_tests_dir)
    printdata('n_rotated', X_train_nrot, X_test_nrot, y_train_nrot, y_test_nrot, weights_train_nrot, preprocess_tests_dir)

    X_train, X_test, y_train, y_test, weights_train = X_train_nrot, X_test_nrot, y_train_nrot, y_test_nrot, weights_train_nrot #use winner of previous test

    X_train_log = safeLog(X_train)
    X_test_log = safeLog(X_test)

    X_train_log_norm1 = X_train_log/safeNorm(X_train_log, 1)
    X_test_log_norm1 = X_test_log/safeNorm(X_train_log, 1)
    X_train_norm1 = X_train/safeNorm(X_train, 1)
    X_test_norm1 = X_test/safeNorm(X_train, 1)

    X_train_log_norm2 = X_train_log/safeNorm(X_train_log, 2)
    X_test_log_norm2 = X_test_log/safeNorm(X_train_log, 2)
    X_train_norm2 = X_train/safeNorm(X_train, 2)
    X_test_norm2 = X_test/safeNorm(X_train, 2)

    X_train_log_std = (X_train_log - safeMean(X_train_log))/safeStd(X_train_log)
    X_test_log_std = (X_test_log - safeMean(X_train_log))/safeStd(X_train_log)
    X_train_std = (X_train - safeMean(X_train))/safeStd(X_train)
    X_test_std = (X_test - safeMean(X_train))/safeStd(X_train)

    X_train_log_mm = (X_train_log - np.min(X_train_log, axis=0))/minMax(X_train_log) - 1
    X_test_log_mm = (X_test_log - np.min(X_train_log, axis=0))/minMax(X_train_log) - 1
    X_train_mm =  (X_train - np.min(X_train, axis=0))/minMax(X_train) - 1
    X_test_mm = (X_test - np.min(X_train, axis=0))/minMax(X_train) - 1

    printdata('norm1_log', X_train_log_norm1, X_test_log_norm1, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('norm1', X_train_norm1, X_test_norm1, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('norm2_log', X_train_log_norm2, X_test_log_norm2, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('norm2', X_train_norm2, X_test_norm2, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('std_log', X_train_log_std, X_test_log_std, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('std', X_train_std, X_test_std, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('mm_log', X_train_log_mm, X_test_log_mm, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('mm', X_train_mm, X_test_mm, y_train, y_test, weights_train, preprocess_tests_dir)
    printdata('log', X_train_log, X_test_log, y_train, y_test, weights_train, preprocess_tests_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate a learning curve.')
    parser.add_argument('--save', default='../preprocess_tests', help='The directory in which models and the curve will be saved.')
    args = parser.parse_args()
    
    preprocess_tests(args.save)

if __name__ == '__main__':
  main()