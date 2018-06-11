import os
from keras.models import load_model
import numpy as np

import data
import constants
if constants.THEANO:
    from keras import backend as K
    K.set_image_dim_ordering('th')
import preprocess_tests

def pixelSize_tests(pixelSize_tests_dir = '../pixelSize_tests'):
    #get data
    constants.DATA_NPY = constants.SIZE50_NPY
    X_train_50, X_test_50, y_train_50, y_test_50, weights_train_50, _ = data.get_train_test()
    constants.DATA_NPY = constants.NROTATED_NPY
    X_train_25, X_test_25, y_train_25, y_test_25, weights_train_25, _ = data.get_train_test()

    #preprocess
    X_train_50, X_test_50 = preprocess_tests.logAnd1Norm(X_train_50, X_test_50)
    X_train_25, X_test_25 = preprocess_tests.logAnd1Norm(X_train_25, X_test_25)

    #calculate results
    preprocess_tests.printdata('size50', X_train_50, X_test_50, y_train_50, y_test_50, weights_train_50, pixelSize_tests_dir)
    preprocess_tests.printdata('size25', X_train_25, X_test_25, y_train_25, y_test_25, weights_train_25, pixelSize_tests_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test images of different sizes')
    parser.add_argument('--save', default='../pixelSize_tests', help='The directory in which models and the curve will be saved.')
    args = parser.parse_args()
    
    pixelSize_tests(args.save)

if __name__ == '__main__':
  main()