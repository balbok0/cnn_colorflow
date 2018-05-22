import os
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

import train
import data
import metrics
import constants
if constants.THEANO:
    from keras import backend as K
    K.set_image_dim_ordering('th')

def lcurve(lcurve_model_dir, total_data_size, step_size=10, min_size=10, max_size=100, recalc=False):
    bins = (max_size-min_size)/step_size
    x = np.zeros(bins)
    y = np.zeros(bins)
    index = 0
    for i in range(min_size, max_size, step_size):
        sample_size = (i*total_data_size)/100
        X_train, X_test, y_train, y_test, weights_train, _ = data.get_train_test(n=sample_size)
        modelFileName = lcurve_model_dir + '/learning' + str(sample_size) + '.h5'
        if os.path.isfile(modelFileName) and not recalc:
            model = load_model(modelFileName)
        else:
            model = train.train_model(X_train, X_test, y_train, y_test, weights_train, lcurve_model_dir, epochs=50)
            model.save(modelFileName)
        y[index] = metrics.fixed_efficiency(X_test, y_test, model)
        x[index] = sample_size
        index = index + 1
    plt.plot(x, y)
    plt.xlabel('Samples Used', fontsize=15)
    plt.ylabel('fpr with tpr=0.5', fontsize=15)
    plt.title('Learning Curve', fontsize=19)
    plt.savefig(lcurve_model_dir + '/lcurve.png')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate a learning curve.')
    parser.add_argument('--save', default=None, help='The directory in which models and the curve will be saved.')
    parser.add_argument('--step_size', type=int, default=10, help='The step size, as a percentage (i.e. step_size = 5 means 5% of total data).')
    parser.add_argument('--min_size', type=int, default=10, help='The min size of data to use, as a percentage.')
    args = parser.parse_args()

    X_train, X_test, _, _, _, _ = data.get_train_test()
    total_size = X_train.shape[0] + X_test.shape[0]
    lcurve(args.save, total_size, args.step_size, args.min_size)

if __name__ == '__main__':
  main()