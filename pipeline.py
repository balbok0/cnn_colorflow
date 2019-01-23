import numpy
import sys
sys.path.append("../utilities")

from constants import SIG_H5, BG_H5, MODEL_NAME
from data import get_train_test
from train import train_model

datasets = ['']

for i in range(6):
    for j in range(6):
        if j >= i:
            continue
        sig = datasets[i]
        bg = datasets[j]
        
        SIG_H5, BG_H5, MODEL_NAME= 'depends'

        X_train, X_test, y_train, y_test, \
        weights_train, weights_test, sig_metadata, \
        bg_metadata, y = get_train_test()

        model_dir = 'depends'
        model = train_model(X_train, X_test, y_train, \
                            y_test, weights_train, model_dir, epochs=200)

        

        

