import numpy as np
import sys
import inspect

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from keras.callbacks import ModelCheckpoint

sys.path.append("../utilities")
sys.path.append("../../SMAC3")

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import data
X_train, X_test, y_train, y_test, weights_train, _ = data.get_train_test(n=500000)

def print_incumb(cfg):
    print('Best model saved in: ' + '../../models/' \
            + str(cfg['first_kernel_size']) + '_' \
            + str(cfg['conv_filters']) + '_' \
            + str(cfg['n_conv']) + '_' \
            + str(cfg['dropout']) + '_' \
            + cfg['activation'] + '_' \
            + str(cfg['dense_width']) + '_' \
            + str(cfg['dense_length']) + '_' \
            + cfg['optimizer'] + '_' \
            + str(cfg['optimizer_lr']) + '_' \
            + str(cfg['learning_decay_rate']) + '.hdf5')

def cnn_from_cfg(cfg):
  saved_model_path = '../../models/' + str(cfg['first_kernel_size']) + '_' \
                    + str(cfg['conv_filters']) + '_' \
                    + str(cfg['n_conv']) + '_' \
                    + str(cfg['dropout']) + '_' \
                    + cfg['activation'] + '_' \
                    + str(cfg['dense_width']) + '_' \
                    + str(cfg['dense_length']) + '_' \
                    + cfg['optimizer'] + '_' \
                    + str(cfg['optimizer_lr']) + '_' \
                    + str(cfg['learning_decay_rate']) + '.hdf5'

  model = Sequential()

  model.add(Conv2D(cfg['conv_filters'], (cfg['first_kernel_size'], cfg['first_kernel_size']), input_shape=(X_train.shape[1], X_train.shape[2], 1)))
  model.add(Activation(cfg['activation']))

  for i in range(0, cfg['n_conv']-1):
    model.add(Conv2D(cfg['conv_filters']*2, (3, 3)))
    model.add(Activation(cfg['activation']))
    if i % 2 == 1:
        model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())

  for i in range(0, cfg['dense_length']):
    model.add(Dense(cfg['dense_width']))
    model.add(Activation(cfg['activation']))
    model.add(Dropout(cfg['dropout']))

  model.add(Dense(1))
  model.add(Activation(cfg['activation']))

  if cfg['optimizer'] == 'adam':
    opt = Adam(lr=cfg['optimizer_lr'], decay = cfg['learning_decay_rate'])
  elif cfg['optimizer'] == 'sgd':
    opt = SGD(lr=cfg['optimizer_lr'], decay = cfg['learning_decay_rate'])
  elif cfg['optimizer'] == 'nadam':
    opt = Nadam(lr=cfg['optimizer_lr'], schedule_decay = cfg['learning_decay_rate'])
  elif cfg['optimizer'] == 'RMSprop':
    opt = RMSprop(lr=cfg['optimizer_lr'], decay = cfg['learning_decay_rate'])

  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae'])
  model.summary()

  model.fit(X_train, y_train, batch_size=1024, epochs=10,
      validation_split=0.2,
      callbacks=[ModelCheckpoint(saved_model_path, monitor='val_loss',
        verbose=2, save_best_only=True)],
      sample_weight=weights_train)

  return model.evaluate(X_test, y_test)[0]

cs = ConfigurationSpace()

first_kernel_size = CategoricalHyperparameter("first_kernel_size", [3, 11], default_value=3)
conv_filters = CategoricalHyperparameter("conv_filters", [8, 16], default_value=8)
n_conv = CategoricalHyperparameter("n_conv", [3, 5, 7], default_value=5)

dropout = CategoricalHyperparameter("dropout", [0.4, 0.5, 0.6], default_value=0.5)
activation = CategoricalHyperparameter("activation", ['relu', 'sigmoid', 'tanh'], default_value='relu')


dense_width = CategoricalHyperparameter("dense_width", [64, 128], default_value=128)
dense_length = UniformIntegerHyperparameter("dense_length", 1, 3, default_value=2)

optimizer = CategoricalHyperparameter("optimizer", ['adam', 'sgd', 'nadam', 'RMSprop'], default_value='RMSprop')
optimizer_lr = CategoricalHyperparameter("optimizer_lr", [.0001, .0003, .001, .003, .01], default_value=.0003)
learning_decay_rate = UniformFloatHyperparameter("learning_decay_rate", 0, 0.9, default_value=.6)

cs.add_hyperparameters([first_kernel_size, conv_filters, n_conv,
        dropout, activation, dense_width, dense_length,
        optimizer, optimizer_lr, learning_decay_rate])

scenario = Scenario({"run_obj": "quality", "runcount-limit": 128, "cs": cs, "deterministic": "true"})
scenario.output_dir_for_this_run = "C:\\NNwork\\SMAC3out"
scenario.output_dir = "C:\\NNwork\\SMAC3out"
smac = SMAC(scenario=scenario, rng=np.random.RandomState(23), tae_runner=cnn_from_cfg)

print_incumb(smac.optimize())