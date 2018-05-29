import numpy as np
import pandas as pd
import os

import constants
import data
import utils

def train_model_save(X_train, X_test, y_train, y_test, weights_train, model_dir, file_name, dropout=0.5, dense_width=125, conv_width=16, batch_size=1024, epochs=200, recalc=False):
  from keras import backend as K
  K.set_image_dim_ordering('th')
  from keras.models import load_model

  modelFileName = model_dir + '/' + file_name
  if os.path.isfile(modelFileName) and not recalc:
      model = load_model(modelFileName)
  else:
      model = train_model(X_train, X_test, y_train, y_test, weights_train, model_dir, dropout, dense_width, conv_width, batch_size, epochs)
      model.save(modelFileName)

  return model

def train_model(X_train, X_test, y_train, y_test, weights_train, model_dir, dropout=0.5, dense_width=125, conv_width=16, batch_size=1024, epochs=200):
  """ Return the model trained on the given data with the given weights.
  Return:
  model -- the keras model trained on the given data.

  Arguments:
  X_train -- the training data.
  X_test -- the testing data.
  y_train -- labels for the training data.
  y_test -- labels for the testing data.
  weights_train -- weights for the training data.
  model_dir -- the directory in which the trained model should be saved.
  dropout -- the dropout probability.
  dense_width -- the width of the dense layers.
  conv_width -- the width of the first conv layer, and half the width of the 
                second conv layer.
  batch_size -- the batch size used in training the model.
  epochs -- the number of epochs to run when training.
  """
  print('[train] Loading keras ...')

  from keras import backend as K
  K.set_image_dim_ordering('th')

  from keras.models import Sequential
  from keras.layers.core import Dense, Dropout, Activation, Flatten
  from keras.layers.convolutional import MaxPooling2D, Conv2D
  from keras.optimizers import RMSprop
  from keras.callbacks import ModelCheckpoint

  print('[train] Building model...')

  model = Sequential()

  model.add(Conv2D(conv_width, (11, 11), input_shape=(1, 25, 25), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(conv_width * 2, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())

  model.add(Dense(dense_width))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))

  model.add(Dense(dense_width))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(optimizer=RMSprop(lr=0.0003), loss='binary_crossentropy', metrics=['mae'])
  model.summary()

  saved_model_path = os.path.join(model_dir, constants.MODEL_NAME)
  model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
      validation_split=0.2,
      callbacks=[ModelCheckpoint(saved_model_path, monitor='val_loss',
        verbose=2, save_best_only=True)],
      sample_weight=weights_train)
  return model

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Compile, train and save a model.')
  parser.add_argument('--run_dir', default=None, help='The directory in which weights and test samples should be saved.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.make_run_dir()
    print('[test] New run directory created at {}'.format(args.run_dir))
  X_train, X_test, y_train, y_test, weights_train, _ = data.get_train_test()
  test_dir = os.path.join(args.run_dir, constants.TEST_DIR)
  try:
    os.makedirs(test_dir)
  except OSError as e:
    print(e)
  X_test_path = os.path.join(test_dir, 'X_test.npy')
  y_test_path = os.path.join(test_dir, 'y_test.npy')
  weights_dir = os.path.join(args.run_dir, constants.WEIGHTS_DIR)
  try:
    os.makedirs(weights_dir)
  except OSError as e:
    print(e)
  np.save(X_test_path, X_test)
  np.save(y_test_path, y_test)
  train_model(X_train, X_test, y_train, y_test, weights_train, 
      weights_dir)

if __name__ == '__main__':
  main()
