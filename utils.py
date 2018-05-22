import os
import datetime
import glob

import numpy as np
from keras.models import load_model

import constants
if constants.THEANO:
  from keras import backend as K
  K.set_image_dim_ordering('th')

def make_run_dir():
  timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
  run_dir = os.path.join(constants.RUN_DIR, timestamp)
  try:
    os.makedirs(run_dir)
  except OSError as e:
    print(e)
  return run_dir

def most_recent_dir():
  return max(glob.glob(os.path.join(constants.RUN_DIR, '*')))

def get_model_test(run_dir):
  model = load_model(os.path.join(run_dir, constants.WEIGHTS_DIR, constants.MODEL_NAME))
  X_test = np.load(os.path.join(run_dir, constants.TEST_DIR, 'X_test.npy'))
  y_test = np.load(os.path.join(run_dir, constants.TEST_DIR, 'y_test.npy'))
  return model, X_test, y_test
