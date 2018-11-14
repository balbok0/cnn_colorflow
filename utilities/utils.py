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

def merge(in1, in2, out):
  inFile1 = np.load(in1)
  inFile2 = np.load(in2)
  print('Input1 Shape: ' + str(inFile1.shape))
  print('Input2 Shape: ' + str(inFile2.shape))

  outFile = np.concatenate((inFile1, inFile2))
  print('Output Shape: ' + str(outFile.shape))

  np.save(out, outFile)

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Utility Functions.')
  parser.add_argument('--fName', default=None, help='Utility function to run')
  parser.add_argument('--args', nargs='+', default=None, help='List of arguments')
  args = parser.parse_args()
  
  if args.fName == 'merge':
    if len(args.args) != 3:
      print('Invalid Number of Arguments')
    else:
      merge(constants.BASE_DIR + '/samples/' + args.args[0],
            constants.BASE_DIR + '/samples/' + args.args[1],
            constants.BASE_DIR + '/samples/' + args.args[2])
  else:
    print('Invalid Function Name')

if __name__ == '__main__':
  main()