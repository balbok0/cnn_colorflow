import os

BASE_DIR = '../../base_dir' #define your own base dir, use None for actual base dir
if BASE_DIR == None:
  if 'BASE_DIR' not in os.environ:
    BASE_DIR = os.path.join('/phys', 'groups', 'tev', 'scratch3', 'users', 'epeml', 'cnn_colorflow') # tev machine base dir
    os.environ['BASE_DIR'] = BASE_DIR
  BASE_DIR = os.environ['BASE_DIR']
if not os.path.exists(BASE_DIR):
  print('[constants] Error: BASE_DIR ({}) does not exist! Please modify your environment to specify BASE_DIR.'.format(BASE_DIR))
  exit()
RUN_DIR = os.path.join(BASE_DIR, 'runs')
DATA_DIR = os.path.join(BASE_DIR, 'samples')

#    old data
#ROTATED_NPY = os.path.join(DATA_DIR, 'rotated.npy')
#NROTATED_NPY = os.path.join(DATA_DIR, 'not_rotated.npy')
#SIZE50_NPY = os.path.join(DATA_DIR, 'size50.npy')
#SIZE100_NPY = os.path.join(DATA_DIR, 'size100.npy')

NEW65_H5 = os.path.join(DATA_DIR, 'both65.h5')

DATA_H5 = NEW65_H5 #change to select default data
WEIGHTS_DIR = 'weights'
TEST_DIR = 'test_samples'
VIS_DIR = 'vis'
MODEL_NAME = 'best_model_cnn.hdf5'
THEANO = False #tell keras to use theano

if not os.path.exists(RUN_DIR):
  os.makedirs(RUN_DIR)
if not os.path.exists(BASE_DIR):
  os.makedirs(BASE_DIR)
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
