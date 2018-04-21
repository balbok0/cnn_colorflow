import os

if 'BASE_DIR' not in os.environ:
  BASE_DIR = os.path.join('/phys', 'groups', 'tev', 'scratch3', 'users', 'epeml', 'cnn_colorflow') # tev machine base dir
  os.environ['BASE_DIR'] = BASE_DIR
if not os.path.exists(BASE_DIR):
  print('[constants] Error: BASE_DIR ({}) does not exist! Please modify your environment to specify BASE_DIR.'.format(BASE_DIR))
  exit()
RUN_DIR = os.path.join(BASE_DIR, 'runs')
DATA_DIR = os.path.join(BASE_DIR, 'samples')
SINGLET_TEXT = os.path.join(DATA_DIR, 'Singlet_Rotated_withDR.txt')
SINGLET_NPY = os.path.join(DATA_DIR, 'Singlet_Rotated_withDR.npy')
OCTET_TEXT = os.path.join(DATA_DIR, 'Octet_Rotated_withDR.txt')
OCTET_NPY = os.path.join(DATA_DIR, 'Octet_Rotated_withDR.npy')
WEIGHTS_DIR = 'weights'
TEST_DIR = 'test_samples'
VIS_DIR = 'vis'
MODEL_NAME = 'best_model_cnn.hdf5'

if not os.path.exists(RUN_DIR):
  os.makedirs(RUN_DIR)
if not os.path.exists(BASE_DIR):
  os.makedirs(BASE_DIR)
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)
