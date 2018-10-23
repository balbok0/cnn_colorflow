import os
import numpy as np
from matplotlib import pyplot as plt
import sys

sys.path.append("../utilities")

import constants
import utils
import data

size = 65

def plot_pearson(run_dir, save_dir, show=False, only_true=False):
  if not only_true:
    model, X_test, y_test = utils.get_model_test(run_dir)
  else:
    _, X_test, _, y_test, _, _ = data.get_train_test()
    X_test = X_test.reshape(X_test.shape[0], size*size)

  y_pearson = np.zeros(X_test.shape)
  for i in range(X_test.shape[0]):
      y_pearson[i, :] = np.full(X_test.shape[1], y_test[i]) 

  X_true = np.zeros(size*size)
  for i in range(size*size):
      X_pearson = np.corrcoef(X_test[:, i], y_pearson[:, i]) #, rowvar = False
      X_true[i] = X_pearson[0, 1]
  X_image = X_true.reshape(size, size)

  plt.clf()
  plt.imshow(X_image, interpolation="none", cmap='seismic', vmin=-0.2, vmax=0.2)
  plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=10)
  plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=10)
  plt.title('PCC for pixel intensity and truthful output', fontsize=15)
  plt.colorbar()
  plt.savefig(os.path.join(save_dir, 'Pearson_truth.png'))
  if show:
    plt.show()

  if(only_true):
    return

  y_hat = model.predict(X_test) > 0.5
  X_test = X_test.reshape(X_test.shape[0], size*size)
  y_pearson = np.zeros(X_test.shape)

  for i in range(X_test.shape[0]):
      y_pearson[i, :] = np.full(X_test.shape[1], y_hat[i])

  X_net = np.zeros(size*size)
  for i in range(size*size):
      X_pearson = np.corrcoef(X_test[:, i], y_pearson[:, i]) #, rowvar = False
      X_net[i] = X_pearson[0, 1]
  X_image = X_net.reshape(size, size)

  plt.imshow(X_image, interpolation="none", cmap='seismic', vmin=-0.2, vmax=0.2)
  plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=10)
  plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=10)
  plt.title('PCC for pixel intensity and network output', fontsize=15)
  plt.colorbar()
  save_dir = os.path.join(run_dir, 'Pearson')
  plt.savefig(os.path.join(save_dir, 'Pearson_net.png'))
  if show:
    plt.show()
  
  X_image = X_net - X_true
  X_image = X_image.reshape(size, size)

  plt.clf()
  plt.imshow(X_image, interpolation="none", cmap='seismic', vmin=-0.2, vmax=0.2)
  plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=10)
  plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=10)
  plt.title('Difference between net and true PCCs', fontsize=15)
  plt.colorbar()
  plt.savefig(os.path.join(save_dir, 'Pearson_diff.png'))
  if show:
    plt.show()

  X_image = np.divide(X_net, X_true)
  X_image = X_image.reshape(size, size)

  plt.clf()
  plt.imshow(X_image, interpolation="none", cmap='seismic', vmin=-8, vmax=10)
  plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=10)
  plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=10)
  plt.title('Net divided by True', fontsize=15)
  plt.colorbar()
  plt.savefig(os.path.join(save_dir, 'Pearson_div.png'))
  if show:
    plt.show()

  print('[Pearson] Done!')

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Load a given model and perform Pearson coefficient analysis and visualization.')
  parser.add_argument('--run_dir', default=None, help='The run directory that should be used (see train.py). If unspecified, the most recent run directory is used.')
  parser.add_argument('--save_dir', default='./', help='Where to save images. Defaults to this folder.')
  parser.add_argument('--only_true', default=False, help='Use only the truth value, without the need for training a network. Default is False.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.most_recent_dir()
    print('[Pearson] run_dir not specified, using {}'.format(args.run_dir))
  plot_pearson(args.run_dir, args.save_dir, False, args.only_true)

if __name__ == '__main__':
  main()
