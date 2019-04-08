import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys

sys.path.append("../utilities")

import constants
import utils
import data

size = constants.DATA_SIZE

def plot_pearson(run_dir, save_dir, name, show=False, only_true=False, show_obs=False, provide_data=False, X_test=None, y_test=None, model=None):
  if only_true:
    _, X_test, _, y_test, _, _, _, _, _ = data.get_train_test()
  elif not provide_data:
    model, X_test, y_test = utils.get_model_test(run_dir)
  X_test_re = X_test.reshape(X_test.shape[0], size*size)

  y_pearson = np.zeros(X_test_re.shape)
  for i in range(X_test_re.shape[0]):
      y_pearson[i, :] = np.full(X_test_re.shape[1], y_test[i]) 

  X_true = np.zeros(size*size)
  for i in range(size*size):
      X_pearson = np.corrcoef(X_test_re[:, i], y_pearson[:, i]) #, rowvar = False
      X_true[i] = X_pearson[0, 1]
  X_image = X_true.reshape(size, size)

  plt.clf()
  fig,ax = plt.subplots(1)
  plt.imshow(X_image, interpolation="none", cmap='seismic', vmin=-0.2, vmax=0.2)
  plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=10)
  plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=10)
  plt.title('PCC for pixel intensity and truthful output', fontsize=15)
  plt.colorbar()

  if show_obs:
    ax.add_patch(patches.Circle((32,32),1,linewidth=1,edgecolor='g',facecolor='none'))
    ax.add_patch(patches.Circle((32,32),6,linewidth=1,edgecolor='g',facecolor='none'))
    ax.add_patch(patches.Circle((32,43),5,linewidth=1,edgecolor='g',facecolor='none'))
    ax.add_patch(patches.Ellipse((32,53),5,12,linewidth=1,edgecolor='g',facecolor='none'))

  plt.savefig(save_dir + 'truths/' + name + '_pearson_truth.png')
  plt.savefig(save_dir + 'truths/' + name + '_pearson_truth.pdf')
  if show:
    plt.show()

  if(only_true):
    return

  y_hat = model.predict(X_test) > 0.5
  y_pearson = np.zeros(X_test_re.shape)

  for i in range(X_test_re.shape[0]):
      y_pearson[i, :] = np.full(X_test_re.shape[1], y_hat[i])

  X_net = np.zeros(size*size)
  for i in range(size*size):
      X_pearson = np.corrcoef(X_test_re[:, i], y_pearson[:, i]) #, rowvar = False
      X_net[i] = X_pearson[0, 1]
  X_image = X_net.reshape(size, size)

  plt.clf()
  plt.imshow(X_image, interpolation="none", cmap='seismic', vmin=-0.2, vmax=0.2)
  plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=10)
  plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=10)
  plt.title('PCC for pixel intensity and network output', fontsize=15)
  plt.colorbar()
  plt.savefig(save_dir + 'NNs/' + name + '_pearson_nn.png')
  plt.savefig(save_dir + 'NNs/' + name + '_pearson_nn.pdf')
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
  plt.savefig(save_dir + 'diffs/' + name + '_pearson_diff.png')
  plt.savefig(save_dir + 'diffs/' + name + '_pearson_diff.pdf')
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
  plot_pearson(args.run_dir, args.save_dir, False, args.only_true, '')

if __name__ == '__main__':
  main()
