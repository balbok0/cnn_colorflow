import os

import numpy as np
from matplotlib import pyplot as plt
from math import log
import sys

import keras
from keras.models import load_model

sys.path.append("../utilities")

import constants
from constants import DATA_SIZE
import utils

# adjust these values to mask Center
X_CENTER = DATA_SIZE // 2
Y_CENTER = DATA_SIZE // 2

# mask size (should be odd)
# if you change this variable, also change edge normalization
MASK_SIZE = 5

# adjust if not odd
MASK_PAD = MASK_SIZE // 2

# Heatmap generation takes a long time.
# With this order, you can begin doing some analysis before the entire script is complete.
CLUSTER_ORDER = [0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19]

def plot_heatmaps(run_dir, cluster_dir, n_clusters):
  model = load_model(os.path.join(run_dir, constants.WEIGHTS_DIR, constants.MODEL_NAME))
  X_test = np.load(cluster_dir + '/test_data_x.npy')
  y_test = np.load(cluster_dir + '/test_data_y.npy')
  clusters = np.load(cluster_dir + '/clusters.npy')

  def jet_image(arr, subtitle, path, vmin=None, vmax=None):
    plt.clf()
    plt.imshow(arr, interpolation='none', cmap='GnBu', vmin=vmin, vmax=vmax)
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title(subtitle, fontsize=14)
    plt.colorbar()
    plt.savefig(cluster_dir + path)

  for k in CLUSTER_ORDER:
    print('PROGRESS: '+ str(k) + ' / ' + str(n_clusters) + ' clusters calculated')
    acc_total = np.zeros((DATA_SIZE, DATA_SIZE))
    mask = (clusters == k)
    xSample = X_test[mask]
    ran = min(xSample.shape[0], 1000)
    for i in range(ran):
      acc = np.zeros((DATA_SIZE+MASK_PAD, DATA_SIZE+MASK_PAD))
      for j in range(DATA_SIZE*DATA_SIZE):
          #format and wrap
          X_test_b = np.pad(xSample[i:(i+1), :, :, :], ((0, 0), (MASK_PAD, MASK_PAD), (MASK_PAD, MASK_PAD), (0, 0)), mode='wrap')

          ver = j % DATA_SIZE
          hor = int(j // DATA_SIZE)

          #block 5 by 5 squares, starting in upper left corner of pad
          #(the first block will center on the first real pixel, not pad)
          for m in range(hor, hor + MASK_SIZE):
              for n in range(ver, ver + MASK_SIZE):
                  if m != X_CENTER or n != Y_CENTER:  # adjust these values to mask center
                      X_test_b[:, m, n, 0] = 0

          #de-wrap
          X_test_b = X_test_b[:, MASK_PAD:DATA_SIZE+MASK_PAD, MASK_PAD:DATA_SIZE+MASK_PAD, :]

          #how good is prediction?
          prediction = np.array(model.predict(X_test_b))[0][0]
          if y_test[np.asarray(np.nonzero(mask))[0, i]] == 0:
              prediction = 1 - prediction
          
          acc[hor:(hor+MASK_SIZE), ver:(ver+MASK_SIZE)] += prediction

      acc = acc[MASK_PAD:DATA_SIZE+MASK_PAD, MASK_PAD:DATA_SIZE+MASK_PAD].reshape(DATA_SIZE, DATA_SIZE)

      #edges need normalized, must be changed when MASK_SIZE is changed
      acc[0, :] *= 5.0/3.0 #normal passes / edge passes
      acc[:, 0] *= 5.0/3.0
      acc[DATA_SIZE-1, :] *= 5.0/3.0
      acc[:, DATA_SIZE-1] *= 5.0/3.0
      acc[1, :] *= 5.0/4.0
      acc[:, 1] *= 5.0/4.0
      acc[DATA_SIZE-2, :] *= 5.0/4.0
      acc[:, DATA_SIZE-2] *= 5.0/4.0

      acc_total += acc

      if i < 3:
        #plot heatmap
        jet_image(np.log(acc), 'Heatmap', '/kmeans_separated/cluster_' + str(k) + '/' + str(i) + '_heat.png', 2.2, 3)
        jet_image(np.log(acc), 'Heatmap', '/kmeans_separated/cluster_' + str(k) + '/' + str(i) + '_heat_autoscale.png')

    jet_image(np.log(acc_total), 'Average Heatmap for cluster', '/kmeans_separated/cluster_' + str(k) + '/average_1000_heat.png', 9.3, 9.65)
    jet_image(np.log(acc_total), 'Average Heatmap for cluster', '/kmeans_separated/cluster_' + str(k) + '/average_1000_heat_autoscale.png')

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Plot clusters on given data.')
  parser.add_argument('--run_dir', default='../best_model', help='The directory in which the model is located.')
  parser.add_argument('--cluster_dir', default='../clusters', help='The directory where clusters.npy was saved.')
  parser.add_argument('--n_clusters', '-n', default=20, help='The number of clusters used.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.make_run_dir()
    print('[heatmaps] New run directory created at {}'.format(args.run_dir))
  plot_heatmaps(args.run_dir, args.cluster_dir, args.n_clusters)

if __name__ == '__main__':
  main()
