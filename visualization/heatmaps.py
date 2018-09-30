import os

import numpy as np
from matplotlib import pyplot as plt
from math import log
import sys

sys.path.append("../utilities")

import constants
import utils

def plot_heatmaps(run_dir, cluster_dir, n_clusters):
  model, X_test, y_test = utils.get_model_test(run_dir)
  clusters = np.load(cluster_dir + 'clusters.npy')

  def jet_image(arr, subtitle, path, vmin=None, vmax=None):
    plt.clf()
    plt.imshow(arr, interpolation='none', cmap='GnBu', vmin=vmin, vmax=vmax)
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.titel(subtitle, fontsize=14)
    plt.colorbar()
    plt.savefig(cluster_dir + path)

  for k in range(n_clusters):
    print 'PROGRESS: '+ str(k) + ' / ' + str(sample_n) + ' clusters calculated'
    acc_total = np.zeros((25, 25))
    mask = (clusters == k)
    xSample = X_test[mask]
    ran = min(xSample.shape[0], 1000) # make reasonable timescale
    for i in range(ran):
      acc = np.zeros((29, 29))
      for j in range(625):
          #format and wrap
          X_test_b = np.pad(xSample[i:(i+1), :, :, :], ((0, 0), (0, 0), (2, 2), (2, 2)), mode='wrap')

          ver = j % 25
          hor = int(j // 25)

          #block 5 by 5 squares, starting in upper left corner of pad
          #(the first block will center on the first real pixel, not pad)
          for m in range(hor, hor + 5):
              for n in range(ver, ver + 5):
                  if m != 14 or n != 14: 
                      X_test_b[:, 0, m, n] = 0

          #de-wrap and pad
          X_test_b = np.lib.pad(X_test_b[:, :, 2:27, 2:27], ((0, 0), (0, 0), (0, 7), (0, 7)), 'constant', constant_values=0)

          #how good is prediction?
          prediction = np.array(model.predict(X_test_b))[0][0]
          if y[np.asarray(np.nonzero(mask))[0, i]] == 0:
              prediction = 1 - prediction
          
          acc[hor:(hor+5), ver:(ver+5)] += prediction

      acc = acc[2:27, 2:27].reshape(25, 25)

      #edges need normalized
      acc[0, :] *= 5.0/3.0 #normal passes / edge passes
      acc[:, 0] *= 5.0/3.0
      acc[24, :] *= 5.0/3.0
      acc[:, 24] *= 5.0/3.0
      acc[1, :] *= 5.0/4.0
      acc[:, 1] *= 5.0/4.0
      acc[23, :] *= 5.0/4.0
      acc[:, 23] *= 5.0/4.0

      acc_total += acc

      if i < 3:
        #plot heatmap
        jet_image(np.log(acc), 'Heatmap', 'kmeans_separated/cluster_' + str(k) + '/' + str(i) + '_heat.png', 2.2, 3)
        jet_image(np.log(acc), 'Heatmap', 'kmeans_separated/cluster_' + str(k) + '/' + str(i) + '_heat_autoscale.png')

    jet_image(np.log(acc_total), 'Average Heatmap for cluster', 'kmeans_separated/cluster_' + str(k) + '/average_1000_heat.png', 9.3, 9.65)
    jet_image(np.log(acc_total), 'Average Heatmap for cluster', 'kmeans_separated/cluster_' + str(k) + '/average_1000_heat_autoscale.png')

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Plot clusters on given data.')
  parser.add_argument('--run_dir', default=None, help='The directory in which cluster plots should be saved.')
  parser.add_argument('--cluster_dir', default=None, help='The directory where clusters.npy was saved.')
  parser.add_argument('--n_clusters', '-n', default=20, help='The number of clusters used.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.make_run_dir()
    print('[heatmaps] New run directory created at {}'.format(args.run_dir))
  plot_heatmaps(args.run_dir, args.cluster_dir, args.n_clusters)

if __name__ == '__main__':
  main()
