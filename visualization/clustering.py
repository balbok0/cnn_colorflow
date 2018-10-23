import os
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import sys

sys.path.append("../utilities")

import constants
import utils
import data

def plot_clusters(run_dir, X, y, n_clusters, separate_octet_singlet, picWidth=65):
  if separate_octet_singlet:
      title = 'kmeans_separated'
  else:
    title = 'kmeans'

  if os.path.isfile(run_dir + '/clusters.npy'):
    prediction = np.load(run_dir + '/clusters.npy')
  else:
    if separate_octet_singlet:
      mask = (y == 1)
      X_singlet = X[mask]
      X_octet = X[np.logical_not(mask)]
      cluster_o = KMeans(n_clusters=n_clusters/2, verbose=1).fit(X_octet)
      cluster_s = KMeans(n_clusters=n_clusters/2, verbose=1).fit(X_singlet)
      prediction = np.concatenate((cluster_o.labels_, cluster_s.labels_ + 10))
    else:
      cluster = KMeans(n_clusters=n_clusters).fit(X)
      prediction = cluster.labels_
    np.save(run_dir + '/clusters.npy', prediction)

  def jet_image(arr, subtitle, path):
    plt.clf()
    plt.imshow(np.log(arr).reshape(picWidth, picWidth), interpolation='none', cmap='GnBu')
    plt.xlabel('Proportional to Translated Pseudorapidity', fontsize=7)
    plt.ylabel('Proportional to Translated Azimuthal Angle', fontsize=7)
    plt.title(subtitle, fontsize=14)
    plt.colorbar()
    plt.savefig(path)

  for i in range(n_clusters):
    mask = (prediction == i)
    X_sample = X[mask]
    n_singlet = np.sum(y[mask])
    n_octet = np.sum(1 - y[mask])
    jet_image(np.sum(X_sample, axis=0), 'Cluster {} #oct={}, #sing={}'.format(str(i), str(n_octet), str(n_singlet)), '{}/{}/cluster_{}/average.png'.format(run_dir, title, str(i)))
    # Representative sample.
    for j in range(0, 3):
      jet_image(X_sample[j], 'Cluster {}, pic {}'.format(str(i), str(j)), '{}/{}/cluster_{}/{}.png'.format(run_dir, title, str(i), str(j)))

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Plot clusters on given data.')
  parser.add_argument('--run_dir', default='../clusters', help='The directory in which cluster plots should be saved.')
  parser.add_argument('--n_clusters', '-n', default=20, help='The number of clusters to use.')
  parser.add_argument('--separate', '-s', default=True, action='store_true', help='If set, separate octet and singlet data for clustering.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.make_run_dir()
    print('[clustering] New run directory created at {}'.format(args.run_dir))

  X, _, y, _, _, _ = data.get_train_test()
  plot_clusters(args.run_dir, X.reshape((X.shape[0], X.shape[2]*X.shape[3])), y, args.n_clusters, args.separate)

if __name__ == '__main__':
  main()
