import os
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append("../utilities")

import constants
import utils
import data

def plot_histograms(run_dir, metadata_sig, metadata_bg):
  #trim to even size
  if metadata_bg.shape[0] > metadata_sig.shape[0]:
    metadata_bg = metadata_bg[:metadata_sig.shape[0], :]
  else:
    metadata_sig = metadata_sig[:metadata_bg.shape[0], :]
    
  for j in range(0, 4):
    if j == 0:
      name = 'pull1'
    elif j == 1:
      name = 'pull2'
    elif j == 2:
      name = 'jet_mass'
    elif j == 3:
      name = 'jet_delta_R'

    hist, bins = np.histogram(metadata_sig[:, j], bins = 100)
    plt.plot(bins[:-1], hist, drawstyle='steps-post', color='blue', label='qq')
    hist, bins = np.histogram(metadata_bg[:, j], bins = 100)
    plt.plot(bins[:-1], hist, drawstyle='steps-post', color='red', label='gg')

    plt.title(name)
    plt.legend(loc='upper right')
    plt.savefig(run_dir+name+'.png')
    plt.clf()


def main():
  import argparse
  parser = argparse.ArgumentParser(description='Plot histograms on given data.')
  parser.add_argument('--run_dir', default='../histograms/', help='The directory in which histogram plots should be saved.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.make_run_dir()
    print('[clustering] New run directory created at {}'.format(args.run_dir))

  _, metadata_sig = data.get_pixels_metadata(octet=False)
  _, metadata_bg = data.get_pixels_metadata(octet=True)
  plot_histograms(args.run_dir, np.array(metadata_sig), np.array(metadata_bg))

if __name__ == '__main__':
  main()
