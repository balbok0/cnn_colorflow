import os
import numpy as np 
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

import constants

def get_pixels_metadata(octet=False, n=-1, delta_R_min=float('-inf'), delta_R_max=float('inf'), recalculate=False):
  """Return pixel data and metadata for either the octets or singlets.

  Return:
  pixels -- a (n, 625) numpy array of the pixel data.
  metadata -- a (n, 4) pandas dataframe containing all other data, such as
              mass, jet pull, and delta R.

  Arguments:
  octet -- true (false) if the octet (singlet) data should be collected.
  n -- the number of samples to collect. If n == -1, all samples will be collected.
  delta_R_min -- the minimum delta R allowed for a sample to be included.
  delta_R_max -- the maximum delta R allowed for a sample to be included.
  
  Recalulates from the original text files if necessary or if recalculate is
  true.
  The pixel data is a (n, 625) numpy array.
  The metadata is a (n, 4) pandas array.
  """
  if octet:
    print("[data] Getting octet pixel data ...")
  else:
    print("[data] Getting singlet pixel data ...")
  h5file = constants.DATA_H5
  # Get the appropriate numpy array, either from a saved .npy file or
  # from the original .txt file.
  if recalculate:
    # Remove any row with a nan value.
    if (np.isnan(data).any()):
      print("[data] Warning: non-numeric values encountered, removing affected samples.")
      data = data[~np.isnan(data).any(axis=1)]
    with open(npyfile, 'wb+') as npyfile_handle:
      np.save(npyfile_handle, data)
  else:
    print("[data] Loading from {} ...".format(h5file))
    data = h5py.File(h5file, 'r')

  mask = (data['meta_variables/signal'][()] > 0)
  size = mask.shape[0]

  metadata = np.zeros((size, 4))
  metadata[:, 0] = data['meta_variables/pull1'][()]
  metadata[:, 1] = data['meta_variables/pull2'][()]
  metadata[:, 2] = data['meta_variables/jet_mass'][()]
  metadata[:, 3] = data['meta_variables/jet_delta_R'][()]

  pixels = data['images'][()]
  if octet:
    mask = ~mask
  pixels = pixels[mask]
  metadata = metadata[mask]
  if n != -1:
    frac = float(np.sum(mask)) / float(size)
    pixels = pixels[:int(n*frac)]
    metadata = metadata[:int(n*frac)]

  metadata = pd.DataFrame(metadata, columns=['pull_1', 'pull_2', 'mass', 'delta_R'])
  # Restrict delta R
  pixels = pixels[np.where((metadata['delta_R'] <= delta_R_max) & (metadata['delta_R'] >= delta_R_min))]
  print("[data] {} pixels shape: {}".format("octet" if octet else "singlet", pixels.shape))
  print("[data] {} metadata head:\n {}".format("octet" if octet else "singlet", metadata.head()))
  return pixels, metadata

def get_train_test(n=-1, delta_R_min=float("-inf"), delta_R_max=float("inf"),  weighting_mass=False, train_size=0.8):
  """Returns X, y, and weight arrays for training and testing.

  Return:
  X_train -- numpy array of shape (train_size * n, 1, 32, 32)
  X_test -- numpy array of shape ((1 - train_size) * n, 1, 32, 32)
  y_train -- numpy array of shape (train_size * n)
  y_test -- numpy array of shape ((1 - train_size) * n)
  weights_train -- numpy array of shape (train_size * n)
  weights_test -- numpy array of shape ((1 - train_size) * n)
  
  Arguments:
  n -- same as in get_pixels_metadata()
  delta_R_min -- same as in get_pixels_metadata()
  delta_R_max -- same as in get_pixels_metadata()
  weighting_mass -- if true, modify weights such that when the samples are
                    binned by mass, the number of weighted singlet samples in
                    each bin is equivalent to the number of octet samples.
  """
  bg_pixels, bg_metadata = get_pixels_metadata(octet=True, n=n, delta_R_min=delta_R_min, delta_R_max=delta_R_max)
  sig_pixels, sig_metadata = get_pixels_metadata(octet=False, n=n, delta_R_min=delta_R_min, delta_R_max=delta_R_max)
  sig_weights = np.ones(sig_pixels.shape[0])
  # Calculate weights.
  if weighting_mass:
    mass_min = 0
    mass_max = 400
    mass_num_bins = 100
    mass_bins = np.linspace(mass_min, mass_max, mass_num_bins)
    for i in range(1, mass_num_bins):
      bg_bin = (bg_metadata['delta_R'] < mass_bins[i]) & (bg_metadata['delta_R'] >= mass_bins[i-1])
      sig_bin = (sig_metadata['delta_R'] < mass_bins[i]) & (sig_metadata['delta_R'] >= mass_bins[i-1])
      bg_count = np.sum(bg_bin)
      sig_count = np.sum(sig_bin)
      if sig_count == 0:
        sig_weights[sig_bin] = 0.0
      else:
        sig_weights[sig_bin] = float(bg_count) / float(sig_count)
  bg_weights = np.ones(bg_pixels.shape[0])
  weights = np.concatenate((bg_weights, sig_weights), axis=0)
  # Reshape the pixels into 2D images
  bg_pixels = bg_pixels.reshape(bg_pixels.shape[0], bg_pixels.shape[1], bg_pixels.shape[2], 1)
  sig_pixels = sig_pixels.reshape(sig_pixels.shape[0], sig_pixels.shape[1], sig_pixels.shape[2], 1)
  bg_y = np.zeros(bg_pixels.shape[0])
  sig_y = np.ones(sig_pixels.shape[0])
  X = np.concatenate((bg_pixels, sig_pixels), axis=0)
  y = np.concatenate((bg_y, sig_y), axis=0)
  return train_test_split(X, y, weights, train_size=train_size)

def main():
  X_train, X_test, y_train, y_test, weights_train, weights_test = get_train_test()

if __name__ == '__main__':
  main()
