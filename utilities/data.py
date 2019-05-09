import os
import numpy as np 
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import gc

import constants

def get_pixels_metadata(bg=False, n=-1, delta_R_min=float('-inf'), delta_R_max=float('inf'), recalculate=False, same_file=False):
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

  print("[data] Getting pixel data ...")
  if bg:
    h5file = constants.BG_H5
  else:
    h5file = constants.SIG_H5
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

  sig_cutoff = int(np.sum(data['meta_variables/signal'][()]))
  size = data['meta_variables/pull1'][()].shape[0]

  if n == -1:
    metadata = np.zeros((size, 4))
    metadata[:, 0] = np.array(data['meta_variables/pull1'][()])
    metadata[:, 1] = np.array(data['meta_variables/pull2'][()])
    metadata[:, 2] = np.array(data['meta_variables/jet_mass'][()])
    metadata[:, 3] = np.array(data['meta_variables/jet_delta_R'][()])
    pixels = data['images'][()]
  else:
    metadata = np.zeros((n, 4))
    metadata[:, 0] = np.array(data['meta_variables/pull1'][:n])
    metadata[:, 1] = np.array(data['meta_variables/pull2'][:n])
    metadata[:, 2] = np.array(data['meta_variables/jet_mass'][:n])
    metadata[:, 3] = np.array(data['meta_variables/jet_delta_R'][:n])
    pixels = data['images'][:n]

  metadata = pd.DataFrame(metadata, columns=['pull_1', 'pull_2', 'mass', 'delta_R'])
  # Restrict delta R
  pixels = pixels[np.where((metadata['delta_R'] <= delta_R_max) & (metadata['delta_R'] >= delta_R_min))]
  print("[data] {} pixels shape: {}".format("bg" if bg else "sig", pixels.shape))
  print("[data] {} metadata head:\n {}".format("bg" if bg else "sig", metadata.head()))

  if (same_file):
    if (bg):
      pixels = pixels[sig_cutoff:]
      metadata = metadata[sig_cutoff:]
    else:
      pixels = pixels[:sig_cutoff]
      metadata = metadata[:sig_cutoff]

  return pixels[:n], metadata[:n]

def preprocess(x_train, x_test, no_train=False):
    def safeNorm(arr, ord): #implement own norm, this one sucks
        def shape(arr):
            return arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2] * arr.shape[3])
        def deshape(arr):
            deshape_dim = int(arr.shape[0]**0.5)
            return arr.reshape(deshape_dim, deshape_dim, 1)
        def myNorm(arr, ord):
            size = arr.shape[0]
            return np.power(np.sum(np.power(np.abs(arr), ord), axis=0), 1.0/float(size))

        arr = shape(arr)
        arr = myNorm(arr, ord)
        arr[arr==0] = 1 #only occurs in pixels that are always 0 anyways
        return deshape(arr)

    def safeLog(arr):
        eps = -6
        mask = (arr==0)
        arr = np.log(arr)
        arr[mask] = eps
        arr[arr<eps] = eps
        return 1+arr/6 #put back into reasonable range

    if not no_train:
      x_train = safeLog(x_train)
      x_test = safeLog(x_test)
      norm = safeNorm(x_train, 1)
      np.divide(x_train, norm, out=x_train)
      np.divide(x_test, norm, out=x_test)
    else:
      x_test = safeLog(x_test)
      np.divide(x_test, safeNorm(x_test, 1), out=x_test)

    return x_train, x_test

def get_train_test(n=-1, delta_R_min=float("-inf"), delta_R_max=float("inf"),  weighting_mass=False, same_file=False, train_size=0.8):
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
  bg_pixels, bg_metadata = get_pixels_metadata(bg=True, n=n, delta_R_min=delta_R_min, delta_R_max=delta_R_max, same_file=same_file)
  sig_pixels, sig_metadata = get_pixels_metadata(bg=False, n=n, delta_R_min=delta_R_min, delta_R_max=delta_R_max, same_file=same_file)
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
  
  bg_X_train, bg_X_test, sig_X_train, sig_X_test, \
    bg_y_train, bg_y_test, sig_y_train, sig_y_test, \
      bg_weights_train, bg_weights_test, sig_weights_train, sig_weights_test = \
        train_test_split(bg_pixels, sig_pixels, bg_y, sig_y, bg_weights, sig_weights, train_size=train_size, shuffle=False)

  X_train = np.concatenate((bg_X_train, sig_X_train), axis=0)
  X_test = np.concatenate((bg_X_test, sig_X_test), axis=0)
  y_train = np.concatenate((bg_y_train, sig_y_train), axis=0)
  y_test = np.concatenate((bg_y_test, sig_y_test), axis=0)
  weights_train = np.concatenate((bg_weights_train, sig_weights_train), axis=0)
  weights_test = np.concatenate((bg_weights_test, sig_weights_test), axis=0)

  X_train, y_train, weights_train = \
    shuffle(X_train, y_train, weights_train, random_state = np.random.RandomState(seed=100))
  X_test, y_test, weights_test = \
    shuffle(X_test, y_test, weights_test, random_state = np.random.RandomState(seed=100))

  del sig_y
  del bg_y
  del bg_pixels
  del sig_pixels
  gc.collect() # clean up memory

  X_train, X_test = preprocess(X_train, X_test, no_train=(train_size==0))

  return [X_train, X_test, y_train, y_test, weights_train, weights_test, sig_metadata, bg_metadata]

def main():
  X_train, X_test, y_train, y_test, weights_train, weights_test , _, _ = get_train_test()

if __name__ == '__main__':
  main()
