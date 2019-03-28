import os
import math

import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_ind, poisson
import matplotlib.pyplot as plt

import constants
import utils

def n_pass_hyp(X_test, y_test, model, flip=0, verbose=0):
  y_score = model.predict(X_test)
  y_score = y_score[:, 0]

  if flip:
      y_score = 1 - y_score
      y_test = 1 - y_test

  fpr, tpr, thrs = roc_curve(y_test, y_score)
  sic = np.divide(tpr - fpr, np.sqrt(tpr), out=np.zeros_like(tpr), where=np.sqrt(fpr)!=0)

  max_sic = 0
  max_i = 0
  for i in range(thrs.shape[0]):
    if sic[i] > max_sic:
      max_sic = sic[i]
      max_i = i

  TPR = tpr[max_i]
  FPR = fpr[max_i]

  NH = 1000
  NL = 1

  while (NH-NL > 0.1):
    N = 0.5*(NH+NL)

    myval = 0. #This is the expected value of P(n_back >= N) 
    for obs in range(100):
        p_obs = poisson.pmf(obs, N*TPR)
        p_thisorgreaterunderback = 1 - poisson.cdf(obs,N*FPR) + poisson.pmf(obs,N*FPR)
        myval += p_obs*p_thisorgreaterunderback
    
    if (myval < 0.05):
        NH = N
    else:
        NL = N
    
    if verbose:
      print(N)
      print(myval)
  
  return N

def plot_roc(title, fname, X_test, y_test, model, show=False, use2 = False, X_test2 = None, model2 = None):
  plt.clf()

  y_score = model.predict(X_test)
  fpr, tpr, _ = roc_curve(y_test, y_score)
  AUC = auc(fpr, tpr)
  plt.plot(fpr, tpr, lw=2, drawstyle='steps-post', color='blue')

  if use2:
    y_score2 = model2.predict_proba(X_test2)
    y_score2 = y_score2[:, 1]
    fpr2, tpr2, _ = roc_curve(y_test, y_score2)
    AUC2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, lw=2, drawstyle='steps-post', color='green')

  plt.plot([0,1], [0,1], 'r--')
  plt.text(0.4, 0.2, "AUC Net = {:.3}".format(AUC), fontsize=17, weight=550)
  if use2:
    plt.text(0.4, 0.1, "AUC Obs = {:.3}".format(AUC2), fontsize=17, weight=550)

  plt.xlim([0, 1])
  plt.ylim([0, 1.05])
  plt.xlabel('false positive rate', fontsize=15)
  plt.ylabel('true positive rate', fontsize=15)
  plt.title(title, fontsize=19)
  plt.savefig(fname)
  if show:
    plt.show()

def plot_sic(title, fname, X_test, y_test, model, show=False, use2 = False, X_test2 = None, model2 = None):
  plt.clf()

  y_score = model.predict(X_test)
  fpr, tpr, _ = roc_curve(y_test, y_score)
  sic = np.divide(tpr, np.sqrt(fpr), out=np.zeros_like(tpr), where=np.sqrt(fpr)!=0)
  plt.plot(tpr, sic, lw=2, drawstyle='steps-post', color='red')

  
  if use2:
    y_score2 = model2.predict_proba(X_test2)
    y_score2 = y_score2[:, 1]
    fpr2, tpr2, _ = roc_curve(y_test, y_score2)
    sic2 = np.divide(tpr2, np.sqrt(fpr2), out=np.zeros_like(tpr2), where=np.sqrt(fpr2)!=0)
    plt.plot(tpr2, sic2, lw=2, drawstyle='steps-post', color='green')


  plt.text(0.4, 0.2, "Max SIC Net = {:.3}".format(np.max(sic)), fontsize=17, weight=550)
  if use2:
    plt.text(0.4, 0.1, "Max SIC Obs = {:.3}".format(np.max(sic2)), fontsize=17, weight=550)
  plt.xlabel('true positive rate', fontsize=15)
  plt.ylabel('tpr/sqrt(fpr)', fontsize=15)
  plt.title(title, fontsize=19)
  plt.savefig(fname)
  if show:
    plt.show()

def fixed_efficiency(X_test, y_test, model):
  y_score = model.predict(X_test)
  fpr, tpr, _ = roc_curve(y_test, y_score)
  return fpr[(np.abs(tpr - 0.5)).argmin()]

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Load a given model and calculate performance metrics (roc, sic, etc.).')
  parser.add_argument('--run_dir', default=None, help='The run directory that should be used (see train.py). If unspecified, the most recent run directory is used.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.most_recent_dir()
    print('[metrics] run_dir not specified, using {}'.format(args.run_dir))
  model, X_test, y_test = utils.get_model_test(args.run_dir)
  plot_roc('ROC curve', os.path.join(args.run_dir, 'roc_plot.png'), X_test, y_test, model, show=True)
  plot_sic('SIC', os.path.join(args.run_dir, 'sic_plot.png'), X_test, y_test, model, show=True)
  print('At TPR ~ 0.5, FPR = {}'.format(fixed_efficiency(X_test, y_test, model)))

if __name__ == '__main__':
  main()