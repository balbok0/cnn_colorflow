import os

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import constants
import utils

def plot_roc(title, fname, X_test, y_test, model, show=False):
  y_score = model.predict(X_test, verbose=2)
  fpr, tpr, _ = roc_curve(y_test, y_score)
  AUC = auc(fpr, tpr)
  plt.plot(fpr, tpr, lw=2, drawstyle='steps-post', color='blue')
  plt.plot([0,1], [0,1], 'r--')
  plt.text(0.6, 0.2, "AUC = {:.3}".format(AUC), fontsize=17, weight=550)
  plt.xlim([0, 1])
  plt.ylim([0, 1.05])
  plt.xlabel('false positive rate', fontsize=15)
  plt.ylabel('true positive rate', fontsize=15)
  plt.title(title, fontsize=19)
  plt.savefig(fname)
  if show:
    plt.show()

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Load a given model and perform visualization.')
  parser.add_argument('--run_dir', default=None, help='The run directory that should be used (see train.py). If unspecified, the most recent run directory is used.')
  args = parser.parse_args()
  if not args.run_dir:
    args.run_dir = utils.most_recent_dir()
    print('[plot_roc] run_dir not specified, using {}'.format(args.run_dir))
  model, X_test, y_test = utils.get_model_test(args.run_dir)
  plot_roc('ROC curve', os.path.join(args.run_dir, 'roc_plot.png'), X_test, y_test, model, show=True)

if __name__ == '__main__':
  main()
