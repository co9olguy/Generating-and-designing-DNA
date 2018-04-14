import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr

matplotlib.rcParams.update({'font.size': 14})

parser = argparse.ArgumentParser()
parser.add_argument('--gen_scores', type=str, default=None, help='Location of scores for generated sequences.')
parser.add_argument('--gt_scores', type=str, default=None, help='Location of "ground truth" scores.')
parser.add_argument('--train_scores', type=str, default=None, help='Location of training scores.')
parser.add_argument('--train_skiprows', type=int, default=0, help='Row number where data starts in training set.')
parser.add_argument('--out_loc', type=str, help='Where to save the final plot')
parser.add_argument('--out_name', type=str, default="sample_gen_vals", help="Filename for final figure")
parser.add_argument('--format', default="png", type=str, help="Format for saving plot")
parser.add_argument('--name', type=str, help='Name of dataset to use for plot title')
parser.add_argument('--lower', default=0., type=float, help='Lower limit of scores to plot')
parser.add_argument('--upper', default=1., type=float, help='Upper limit of scores to plot')
parser.add_argument('--train_label', default="True data", type=str, help="Label to use for training data in plot")
parser.add_argument('--gt_label', default='Synthetic "ground truth" data', type=str, help="Label to use for ground-truth data in plot")
parser.add_argument('--gen_label', default='"Ground truth" scores of generated samples', type=str, help="Label to use for generated scores in plot")
parser.add_argument('--xlabel', default="Scores", type=str, help="X-axis label for plot")
parser.add_argument('--ylabel', default="Counts (normalized)", type=str, help="X-axis label for plot")
parser.add_argument('--plot_type', default="hist", type=str, help="Type of plot to make")
args = parser.parse_args()

if args.plot_type=="scatter" and args.gen_scores:
  raise Exception("Scatter plot not available for generated data")

rng = np.linspace(args.lower,args.upper,100)
plt.close()
ax = plt.gca()
if args.train_scores:
  train_scores = np.loadtxt(args.train_scores, skiprows=args.train_skiprows)
  if args.plot_type=="hist":
    plt.hist(train_scores, bins=rng, normed=True, color='C0', alpha=0.5, label=args.train_label)
if args.gt_scores:
  gt_scores = np.loadtxt(args.gt_scores, skiprows=1)
  if args.plot_type=="hist":
    plt.hist(gt_scores, bins=rng, normed=True, color='C1', alpha=0.5, label=args.gt_label)
if args.plot_type=="scatter":
  if len(train_scores) > len(gt_scores): # any remainder after dividing dataset by batch size were not scored by oracle
    print("Warning: omitting remainder of experimental scores where there is no corresponding ground truth score")
    train_scores = train_scores[:len(gt_scores)]
  plt.scatter(train_scores, gt_scores, alpha=0.7, s=0.3)
  #plt.plot([0,1],[0,1], color='C3')
  ax.margins(0.0)
  ax.set_aspect('equal')
  plt.xlim([0,1])
  plt.ylim([0,1])
  pearson = pearsonr(train_scores, gt_scores)
  spearman = spearmanr(train_scores, gt_scores)
  print("Pearson correlation between train and oracle scores: {}".format(pearson))
  print("Spearman correlation between train and oracle scores: {}".format(spearman))
if args.gen_scores:
  gen_scores = np.loadtxt(args.gen_scores, skiprows=1)
  plt.hist(gen_scores, bins=rng, normed=True, color='C2', alpha=0.5, label=args.gen_label)
  plt.legend()
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
title = ""
if args.name: title += "Scores of sequences ('{}' data)".format(args.name)
plt.title(title)
out_file = ".".join([args.out_name, args.format])
if args.out_loc:
  out_file = os.path.join(args.out_loc, out_file)
plt.savefig(out_file, pad_inches=0.0, transparent=False)
plt.close()
print("Done")