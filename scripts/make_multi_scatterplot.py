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
parser.add_argument('--gen_scores1', type=str, default=None, help='Location of target 1 scores for generated sequences.')
parser.add_argument('--gen_scores2', type=str, default=None, help='Location of target 2 scores for generated sequences.')
parser.add_argument('--train_scores1', type=str, default=None, help='Location of training scores for target 1.')
parser.add_argument('--train_scores2', type=str, default=None, help='Location of training scores for target 2.')
parser.add_argument('--train_skiprows1', type=int, default=0, help='Row number where data starts in target 1 training set.')
parser.add_argument('--train_skiprows2', type=int, default=0, help='Row number where data starts in target 2 training set.')
parser.add_argument('--out_loc', type=str, help='Where to save the final plot')
parser.add_argument('--out_name', type=str, default="multi_scatter", help="Filename for final figure")
parser.add_argument('--format', default="png", type=str, help="Format for saving plot")
parser.add_argument('--lower', default=0., type=float, help='Lower limit of scores to plot')
parser.add_argument('--upper', default=1., type=float, help='Upper limit of scores to plot')
parser.add_argument('--train_label', default="Training data", type=str, help="Label to use for training data in plot")
parser.add_argument('--gen_label', default='Generated data', type=str, help="Label to use for generated scores in plot")
parser.add_argument('--xlabel', default="Target 1 scores", type=str, help="X-axis label for plot")
parser.add_argument('--ylabel', default="Target 2 scores", type=str, help="X-axis label for plot")
args = parser.parse_args()

plt.close()
ax = plt.gca()
train_scores_1 = np.loadtxt(args.train_scores1, skiprows=args.train_skiprows1)
train_scores_2 = np.loadtxt(args.train_scores2, skiprows=args.train_skiprows2)
gen_scores_1 = np.loadtxt(args.gen_scores1, skiprows=args.train_skiprows1)
gen_scores_2 = np.loadtxt(args.gen_scores2, skiprows=args.train_skiprows2)
plt.scatter(train_scores_1, train_scores_2, alpha=0.5, marker='o', label=args.train_label)
plt.scatter(gen_scores_1, gen_scores_2, alpha=0.5, marker='^', label=args.gen_label)
ax.margins(0.0)
ax.set_aspect('equal')
plt.xlim([0,1])
plt.ylim([0,1])
rng = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(rng)
plt.yticks(rng)
plt.legend()

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
title = ""
out_file = ".".join([args.out_name, args.format])
if args.out_loc:
  out_file = os.path.join(args.out_loc, out_file)
plt.savefig(out_file, pad_inches=0.0, transparent=False)
plt.close()
print("Done")