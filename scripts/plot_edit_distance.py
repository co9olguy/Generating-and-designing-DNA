import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import argparse

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_loc', type=str, default='/home/nkilloran/Desktop/Max_pipeline/2017.05.30-17h52m42s_pcnathan/gen_edit_distance.txt', help='Data location')
parser.add_argument('--out_loc', type=str, default='/home/nkilloran/Desktop/Max_pipeline/2017.05.30-17h52m42s_pcnathan/', help='Location to save plot')
parser.add_argument('--name', type=str, default=None, help='Name to use in plot')
args = parser.parse_args()

with open(args.data_loc) as f:
   dists = [int(line.split()[1]) for line in f.readlines()]

rng = np.linspace(0,20,21)
plt.hist(dists, rng, normed=True)
plt.xlabel("Edit distance")
plt.xticks(rng)
plt.ylabel("Counts (normalized)")
title = "Edit distance to training set"
if args.name: title += " for {} data".format(args.name)
plt.title(title)
save_name = "edit_dist"
if args.name: save_name += "_{}".format(args.name)
plt.savefig(os.path.join(args.out_loc, save_name + ".png"))

print("Done")