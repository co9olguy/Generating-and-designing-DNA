import os
import numpy as np

base_path = "/home/nkilloran/dna-gen/expts/binding_experiments/"

for p in os.listdir(base_path):
  if p[0]!=".":
    path = os.path.join(base_path, p, "sim_ground_truth_data")
    train = np.loadtxt(os.path.join(path, "train_vals.txt"), skiprows=1)
    valid = np.loadtxt(os.path.join(path, "valid_vals.txt"), skiprows=1)
    test = np.loadtxt(os.path.join(path, "test_vals.txt"), skiprows=1)

    total_vals = sorted(np.hstack([train, valid, test]))
    cutoff = total_vals[int(len(total_vals) * 0.4)]

    #print("{}: {} training datapoints, cutoff={}".format(p, train.shape[0], cutoff))
    print("'{}': {},".format(p, cutoff))
