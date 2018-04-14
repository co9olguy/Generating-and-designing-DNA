import os
import argparse
import numpy as np
import pandas as pd

seq_file = "/home/nkilloran/datasets/leo_binding_data/Mad/sim_ground_truth/valid_data.txt"
val_file = "/home/nkilloran/datasets/leo_binding_data/Mad/sim_ground_truth/valid_vals.txt"
out_loc = "/home/nkilloran/datasets/leo_binding_data/Mad/sim_ground_truth/chopped_dataset"

parser = argparse.ArgumentParser()
parser.add_argument('--sequences_file', type=str, default=seq_file, help='Full filepath of sequences dataset')
parser.add_argument('--values_file', type=str, default=val_file, help='Full filepath of values dataset')
parser.add_argument('--out_loc', type=str, default=out_loc, help='Where to save the chopped data')
parser.add_argument('--name', type=str, default="valid", help="Name to use for new files")
parser.add_argument('--lower', type=float, default=.3, help='Lower limit for removed data')
parser.add_argument('--upper', type=float, default=1, help='Upper limit for removed data')
parser.add_argument('--data_start', type=int, default=0, help='Number of rows to skip when loading data')
args = parser.parse_args()

seqs = pd.read_csv(args.sequences_file, squeeze=True, skiprows=args.data_start)
vals = pd.read_csv(args.values_file, squeeze=True, skiprows=args.data_start)

rows = (vals <= args.lower) | (vals >= args.upper)
num_rows = len(vals) - int(rows.sum())

if num_rows == 0:
  print("Did not find any datapoints in the specified range, aborting.")
else:
  chopped_seqs = seqs[rows]
  chopped_vals = vals[rows]

  os.makedirs(args.out_loc, exist_ok=True)
  with open(os.path.join(args.out_loc, "{}_data.txt".format(args.name)), "w") as f:
    print("\n".join(chopped_seqs), file=f)

  np.savetxt(os.path.join(args.out_loc, "{}_vals.txt".format(args.name)), chopped_vals)

print("Done")