import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_loc', type=str, help='Location of raw data file')
parser.add_argument('--valid_frac', type=float, default=0.1, help='Percentage of raw data to use for validation set')
parser.add_argument('--test_frac', type=float, default=0.1, help='Percentage of raw data to use for test set')
parser.add_argument('--remove_N', action='store_false', help='Remove sequences with unknown characters')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

data_path = os.path.expanduser(os.path.split(args.data_loc)[0])

with open(os.path.expanduser(args.data_loc)) as f:
  if args.remove_N:
    seqs = [line.strip().upper() for line in f.readlines() if 'N' not in line.upper()]
  else:
    seqs = [line.strip().upper() for line in f.readlines()]

np.random.seed(args.seed)
order = np.random.choice(len(seqs), size=len(seqs), replace=False)

seqs = np.array(seqs)[order]

valid_size = int(args.valid_frac * len(seqs))
test_size = int(args.test_frac * len(seqs))

valid_seqs = seqs[:valid_size]
test_seqs = seqs[valid_size : valid_size + test_size]
train_seqs = seqs[valid_size + test_size :]

with open(os.path.join(data_path, "train_data.txt"), "w")  as f:
  f.write("\n".join(train_seqs))
with open(os.path.join(data_path, "valid_data.txt"), "w")  as f:
  f.write("\n".join(valid_seqs))
with open(os.path.join(data_path, "test_data.txt"), "w")  as f:
  f.write("\n".join(test_seqs))
