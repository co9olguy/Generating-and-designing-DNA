import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_loc', type=str, help='Path for raw data')
parser.add_argument('--valid_frac', type=float, default=0.1, help='Percentage of raw data to use for validation set')
parser.add_argument('--test_frac', type=float, default=0.1, help='Percentage of raw data to use for test set')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

files = os.listdir(args.data_loc)
candidates = [file for file in files if "exon" in file]
if len(candidates) > 2:
  raise Exception("Too many candidate files in provided data_loc. There should be one sequence and one annotation file")
else:
  seqs_file = [file for file in candidates if "exon_seqs_" in file]
  ann_file = [file for file in candidates if "exon_ann_" in file]
if seqs_file:
  seqs_file = seqs_file[0]
else:
  raise Exception("No sequence file recognized")
if ann_file:
  ann_file = ann_file[0]
else:
  raise Exception("No annotation file recognized")


with open(os.path.join(args.data_loc, seqs_file)) as f:
  seqs = [line.strip() for line in f.readlines()]
ann = np.loadtxt(os.path.join(args.data_loc, ann_file))
assert len(seqs)==len(ann), "Sequence and annotation files are incompatible shapes"

np.random.seed(args.seed)
order = np.random.choice(len(seqs), size=len(seqs), replace=False)

seqs = np.array(seqs)[order]
ann = ann[order]

valid_size = int(args.valid_frac * len(seqs))
test_size = int(args.test_frac * len(seqs))

valid_seqs = seqs[:valid_size]
valid_ann = ann[:valid_size]
test_seqs = seqs[valid_size : valid_size + test_size]
test_ann = ann[valid_size : valid_size + test_size]
train_seqs = seqs[valid_size + test_size :]
train_ann = ann[valid_size + test_size :]

with open(os.path.join(args.data_loc, "train_data.txt"), "w")  as f:
  f.write("\n".join(train_seqs))
with open(os.path.join(args.data_loc, "valid_data.txt"), "w")  as f:
  f.write("\n".join(valid_seqs))
with open(os.path.join(args.data_loc, "test_data.txt"), "w")  as f:
  f.write("\n".join(test_seqs))
np.savetxt(os.path.join(args.data_loc, "train_ann.txt"), train_ann, fmt='%d')
np.savetxt(os.path.join(args.data_loc, "valid_ann.txt"), valid_ann, fmt='%d')
np.savetxt(os.path.join(args.data_loc, "test_ann.txt"), test_ann, fmt='%d')