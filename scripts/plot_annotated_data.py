import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seqs', type=str, help="File containing sequences")
parser.add_argument('--ann', type=str, help="File containing annotations")
parser.add_argument('--out', type=str, help="Filepath for final output")
parser.add_argument('--num_seqs', type=int, default=None, help="Number of sequences to use for final output")
args = parser.parse_args()

with open(os.path.expanduser(args.seqs), "r") as f:
  s = [line.strip() for line in f.readlines() if line.strip()]
a = np.loadtxt(os.path.expanduser(args.ann))

assert len(s)==a.shape[0], "Number of sequences and annotations must match."

def inline_annotate(sequence, annotation):
  out_seq = ""
  for char, val in zip(sequence, annotation):
    out_seq += char.upper() if val >=0.5 else char.lower()
  return out_seq

ann_seqs = [inline_annotate(seq, ann) for seq, ann in zip(s,a)]

with open(os.path.expanduser(args.out), "w") as f:
  f.write("\n".join(ann_seqs))

