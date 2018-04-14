import argparse
import os
import itertools
import numpy as np
import lib

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, help='Base save folder')
parser.add_argument('--max_seq_len', type=int, default=36, help="Maximum sequence length of data")
parser.add_argument('--num_seqs', type=int, default=100000, help="Number of coding sequences to generate")
args = parser.parse_args()

assert args.max_seq_len % 3 == 0, "Sequence length must be a multiple of 3"

num_codons = args.max_seq_len // 3

charmap, rev_charmap = lib.dna.get_vocab("dna_nt_only", "ACGT")

start_codon = 'ATG'
stop_codons = ['TAA', 'TAG', 'TGA']
good_codons = ["".join(combo) for combo in itertools.product("ACGT", repeat=3) if "".join(combo) not in stop_codons]

start = np.tile(np.expand_dims([c for c in start_codon], 1), args.num_seqs).T
stops = np.vstack([list(stop_codons[n]) for n in np.random.choice(3, size=[args.num_seqs])])
coding = np.vstack([list(good_codons[c]) for c in np.random.choice(len(good_codons), size=[(num_codons - 2) * args.num_seqs])])
coding = np.reshape(coding, [args.num_seqs, (num_codons - 2) * 3])

seqs = np.hstack([start, coding, stops])
seqs = ["".join(row) for row in seqs]

os.makedirs(os.path.expanduser(args.save_dir), exist_ok=True)
with open(os.path.join(os.path.expanduser(args.save_dir), "toy_coding_seqs.txt"), "w") as f:
  f.write("\n".join(seqs))
