import pandas as pd
import twobitreader as tb
import numpy as np
import os

exons_filepath = "/home/nkilloran/datasets/human_genome/knownCanonical.txt"
genome_filepath = "/home/nkilloran/datasets/human_genome/hg38.2bit"
out_dir = "/home/nkilloran/datasets/human_genome/"

min_size = 50
max_size = 400
window_size = 500

df = pd.read_csv(exons_filepath, skiprows=1, sep='\t')
exon_chrs = []
exon_coords = np.empty([0,2], np.int32)
for _, row in df.iterrows():
  new_chr = row["#hg38.knownCanonical.chrom"]
  new_starts = [int(string) for string in "".join(row["hg38.knownGene.exonStarts"]).split(",") if string]
  new_ends = [int(string) for string in "".join(row["hg38.knownGene.exonEnds"]).split(",") if string]
  assert len(new_starts) == len(new_ends)
  for idx, _ in enumerate(new_starts):
    exon_chrs.append(new_chr)
    new_coords = np.array([new_starts[idx], new_ends[idx]]).reshape([1,2])
    exon_coords = np.vstack([exon_coords, new_coords])

exon_size = exon_coords[:,1]-exon_coords[:,0]

filt_chrs = np.array(exon_chrs)[(exon_size >= min_size) & (exon_size <= max_size)]
filt_exons = exon_coords[(exon_size >= min_size) & (exon_size <= max_size)]

g = tb.TwoBitFile(genome_filepath)

file_strs = []
ann = np.empty([0, window_size], np.int32)
for chr, exon in zip(filt_chrs, filt_exons):
  c = g[chr]
  exon_start, exon_end = exon
  exon_len = exon_end - exon_start
  pad_size = window_size - exon_len
  left_pad = pad_size // 2
  right_pad = pad_size - left_pad
  str_ = c[exon_start - left_pad : exon_end + right_pad].upper()

  if 'N' not in str_ and len(str_) == window_size:
    file_strs.append(str_)

    base_ann = np.ones(window_size, np.int32)
    base_ann[:left_pad] = 0
    base_ann[left_pad + exon_len:] = 0
    ann = np.vstack([ann, np.expand_dims(base_ann, 0)])

with open(os.path.join(out_dir , "exon_seqs_min{}_max{}_win{}.txt".format(min_size, max_size, window_size)), "w") as f:
  f.write("\n".join(file_strs))
np.savetxt(os.path.join(out_dir, "exon_ann_min{}_max{}_win{}.txt".format(min_size, max_size, window_size)), ann, fmt='%d')
