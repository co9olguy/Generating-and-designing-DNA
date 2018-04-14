"""Helpers for working with DNA/RNA data"""

import os
import numpy as np

# fix vocabulary
dna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3,
             "*":4} # catch-all auxiliary token
rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}
dna_nt_only_vocab = {k:v for k,v in dna_vocab.items() if k in "ACGT"}
rna_nt_only_vocab = {k:v for k,v in dna_vocab.items() if k in "ACGU"}

rev_dna_vocab = {v:k for k,v in dna_nt_only_vocab.items()}
rev_rna_vocab = {v:k for k,v in rna_vocab.items()}
rev_dna_nt_only_vocab = {v:k for k,v in dna_vocab.items()}
rev_rna_nt_only_vocab = {v:k for k,v in rna_nt_only_vocab.items()}

def get_vocab(vocab_name, vocab_order=None):
  if vocab_name=="dna":
    charmap = dna_vocab
  elif vocab_name=="rna":
    charmap = rna_vocab
  elif vocab_name=="dna_nt_only":
    charmap = dna_nt_only_vocab
  elif vocab_name=="rna_nt_only":
    charmap = rna_nt_only_vocab
  else:
    raise Exception("Unknown vocabulary name.")

  if vocab_order:
    if set(vocab_order) != set(charmap):
      raise ValueError("Provided `vocab` and `vocab_order` arguments are not compatible")
    else:
      charmap = {c: idx for idx, c in enumerate(vocab_order)}

  rev_charmap = {v: k for k, v in charmap.items()}
  return charmap, rev_charmap

def _process_line(line, max_len, charmap):
  chars = line.strip()
  I = np.eye(len(charmap))
  try:
    base = [I[charmap[c]] for c in chars]
    if len(chars) < max_len:
      extra = []
      if "*" in charmap: extra = [I[charmap["*"]]] * (max_len - len(chars))
    else:
      extra = []
    arr = np.array(base + extra)
  except:
    raise Exception("Unable to process line: {}".format(chars))
  return np.expand_dims(arr, 0)

def load(data_loc, max_seq_len=None, vocab="dna", vocab_order=None, data_start_line=0, scores=False, valid=False, test=False, filenames=None, annotate=False):
  charmap, _ = get_vocab(vocab, vocab_order)

  if filenames:
    if type(filenames)==list:
      seq_filenames = filenames
    else:
      seq_filenames = [filenames]
  else:
    seq_filenames = ["train_data.txt"]
    if valid:
      seq_filenames.append("valid_data.txt")
    if test:
      seq_filenames.append("test_data.txt")

  data = []
  for name in seq_filenames:
    with open(os.path.join(data_loc, name)) as f:
      lines = f.readlines()
    lines = lines[data_start_line:]
    if not max_seq_len:
      print("Warning: max_seq_len not provided. Inferring size from data.")
      max_seq_len = len(max(lines, key=len)) - 1
    lines = [_process_line(l, max_seq_len, charmap) for l in lines]
    data.append(np.vstack(lines))
    
    if scores:
      score_filename = name.split("_")[0] + "_vals.txt"
      scores_array = np.loadtxt(os.path.join(data_loc, score_filename), skiprows=data_start_line)
      data.append(scores_array)

    if annotate:
      ann_filename = name.split("_")[0] + "_ann.txt"
      ann_array = np.loadtxt(os.path.join(data_loc, ann_filename), skiprows=data_start_line)
      ann_array = np.expand_dims(ann_array, 2)
      data.append(ann_array)
  if type(data) == list and len(data) == 1:
    data = data[0]
  return data
    