import os
import socket
import datetime
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .dna import get_vocab

def get_vars(scope):
  """Function to find tensorflow variables within a scope"""
  try:
    if type(scope) == str:
      s = scope
    else:
      s = scope.name
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=s)
  except:
    raise TypeError("Unrecognized scope type")
  
def log(args, samples_dir=False):
  """Create logging directory structure according to args."""
  if hasattr(args, "checkpoint") and args.checkpoint:
    return _log_from_checkpoint(args)
  else:
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join(args.log_dir, args.log_name, stamp)
    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    args.log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    _log_args(full_logdir, args)
  return full_logdir, 0

def _log_from_checkpoint(args):
  """Infer logging directory from checkpoint file."""
  checkpoint_folder = os.path.dirname(args.checkpoint)
  int_dir, checkpoint_name = os.path.split(checkpoint_folder)
  logdir = os.path.dirname(int_dir)
  checkpoint_num = int(checkpoint_name.split('_')[1])
  _log_args(logdir, args, modified_iter=checkpoint_num)
  return logdir, checkpoint_num

def _log_args(logdir, args, modified_iter=0):
  """Write log of current arguments to text."""
  keys = sorted(arg for arg in dir(args) if not arg.startswith("_"))
  args_dict = {key: getattr(args, key) for key in keys}
  with open(os.path.join(logdir, "config.txt"), "a") as f:
    f.write("Values at iteration {}\n".format(modified_iter))
    for k in keys:
      s = ": ".join([k,str(args_dict[k])]) + "\n"
      f.write(s)
    vocab_order = args.vocab_order if hasattr(args, "vocab_order") else None
    charmap, _ = get_vocab(args.vocab, vocab_order)
    charmap_strs = []
    for k,v in charmap.items():
      charmap_strs.append("'{}':{}".format(k,v))
    f.write("charmap: " + ", ".join(charmap_strs))
    f.write("\n")
      
def save_samples(logdir, samples, iteration, rev_charmap, annotated=False):
  """Convert samples to strings and save to log directory."""
  if annotated:
    char_probs = samples[:,:,:-1]
    ann = samples[:,:,-1]
  else:
    char_probs = samples
  argmax = np.argmax(char_probs, 2)
  with open(os.path.join(logdir, "samples", "samples_{}".format(iteration)), "w") as f:
    for line in argmax:
      s = "".join(rev_charmap[d] for d in line) + "\n"
      f.write(s)
  if annotated:
    np.savetxt(os.path.join(logdir, "samples", "samples_ann_{}".format(iteration)), ann)
      
def plot(y, x, logdir, name, xlabel=None, ylabel=None, title=None):
  """Make plot of training curves"""
  plt.close()
  plt.plot(y,x)
  if xlabel:
    plt.xlabel(xlabel)
  if ylabel:
    plt.ylabel(ylabel)
  if title:
    plt.title = title
  plt.savefig(os.path.join(logdir, "{}".format(name) + ".png"))
  
def feed(data, batch_size, reuse=True):
  """Feed data in batches"""
  if type(data)==list or type(data)==tuple and len(data)==2:
    data_seqs, data_vals = data
    yield_vals = True
  else:
    data_seqs = data
    yield_vals = False
  num_batches = len(data_seqs) // batch_size
  if num_batches == 0:
    raise Exception("Dataset not large enough to accomodate batch size")
  while True:
    for ctr in range(num_batches):
      out = data_seqs[ctr * batch_size : (ctr + 1) * batch_size]
      if yield_vals:
        out = (out, data_vals[ctr * batch_size : (ctr + 1) * batch_size])
      yield out
    if not reuse and ctr == num_batches - 1:
      yield None