import numpy as np
import tensorflow as tf
import lib
from lib.explicit import max_match
import argparse
import os, shutil

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, help='Folder to save the computational graph in')
parser.add_argument('--save_name', type=str, help='Name to use when saving this model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
args = parser.parse_args()

charmap, rev_charmap = lib.dna.get_vocab(args.vocab)
vocab_size = len(charmap)

latents = tf.Variable(tf.constant(0., shape=[args.batch_size, args.max_seq_len * vocab_size]), name='latent_vars') # (batch, seq_len * vocab)
gen_output = tf.reshape(latents, [args.batch_size, args.max_seq_len, vocab_size])
tf.add_to_collection('latents', latents)
tf.add_to_collection('outputs', gen_output)

save_loc = os.path.join(os.path.expanduser(args.save_dir), args.save_name)
os.makedirs(save_loc, exist_ok=True)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  tf.train.Saver().save(sess, os.path.join(save_loc, "generator"))
print("Done")