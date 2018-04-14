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
parser.add_argument('--max_seq_len', type=int, default=50, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
parser.add_argument('--pwm_file', type=str, default=None, help='CSV file to load PWM pattern from')
parser.add_argument('--pattern', type=str, default=None, help='Hard-pattern string to match')
parser.add_argument('--padding', type=str, default="VALID", help='Padding to use in conv1d. Options are "SAME" and "VALID"')
parser.add_argument('--stride', type=int, default=1, help='Stride size for conv1d.')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()


charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)
I = np.eye(vocab_size)

if args.pwm_file and args.pattern:
  raise Exception("Please specify only one of pwm_file or pattern")
elif not (args.pwm_file or args.pattern):
  raise Exception("Please specify either a pwm_file or a pattern")
elif args.pwm_file:
  pwm = np.loadtxt(args.pwm_file)
elif args.pattern:
  indices = [charmap[c] for c in args.pattern]
  pwm = np.vstack([I[idx] for idx in indices])

inputs = tf.Variable(tf.constant(0., shape=[args.batch_size, args.max_seq_len, vocab_size]), name="Input_layer") # (batch, seq_len, vocab)
predictions = max_match(inputs, pwm, padding=args.padding, stride=args.stride) # (batch,)
tf.add_to_collection('inputs', inputs)
tf.add_to_collection('predictions', predictions)

save_loc = os.path.join(os.path.expanduser(args.save_dir), args.save_name)
os.makedirs(save_loc, exist_ok=True)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  tf.train.Saver().save(sess, os.path.join(save_loc, "predictor"))
if args.pattern:
  with open(os.path.join(save_loc, "pattern.txt"), "w") as f:
    f.write(args.pattern)
elif args.pwm_file:
  shutil.copy(args.pwm_file, os.path.join(save_loc, "pattern.txt"))
print("Done")