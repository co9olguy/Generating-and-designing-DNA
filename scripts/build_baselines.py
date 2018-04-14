import tensorflow as tf
import argparse
import lib
import os

parser = argparse.ArgumentParser()
parser.add_argument('--latent', action='store_true', help="Include a single latent layer with softmax.")
parser.add_argument('--save_dir', type=str, help='Base save folder')
parser.add_argument('--save_name', type=str, help='Name to use when saving this model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=36, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
args = parser.parse_args()


# fix vocabulary of model
charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)

# build graph
latent_vars = tf.Variable(tf.random_normal(shape=[args.batch_size, args.max_seq_len * vocab_size]), name='latent_vars')
reshaped_latents = tf.reshape(latent_vars, [args.batch_size, args.max_seq_len,  vocab_size])
if args.latent:
  gen_data = tf.nn.softmax(reshaped_latents)
else:
  gen_data = reshaped_latents

session = tf.Session()
session.run(tf.global_variables_initializer())

# save model
tf.add_to_collection('latents', latent_vars)
tf.add_to_collection('outputs', gen_data)
saver = tf.train.Saver()
save_loc = os.path.join(os.path.expanduser(args.save_dir), args.save_name)
os.makedirs(save_loc, exist_ok=True)
if args.latent:
  name = "latent_generator"
else:
  name = "no_latent_generator"
saver.save(session, os.path.join(save_loc, name))

print("Done")