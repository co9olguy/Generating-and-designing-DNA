import os
import argparse
import tensorflow as tf
import numpy as np
import lib

checkpoint = None

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_loc', type=str, default='/home/nkilloran/datasets/leo_binding_data/Max/sim_ground_truth', help='Data location')
parser.add_argument('--data_start', type=int, default=0, help='Line number to start when parsing data (useful for ignoring header)')
parser.add_argument('--log_dir', type=str, default=os.path.abspath("../logs"), help='Base log folder')
parser.add_argument('--log_name', type=str, default="predictor", help='Name to use when logging this model')
parser.add_argument('--checkpoint', type=str, default=checkpoint, help='Filename of checkpoint to load')
parser.add_argument('--num_epochs', type=int, default=75, help='Number of epochs to train for')
parser.add_argument('--checkpoint_iters', type=int, default=100, help='Number of epochs before saving checkpoint')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=36, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
parser.add_argument('--filter_size', type=int, default=15, help="Size of convolutional filters")
parser.add_argument('--num_filters', type=int, default=100, help="Number of convolutional filters")
parser.add_argument('--num_layers', type=int, default=2, help="Number of fully connected hidden layers")
parser.add_argument('--hidden_size', type=int, default=200, help="Dimension of fully connected layers")
parser.add_argument('--learning_rate', type=float, default=0.00005, help="Learning rate for training")
parser.add_argument('--final_activation', type=str, default="sigmoid", help="Activation for final layer before output")
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

# set RNG
seed = args.seed
tf.set_random_seed(seed)

# fix vocabulary of model
charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)

# organize model logs/checkpoints
logdir, checkpoint_baseline = lib.log(args)

# build model
def gradient_relu(input):
  return tf.maximum(0.1 * input, input)
inputs = tf.Variable(tf.constant(0., shape=[args.batch_size, args.max_seq_len, vocab_size]), name="Input_layer") # (batch, seq_len, vocab)
pad_dim = args.filter_size // 2

#NOTE: there appears to be a size-mismatch bug for even filter sizes. Still needs fixing

pad = 0.25 * np.ones(shape=[args.batch_size, pad_dim, vocab_size])
for c in charmap:
  if c not in "ACGT":
    pad[:,:,charmap[c]] = 0.0
pad_tensor = tf.cast(pad, tf.float32)
padded_inputs = tf.concat([pad_tensor, inputs, pad_tensor], 1)
filters = tf.Variable(tf.truncated_normal(stddev=0.01, shape=[args.filter_size, len(charmap), args.num_filters])) #(width, in_channels, out_channels)
conv = tf.nn.conv1d(padded_inputs, filters, stride=1, padding="VALID")
bias = tf.Variable(tf.constant(0.0, shape=[args.num_filters]))
relu = gradient_relu(conv + bias)
features = tf.expand_dims(relu, 1)
max_pool = tf.nn.max_pool(features, ksize=[1,1,args.max_seq_len,1], strides=[1,1,args.max_seq_len,1], padding="SAME")
avg_pool = tf.nn.avg_pool(features, ksize=[1,1,args.max_seq_len,1], strides=[1,1,args.max_seq_len,1], padding="SAME")
combined_features = tf.concat([max_pool, avg_pool], axis=3)
layer_in = tf.squeeze(combined_features)
in_size = 2 * args.num_filters
out_size = args.hidden_size
for l in range(args.num_layers):
  W = tf.Variable(tf.truncated_normal(shape=[in_size, out_size], stddev=tf.sqrt(2. / tf.cast(in_size, tf.float32))))
  b = tf.Variable(tf.constant(0.0, shape=[out_size]))
  linear = tf.matmul(layer_in, W) + b
  layer_out = gradient_relu(linear)
  layer_in = layer_out
  in_size = out_size
  
W_final = tf.Variable(tf.truncated_normal(shape=[args.hidden_size, 1], stddev=tf.sqrt(2. / tf.cast(args.hidden_size, tf.float32))))
b_final = tf.Variable(tf.constant(0.0, shape=[1]))
final_linear = tf.matmul(layer_in, W_final) + b_final
final_out = gradient_relu(final_linear)

if args.final_activation=="sigmoid":
  final_activation = tf.nn.sigmoid(final_out)
elif args.final_activation=="relu":
  final_activation = tf.nn.relu(final_out)
elif args.final_activation=="grad_relu":
  final_activation = gradient_relu(final_out)
else:
  raise Exception("Unknown final activation type")
predictions = tf.reshape(final_activation, [-1], name="predictions")
true_vals = tf.placeholder(dtype=tf.float32, shape=[args.batch_size])

# cost function
cost = tf.reduce_mean(0.5 * (predictions - true_vals) ** 2)
model_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v != inputs]
optimizer = tf.train.AdamOptimizer(name='optimizer', learning_rate=args.learning_rate)
train_op = optimizer.minimize(cost, var_list=model_vars)
tf.add_to_collection('inputs', inputs)
tf.add_to_collection('predictions', predictions)

session = tf.Session()
session.run(tf.global_variables_initializer())

# load dataset
d = lib.dna.load(args.data_loc, max_seq_len=args.max_seq_len, vocab=args.vocab,
                 data_start_line=args.data_start, valid=True, test=True, scores=True)
train_data, train_vals = d[:2]
valid_data, valid_vals = d[2:4]
test_data, test_vals = d[4:]

train_feed = lib.feed([train_data, train_vals], args.batch_size, reuse=False)
valid_feed = lib.feed([valid_data, valid_vals], args.batch_size, reuse=False)
test_feed = lib.feed([test_data, test_vals], args.batch_size, reuse=False)

# load checkpoint
saver = tf.train.Saver()
if args.checkpoint:
  saver.restore(session, args.checkpoint)

# train predictor
print("Training Predictor")
print("================================================")
train_cost = []
train_counts = 0
valid_cost = []
valid_counts = []
for idx in range(args.num_epochs):
  epoch_count = idx + 1 + checkpoint_baseline

  train_data = next(train_feed)
  while train_data is not None:
    train_seqs, train_vals = train_data
    cost_val, _ = session.run([cost, train_op], {inputs:train_seqs, true_vals:train_vals})
    train_cost.append(cost_val)
    train_counts += 1
    train_data = next(train_feed)

  # validation
  cost_vals = []
  valid_data = next(valid_feed)
  while valid_data is not None:
    valid_seqs, valid_vals = valid_data
    cost_val = session.run(cost, {inputs:valid_seqs, true_vals:valid_vals})
    cost_vals.append(cost_val)
    valid_data = next(valid_feed)
  valid_cost.append(np.mean(cost_vals))
  valid_counts.append(epoch_count)
  name = "valid_cost"
  if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
  lib.plot(valid_counts, valid_cost, logdir, name, xlabel="Epoch", ylabel="Cost")

  # log results
  print("Epoch {}: train cost={:.8f}, valid_cost={:.8f}".format(epoch_count, cost_val, valid_cost[-1]))
  name = "train_cost"
  if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
  lib.plot(range(train_counts), train_cost, logdir,name, xlabel="Iteration", ylabel="Cost")
    
  # save checkpoint
  checkpoint_epoch = args.checkpoint_iters and (epoch_count % args.checkpoint_iters == 0) or (idx == args.num_epochs - 1)
  if checkpoint_epoch:
    ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(epoch_count))
    os.makedirs(ckpt_dir, exist_ok=True)
    saver.save(session, os.path.join(ckpt_dir, "trained_predictor.ckpt"))
    
# test
cost_vals = []
test_data = next(test_feed)
while test_data is not None:
  test_seqs, test_vals = test_data
  cost_val = session.run(cost, {inputs:test_seqs, true_vals:test_vals})
  cost_vals.append(cost_val)
  test_data = next(test_feed)
test_cost = np.mean(cost_vals)
print("Training finished.")
print("Final test cost={:.8f}".format(test_cost))

print("Done")