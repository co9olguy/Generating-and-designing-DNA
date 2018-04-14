import os
import argparse
import tensorflow as tf
import numpy as np
import lib

checkpoint = None

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--generic', action='store_true', help="Generate generic data on the fly (ignores data_loc and data_start args)")
parser.add_argument('--data_loc', type=str, default='/home/nkilloran/datasets/leo_binding_data/Max', help='Data location')
parser.add_argument('--data_start', type=int, default=0, help='Line number to start when parsing data (useful for ignoring header)')
parser.add_argument('--log_dir', type=str, help='Base log folder')
parser.add_argument('--log_name', type=str, default="gan_test", help='Name to use when logging this model')
parser.add_argument('--checkpoint', type=str, default=checkpoint, help='Filename of checkpoint to load')
parser.add_argument('--model_type', type=str, default="resnet", help='Which type of model architecture to use')
parser.add_argument('--train_iters', type=int, default=100000, help='Number of iterations to train GAN for')
parser.add_argument('--disc_iters', type=int, default=5, help='Number of iterations to train discriminator for at each training step')
parser.add_argument('--checkpoint_iters', type=int, default=250, help='Number of iterations before saving checkpoint')
parser.add_argument('--latent_dim', type=int, default=100, help='Size of latent space')
parser.add_argument('--gen_dim', type=int, default=100, help='Generator dimension parameter')
parser.add_argument('--disc_dim', type=int, default=100, help='Discriminator dimension parameter')
parser.add_argument('--gen_layers', type=int, default=5, help='How many layers for generator')
parser.add_argument('--disc_layers', type=int, default=5, help='How many layers for discriminator')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=36, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
parser.add_argument('--annotate', action='store_true', help="Include annotation as part of training/generation process?")
parser.add_argument('--validate', action='store_false', help="Whether to use validation set")
parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for the optimizer")
parser.add_argument('--lmbda', type=float, default=10., help='Lipschitz penalty hyperparameter')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

# set RNG
seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

# fix vocabulary of model
charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)

I = np.eye(vocab_size)

# organize model logs/checkpoints
logdir, checkpoint_baseline = lib.log(args, samples_dir=True)

# build GAN
latent_vars = tf.Variable(tf.random_normal(shape=[args.batch_size, args.latent_dim], seed=seed), name='latent_vars')
if args.annotate:
  data_enc_dim = vocab_size + 1
else:
  data_enc_dim = vocab_size
data_size = args.max_seq_len * data_enc_dim

with tf.variable_scope("Generator", reuse=None) as scope:
  if args.model_type=="mlp":
    gen_data = lib.models.mlp_generator(latent_vars, dim=args.gen_dim, input_size=args.latent_dim, output_size=data_size, num_layers=args.gen_layers)
  elif args.model_type=="resnet":
    gen_data = lib.models.resnet_generator(latent_vars, args.gen_dim, args.max_seq_len, data_enc_dim, args.annotate)
  gen_vars = lib.get_vars(scope)

if args.model_type=="mlp":
  real_data = tf.placeholder(tf.float32, shape=[args.batch_size, args.max_seq_len])
  eps = tf.random_uniform([args.batch_size, 1])
elif args.model_type=="resnet":
  real_data = tf.placeholder(tf.float32, shape=[args.batch_size, args.max_seq_len, data_enc_dim])
  eps = tf.random_uniform([args.batch_size, 1, 1])
interp = eps * real_data + (1 - eps) * gen_data

with tf.variable_scope("Discriminator", reuse=None) as scope:
  if args.model_type=="mlp":
    gen_score = lib.models.discriminator(gen_data, dim=args.disc_dim, input_size=data_size, num_layers=args.disc_layers)
  elif args.model_type=="resnet":
    gen_score = lib.models.resnet_discriminator(gen_data, args.disc_dim, args.max_seq_len, data_enc_dim, res_layers=args.disc_layers)
  disc_vars = lib.get_vars(scope)
with tf.variable_scope("Discriminator", reuse=True) as scope:
  if args.model_type=="mlp":
    real_score = lib.models.mlp_discriminator(real_data, dim=args.disc_dim, input_size=data_size, num_layers=args.disc_layers)
    interp_score = lib.models.mlp_discriminator(interp, dim=args.disc_dim, input_size=data_size, num_layers=args.disc_layers)
  elif args.model_type=="resnet":
    real_score = lib.models.resnet_discriminator(real_data, args.disc_dim, args.max_seq_len, data_enc_dim, res_layers=args.disc_layers)
    interp_score = lib.models.resnet_discriminator(interp, args.disc_dim, args.max_seq_len, data_enc_dim, res_layers=args.disc_layers)
    
# cost function
gen_cost = - tf.reduce_mean(gen_score)
disc_diff = tf.reduce_mean(gen_score) - tf.reduce_mean(real_score)
# gradient penalty
grads = tf.gradients(interp_score, interp)[0]
grad_norms = tf.norm(grads, axis=[1,2]) # might need extra term for numerical stability of SGD
grad_penalty = args.lmbda * tf.reduce_mean((grad_norms - 1.) ** 2)
disc_cost = disc_diff + grad_penalty

gen_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9, name='gen_optimizer') #Note: Adam optimizer requires fixed shape
disc_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9, name='disc_optimizer')
gen_train_op = gen_optimizer.minimize(gen_cost, var_list=gen_vars)
disc_train_op = disc_optimizer.minimize(disc_cost, var_list=disc_vars)
tf.add_to_collection('latents', latent_vars)
tf.add_to_collection('outputs', gen_data)

session = tf.Session()
session.run(tf.global_variables_initializer())

# load dataset
if args.generic:
  if args.annotate: raise Exception("args `annotate` and `generic` are incompatible.")

  def feed(batch_size=args.batch_size, seq_len=args.max_seq_len, data_len=None):
    if data_len: feed_ctr = 0
    while True:
      samples = np.random.choice(vocab_size, [batch_size, seq_len])
      data = np.vstack([np.expand_dims(I[vec],0) for vec in samples])
      if args.model_type == "mlp":
        reshaped_data = np.reshape(data, [batch_size, -1])
      elif args.model_type == "resnet":
        reshaped_data = data
      if data_len:
        feed_ctr += batch_size
        if feed_ctr > data_len:
          feed_ctr = 0
          yield None
        else:
          yield reshaped_data
      else:
        yield reshaped_data

  train_seqs = feed()
  if args.validate: valid_seqs = feed(data_len=10000)

else:
  data = lib.dna.load(args.data_loc, vocab_order=args.vocab_order, max_seq_len=args.max_seq_len,
                      data_start_line=args.data_start, vocab=args.vocab, valid=args.validate,
                      annotate=args.annotate)
  if args.validate:
    split = len(data) // 2
    train_data = data[:split]
    valid_data = data[split:]
    if len(train_data) == 1: train_data = train_data[0]
    if len(valid_data) == 1: valid_data = valid_data[0]
  else:
    train_data = data
  if args.annotate:
    if args.validate: valid_data = np.concatenate(valid_data, 2)
    train_data = np.concatenate(train_data, 2)

  def feed(data, batch_size=args.batch_size, reuse=True):
    num_batches = len(data) // batch_size
    if args.model_type=="mlp":
      reshaped_data = np.reshape(data, [data.shape[0], -1])
    elif args.model_type=="resnet":
      reshaped_data = data
    while True:
      for ctr in range(num_batches):
        yield reshaped_data[ctr * batch_size : (ctr + 1) * batch_size]
      if not reuse and ctr == num_batches - 1:
        yield None
  train_seqs = feed(train_data)
  valid_seqs = feed(valid_data, reuse=False)

# load checkpoint
saver = tf.train.Saver()
if args.checkpoint:
  saver.restore(session, args.checkpoint)

# train GAN
print("Training GAN")
print("================================================")
fixed_latents = np.random.normal(size=[args.batch_size, args.latent_dim])
train_cost = []
train_counts = []
valid_cost = []
valid_counts = []
for idx in range(args.train_iters):
  true_count = idx + 1 + checkpoint_baseline
  train_counts.append(true_count)

  # train generator
  if idx > 0:
    noise = np.random.normal(size=[args.batch_size, args.latent_dim])
    _ = session.run(gen_train_op, {latent_vars: noise})

  # train discriminator "to optimality"
  for d in range(args.disc_iters):
    data = next(train_seqs)
    noise = np.random.normal(size=[args.batch_size, args.latent_dim])
    cost, _ = session.run([disc_cost, disc_train_op], {latent_vars: noise, real_data: data})
  train_cost.append(cost)

  if true_count % args.checkpoint_iters == 0:
    # validation
    cost_vals = []
    data = next(valid_seqs)
    while data is not None:
      noise = np.random.normal(size=[args.batch_size, args.latent_dim])
      score_diff = session.run(disc_diff, {latent_vars: noise, real_data: data})
      cost_vals.append(score_diff)
      data = next(valid_seqs)
    valid_cost.append(np.mean(cost_vals))
    valid_counts.append(true_count)
    name = "valid_disc_cost"
    if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
    lib.plot(valid_counts, valid_cost, logdir, name, xlabel="Iteration", ylabel="Discriminator cost")

    # log results
    print("Iteration {}: train_disc_cost={:.5f}, valid_disc_cost={:.5f}".format(true_count, cost, score_diff))
    samples = session.run(gen_data, {latent_vars: fixed_latents}).reshape([-1, args.max_seq_len, data_enc_dim])
    lib.save_samples(logdir, samples, true_count, rev_charmap, annotated=args.annotate)
    name = "train_disc_cost"
    if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
    lib.plot(train_counts, train_cost, logdir,name, xlabel="Iteration", ylabel="Discriminator cost")
    
  # save checkpoint
  if args.checkpoint_iters and true_count % args.checkpoint_iters == 0:
    ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(true_count))
    os.makedirs(ckpt_dir, exist_ok=True)
    saver.save(session, os.path.join(ckpt_dir, "trained_gan.ckpt"))

print("Done")