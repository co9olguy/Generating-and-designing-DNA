import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lib

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default=os.path.abspath("../logs"), help='Base log folder')
parser.add_argument('--log_name', type=str, default="pp_test", help='Name to use when logging this script')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for previous optimization')
parser.add_argument('--generator', type=str, help="Location of generator model (filename ends with '.meta')")
parser.add_argument('--predictor', type=str, help="Location of predictor model (filename ends with '.meta')")
parser.add_argument('--target', default="max", help="Optimization target. Can be either 'max', 'min', or a target score number given as a float")
parser.add_argument('--prior_weight', default=0., type=float, help="Relative weighting for the latent prior term in the optimization")
parser.add_argument('--checkpoint_iters', type=int, default=5000, help='Number of iterations to run between checkpoints of the optimization')
parser.add_argument('--optimizer', type=str, default="adam", help="Which optimizer to use. Options are 'adam' or 'sgd'")
parser.add_argument('--step_size', type=float, default=1e-1, help="Step-size for optimization.")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
parser.add_argument('--noise', type=float, default=1e-5, help="Scale of random gaussian noise to add to gradients")
parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations to run the optimization for")
parser.add_argument('--log_interval', type=int, default=250, help="Iteration interval at which to report progress")
parser.add_argument('--save_samples', type=bool, default=True, help="Whether to save samples during optimization")
parser.add_argument('--plot_mode', type=str, default="fill", help="How to plot the scores within the optimized batch")
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

assert args.generator[-5:]==args.predictor[-5:]==".meta", "Please provide '.meta' files for restoring models"

# set RNG
seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
I = np.eye(len(charmap)) # for one-hot encodings
step_size = args.step_size
alpha = args.prior_weight

# set up logging
logdir, checkpoint_baseline = lib.log(args, samples_dir=args.save_samples)

session = tf.Session()


# restore previous optimization from checkpoint or import models for new optimization
if args.checkpoint:
  ckpt_saver = tf.train.import_meta_graph(args.checkpoint)
  ckpt_saver.restore(session, args.checkpoint[:-5])
  latents = tf.get_collection('latents')[0]
  gen_output = tf.get_collection('outputs')[0]
  pred_input = tf.get_collection('inputs')[0]
  predictions = tf.get_collection('predictions')[0]
  design_op = tf.get_collection('design_op')[0]
  global_step = tf.get_collection('global_step')[0]
  prior_weight = tf.get_collection('prior_weight')[0]
  batch_size, latent_dim = session.run(tf.shape(latents))
  update_pred_input = tf.assign(pred_input, gen_output)
else:
  gen_saver = tf.train.import_meta_graph(args.generator, import_scope="generator")
  gen_saver.restore(session, args.generator[:-5])
  pred_saver = tf.train.import_meta_graph(args.predictor, import_scope="predictor")
  pred_saver.restore(session, args.predictor[:-5])
  
  latents = tf.get_collection('latents')[0]
  gen_output = tf.get_collection('outputs')[0]
  pred_input = tf.get_collection('inputs')[0]
  predictions = tf.get_collection('predictions')[0]
  
  batch_size, latent_dim = session.run(tf.shape(latents))
  latent_vars = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'generator/latent_vars' in c.name][0]
  sequence_vars = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'predictor/Input_layer' in c.name][0]
  
  assert gen_output.get_shape()==pred_input.get_shape(), "Generator output and predictor input must match."
  
  # initialize latent space and corresponding generated sequence
  start_noise = np.random.normal(size=[batch_size, latent_dim])
  session.run(tf.assign(latent_vars, start_noise))
  update_pred_input = tf.assign(pred_input, gen_output)
  
  # calculate relevant gradients
  prior_weight = tf.Variable(alpha, trainable=False)
  session.run(prior_weight.initializer)
  tf.add_to_collection('prior_weight', prior_weight)
  log_pz = tf.reduce_sum(- latents ** 2, 1)
  target = args.target
  if type(target)==str:
    if target=="max":
      cost = tf.reduce_mean(-predictions)
    elif target=="min":
      cost = tf.reduce_mean(predictions)
  elif type(target)==int or type(target)==float:
    mean, var = tf.nn.moments(predictions, axes=[0])
    cost = 0.5 * (mean - tf.cast(target, tf.float32)) ** 2 + 0.5 * (var - 0.0) ** 2
  else:
    raise TypeError("Argument 'target' must be either 'max', 'min', or a number")
  grad_cost_seq = tf.gradients(ys=cost, xs=pred_input)[0]
  grad_cost_latent = tf.gradients(ys=gen_output, xs=latents, grad_ys=grad_cost_seq)[0] + prior_weight * tf.squeeze(tf.gradients(ys=tf.reduce_mean(log_pz), xs=latents))
  # gives dcost/dz_j] for each latent entry z_j
  
  noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=args.noise)
  global_step = tf.Variable(args.step_size, trainable=False)
  session.run(global_step.initializer)
  tf.add_to_collection('global_step', global_step)
  if args.optimizer=="adam":
    if args.step_size:
      optimizer = tf.train.AdamOptimizer(learning_rate=global_step)
    else:
      optimizer = tf.train.AdamOptimizer()
    design_op = optimizer.apply_gradients([(grad_cost_latent + noise, latent_vars)])
    adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
    session.run(adam_initializers)
  elif args.optimizer=="sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=global_step)
    design_op = optimizer.apply_gradients([(grad_cost_latent + noise, latent_vars)])
  tf.add_to_collection('design_op', design_op)

s = session.run(tf.shape(latents))
session.run(update_pred_input, {latents: np.random.normal(size=s)})

saver = tf.train.Saver()
sigfigs = int(np.floor(np.log10(args.iterations))) + 1
means = []
means_onehot = []
maxes = []
mins = []
dist = []
for ctr in range(args.iterations):
  true_ctr = ctr + checkpoint_baseline + 1
  gen_outputs, _ = session.run([gen_output, design_op], {global_step: step_size, prior_weight: alpha})
  predictor_input, preds = session.run([update_pred_input, predictions])
  mean_pred = np.mean(preds)
  means.append(mean_pred)
  maxes.append(np.max(preds))
  mins.append(np.min(preds))
  dist.append(preds)

  if true_ctr == checkpoint_baseline + 1 or true_ctr % args.log_interval == 0:
    pred_onehot = session.run(predictions, {pred_input: I[np.argmax(predictor_input, -1)]})
    seq0 = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[0], -1))
    mean_pred_onehot = np.mean(pred_onehot)
    means_onehot.append(mean_pred_onehot)
    print("Iter {}: {}: score: {:.6f}; mean score: {:.6f}; std: {:.6f}; mean score (one-hot): {:.6f}".format(true_ctr, seq0, preds[0], mean_pred, np.std(preds), mean_pred_onehot))

        
    best_idx = np.argmax(preds, 0)
    z = session.run(latents)
    rev_outputs = session.run(gen_output, {latents: -z})
    best_seq = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[best_idx], -1))
    neg_best_seq = "".join(rev_charmap[n] for n in np.argmax(rev_outputs[best_idx], -1))
    print("for_best: {}\nrev_best: {}".format(best_seq, neg_best_seq))
    
    
    
    plt.cla()
    #plt.ylim([0., 1.])
    plt.xlabel("Iteration")
    plt.ylabel("Scores of sequences in batch")
    plt.plot(np.linspace(checkpoint_baseline, true_ctr, len(means)), means, color='C2', label='Mean score of generated sequences');
    if args.plot_mode=="fill":
      plt.fill_between(np.linspace(checkpoint_baseline, true_ctr, len(means)), mins, maxes, color='C0', alpha=0.5, label='Min/max score of generated sequences')
    elif args.plot_mode=="scatter":
      dist_x = np.reshape([[c] * 64 for c in np.linspace(checkpoint_baseline, true_ctr, len(dist))], [-1])
      plt.scatter(dist_x, np.reshape(dist,[-1]), color='C0', s=0.5, alpha=0.01)
    plt.plot(np.linspace(checkpoint_baseline, true_ctr, len(means_onehot)), means_onehot, color='C1', ls='--', label='Mean score of one-hot re-encoded seqs')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    def key(label):
      if "one-hot" in label:
        return 0
      elif "Mean" in label:
        return 1
      elif "max" in label:
        return 2
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: key(t[0])))

    if args.target=="max":
      ax.legend(handles, labels, loc='lower right')
    elif args.target=="min":
      ax.legend(handles, labels, loc='upper right')
    else:
      ax.legend(handles, labels, )
    name = "scores_opt"
    if checkpoint_baseline > 0: name += "_from_{}".format(checkpoint_baseline)
    plt.savefig(os.path.join(logdir, name + ".png"))
    if args.save_samples:
      ctr_with_0s = str(true_ctr).zfill(sigfigs)
      with open(os.path.join(logdir, "samples", "samples_{}.txt".format(ctr_with_0s)), "w") as f:
        f.write("\n".join("".join(rev_charmap[n] for n in np.argmax(row, -1)) for row in gen_outputs))
      with open(os.path.join(logdir, "samples", "rev_samples_{}.txt".format(ctr_with_0s)), "w") as f:
        f.write("\n".join("".join(rev_charmap[n] for n in np.argmax(row, -1)) for row in rev_outputs))
  plt.close()
        
  # save checkpoint
  if args.checkpoint_iters and true_ctr % args.checkpoint_iters == 0:
    ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(true_ctr))
    os.makedirs(ckpt_dir, exist_ok=True)
    saver.save(session, os.path.join(ckpt_dir, "pp_opt.ckpt"))

print("Done")