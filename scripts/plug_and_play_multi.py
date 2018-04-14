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
parser.add_argument('--predictor1', type=str, help="Location of first predictor model (filename ends with '.meta')")
parser.add_argument('--predictor2', type=str, help="Location of second predictor model (filename ends with '.meta')")
parser.add_argument('--target1', default="max",
                    help="Optimization target for first predictor. Can be either 'max', 'min', or a target score number given as a float")
parser.add_argument('--target2', default="min",
                    help="Optimization target for second predictor. Can be either 'max', 'min', or a target score number given as a float")
parser.add_argument('--target1_scale', default=1.0, type=float, help="Scaling parameter for target1 cost function (used to (de-)emphasize target1 vs target2)")
parser.add_argument('--prior_weight', default=0., type=float,
                    help="Relative weighting for the latent prior term in the optimization")
parser.add_argument('--checkpoint_iters', type=int, default=5000,
                    help='Number of iterations to run between checkpoints of the optimization')
parser.add_argument('--optimizer', type=str, default="adam", help="Which optimizer to use. Options are 'adam' or 'sgd'")
parser.add_argument('--step_size', type=float, default=1e-1, help="Step-size for optimization.")
parser.add_argument('--vocab', type=str, default="dna",
                    help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None,
                    help="Specific order for the one-hot encodings of vocab characters")
parser.add_argument('--noise', type=float, default=1e-5, help="Scale of random gaussian noise to add to gradients")
parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations to run the optimization for")
parser.add_argument('--log_interval', type=int, default=250, help="Iteration interval at which to report progress")
parser.add_argument('--save_samples', type=bool, default=True, help="Whether to save samples during optimization")
parser.add_argument('--plot_mode', type=str, default="fill", help="How to plot the scores within the optimized batch")
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

assert args.generator[-5:] == args.predictor1[-5:] == args.predictor2[-5:] ==".meta", "Please provide '.meta' files for restoring models"

# set RNG
seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
I = np.eye(len(charmap))  # for one-hot encodings
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
  pred_input1 = tf.get_collection('inputs')[0]
  predictions1 = tf.get_collection('predictions')[0]
  pred_input2 = tf.get_collection('inputs')[1]
  predictions2 = tf.get_collection('predictions')[1]
  design_op = tf.get_collection('design_op')[0]
  global_step = tf.get_collection('global_step')[0]
  prior_weight = tf.get_collection('prior_weight')[0]
  batch_size, latent_dim = session.run(tf.shape(latents))
  pred_input = [pred_input1, pred_input2]
  update_pred_input = [tf.assign(pred_input1, gen_output), tf.assign(pred_input2, gen_output)]
else:
  gen_saver = tf.train.import_meta_graph(args.generator, import_scope="generator")
  gen_saver.restore(session, args.generator[:-5])
  latents = tf.get_collection('latents')[0]
  gen_output = tf.get_collection('outputs')[0]

  pred_saver1 = tf.train.import_meta_graph(args.predictor1, import_scope="predictor1")
  pred_saver1.restore(session, args.predictor1[:-5])
  pred_input1 = tf.get_collection('inputs')[0]
  predictions1 = tf.get_collection('predictions')[0]
  pred_saver2 = tf.train.import_meta_graph(args.predictor2, import_scope="predictor2")
  pred_saver2.restore(session, args.predictor2[:-5])
  pred_input2 = tf.get_collection('inputs')[1]
  predictions2 = tf.get_collection('predictions')[1]

  predictions = [predictions1, predictions2]
  pred_input = [pred_input1, pred_input2]

  batch_size, latent_dim = session.run(tf.shape(latents))
  latent_vars = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'generator/latent_vars' in c.name][0]
  sequence_vars1 = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'predictor1/Input_layer' in c.name][0]
  sequence_vars2 = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'predictor2/Input_layer' in c.name][0]

  assert gen_output.get_shape() == pred_input1.get_shape() == pred_input2.get_shape(), "Generator output and predictor input must match."

  # initialize latent space and corresponding generated sequence
  start_noise = np.random.normal(size=[batch_size, latent_dim])
  session.run(tf.assign(latent_vars, start_noise))
  update_pred_input = [tf.assign(pred_input1, gen_output), tf.assign(pred_input2, gen_output)]

  # calculate relevant gradients
  prior_weight = tf.Variable(alpha, trainable=False)
  session.run(prior_weight.initializer)
  tf.add_to_collection('prior_weight', prior_weight)
  log_pz = tf.reduce_sum(- latents ** 2, 1)
  target1 = args.target1
  if type(target1) == str:
    if target1 == "max":
      cost1 = tf.reduce_mean(-predictions1)
    elif target1 == "min":
      cost1 = tf.reduce_mean(predictions1)
    else:
      try:
        target1 = float(target1)
        mean1, var1 = tf.nn.moments(predictions1, axes=[0])
        cost1 = 0.5 * (mean1 - tf.cast(target1, tf.float32)) ** 2 + 0.5 * (var1 - 0.0) ** 2
      except Exception as e:
        print(e)
        raise Exception("Argument 'target1' must be either 'max', 'min', or a number")
  target2 = args.target2
  if type(target2) == str:
    if target2 == "max":
      cost2 = tf.reduce_mean(-predictions2)
    elif target2 == "min":
      cost2 = tf.reduce_mean(predictions2)
    else:
      try:
        target2 = float(target2)
        mean2, var2 = tf.nn.moments(predictions2, axes=[0])
        cost2 = 0.5 * (mean2 - tf.cast(target2, tf.float32)) ** 2 + 0.5 * (var2 - 0.0) ** 2
      except:
        raise Exception("Argument 'target2' must be either 'max', 'min', or a number")
  cost = args.target1_scale * cost1 + cost2

  grad_cost_seq = tf.reduce_sum(tf.gradients(ys=cost, xs=pred_input),0)

  grad_cost_latent = tf.gradients(ys=gen_output, xs=latents, grad_ys=grad_cost_seq)[0] + prior_weight * tf.squeeze(
    tf.gradients(ys=tf.reduce_mean(log_pz), xs=latents))
  # gives dcost/dz_j] for each latent entry z_j

  noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=args.noise)
  global_step = tf.Variable(args.step_size, trainable=False)
  session.run(global_step.initializer)
  tf.add_to_collection('global_step', global_step)
  if args.optimizer == "adam":
    if args.step_size:
      optimizer = tf.train.AdamOptimizer(learning_rate=global_step)
    else:
      optimizer = tf.train.AdamOptimizer()
    design_op = optimizer.apply_gradients([(grad_cost_latent + noise, latent_vars)])
    adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
    session.run(adam_initializers)
  elif args.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=global_step)
    design_op = optimizer.apply_gradients([(grad_cost_latent + noise, latent_vars)])
  tf.add_to_collection('design_op', design_op)

s = session.run(tf.shape(latents))
session.run(update_pred_input, {latents: np.random.normal(size=s)})

saver = tf.train.Saver()
sigfigs = int(np.floor(np.log10(args.iterations))) + 1
means1 = []
means_onehot1 = []
maxes1 = []
mins1 = []
dist1 = []
means2 = []
means_onehot2 = []
maxes2 = []
mins2 = []
dist2 = []
for ctr in range(args.iterations):
  true_ctr = ctr + checkpoint_baseline + 1
  gen_outputs, _ = session.run([gen_output, design_op], {global_step: step_size, prior_weight: alpha})
  predictor_input, preds = session.run([update_pred_input, predictions])
  mean_pred = [np.mean(p) for p in preds]
  min_pred = [np.min(p) for p in preds]
  max_pred = [np.max(p) for p in preds]
  means1.append(mean_pred[0])
  mins1.append(min_pred[0])
  maxes1.append(max_pred[0])
  dist1.append(preds[0])
  means2.append(mean_pred[1])
  mins2.append(min_pred[1])
  maxes2.append(max_pred[1])
  dist2.append(preds[1])

  if true_ctr == checkpoint_baseline + 1 or true_ctr % args.log_interval == 0:
    pred_onehot = session.run(predictions, {pred_input1: I[np.argmax(predictor_input[0], -1)],
                                            pred_input2: I[np.argmax(predictor_input[1], -1)],
                                            })
    seq0 = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[0], -1))
    mean_pred_onehot1 = np.mean(pred_onehot[0])
    means_onehot1.append(mean_pred_onehot1)
    mean_pred_onehot2 = np.mean(pred_onehot[1])
    means_onehot2.append(mean_pred_onehot2)
    print(
      "Iter {}: {}: score1: {:.6f}; score2: {:.6f}; mean score1: {:.6f}; mean score2: {:.6f}".format(true_ctr, seq0, preds[0][0], preds[1][0],
                                                                                                     mean_pred[0],  mean_pred[1]))

    best_idx = np.argmax(preds[0]+preds[1], 0)
    z = session.run(latents)
    best_seq = "".join(rev_charmap[n] for n in np.argmax(gen_outputs[best_idx], -1))
    print("best: {}".format(best_seq))

    plt.cla()
    # plt.ylim([0., 1.])
    plt.xlabel("Iteration")
    plt.ylabel("Scores of sequences in batch")
    plt.plot(np.linspace(checkpoint_baseline, true_ctr, len(means1)), means1, color='C2',
             label='Mean score of generated sequences');
    if args.plot_mode == "fill":
      plt.fill_between(np.linspace(checkpoint_baseline, true_ctr, len(means1)), mins1, maxes1, color='C0', alpha=0.5,
                       label='Min/max score of generated sequences')
    elif args.plot_mode == "scatter":
      dist_x = np.reshape([[c] * 64 for c in np.linspace(checkpoint_baseline, true_ctr, len(dist1))], [-1])
      plt.scatter(dist_x, np.reshape(dist, [-1]), color='C0', s=0.5, alpha=0.01)
    plt.plot(np.linspace(checkpoint_baseline, true_ctr, len(means_onehot1)), means_onehot1, color='C1', ls='--',
             label='Mean score of one-hot re-encoded seqs')

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

    if args.target1 == "max":
      ax.legend(handles, labels, loc='lower right')
    elif args.target1 == "min":
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
  plt.close()

  # save checkpoint
  if args.checkpoint_iters and true_ctr % args.checkpoint_iters == 0:
    ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(true_ctr))
    os.makedirs(ckpt_dir, exist_ok=True)
    saver.save(session, os.path.join(ckpt_dir, "pp_opt.ckpt"))

print("Done")