import os
import lib
import socket
import argparse
import tensorflow as tf
import numpy as np

gt_model_loc = "/home/nkilloran/Documents/projects/dna-gen/trained_models/predictors/max/ground_truth/2017.05.30-14h48m05s_pcnathan/checkpoints/checkpoint_75/trained_predictor.ckpt"
data_filepath = '/home/nkilloran/Documents/projects/dna-gen/logs/pp_test/2017.05.30-15h09m02s_pcnathan/samples/samples_100000.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--ground_truth_model', type=str, default=gt_model_loc, help='Location of model to use as ground truth for scores')
parser.add_argument('--data_filepath', type=str, default=data_filepath, help='Full filepath of sequences to score against ground truth')
parser.add_argument('--out_loc', type=str, default=None, help='Where to save the output scores')
parser.add_argument('--out_name', type=str, default="gen_seq_scores.txt", help="Name of scores output file")
parser.add_argument('--vocab', type=str, default="dna", help="Type of vocab")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

args = parser.parse_args()

charmap, rev_charmap = lib.dna.get_vocab(args.vocab)

# restore trained model
session = tf.Session()
new_saver = tf.train.import_meta_graph(args.ground_truth_model + ".meta")
new_saver.restore(session, args.ground_truth_model)
inputs = tf.get_collection('inputs')[0]
predictions = tf.get_collection('predictions')[0]

# load data
filepath, filename = os.path.split(args.data_filepath)
data = lib.dna.load(filepath, vocab="dna", filenames=filename)

def test_preds(dataset, batch_size, out_loc, name):
  data_feed = lib.feed(dataset, batch_size, reuse=False)
  data = next(data_feed)
  preds = []
  while data is not None:
    seqs = data
    preds.append(session.run(predictions, {inputs: seqs}))
    data = next(data_feed)
  preds = np.hstack(preds)
  with open(os.path.join(out_loc, name), "w") as f:
    f.write("Scores against ground truth model saved at {}:{}\n".format(socket.gethostname(), args.ground_truth_model))
    f.write("\n".join(str(p) for p in preds))

if args.out_loc:
  out_loc = args.out_loc
else:
  out_loc = filepath

test_preds(data, args.batch_size, out_loc, args.out_name)
print("Done")