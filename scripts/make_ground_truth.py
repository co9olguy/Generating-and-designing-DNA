import os
import lib
import socket
import argparse
import tensorflow as tf
import numpy as np

model_loc = "/home/nkilloran/Documents/projects/dna-gen/trained_models/2017.05.25-11h36m31s_pcnathan/checkpoints/checkpoint_75/trained_predictor.ckpt"
data_loc= '/home/nkilloran/datasets/leo_binding_data/Max'
out_loc = '/home/nkilloran/datasets/leo_binding_data/Max'

parser = argparse.ArgumentParser()
parser.add_argument('--model_loc', type=str, default=model_loc, help='Location of model to use as ground truth')
parser.add_argument('--data_loc', type=str, default=data_loc, help='Data location')
parser.add_argument('--out_loc', type=str, default=out_loc, help='Directory to put the ground truth data in')
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna' or 'rna'")
args = parser.parse_args()

batch_size = 64
charmap, rev_charmap = lib.dna.get_vocab(args.vocab)

# restore trained model
session = tf.Session()
new_saver = tf.train.import_meta_graph(args.model_loc + ".meta")
new_saver.restore(session, args.model_loc)
inputs = tf.get_collection('inputs')[0]
predictions = tf.get_collection('predictions')[0]

# load data
train_data, valid_data, test_data = lib.dna.load(args.data_loc, vocab=args.vocab, valid=True, test=True, scores=False)

def save_data(dataset, name, size):
  chars = np.argmax(dataset, -1)
  seqs_list = []
  for row in chars:
    seqs_list.append("".join(rev_charmap[c] for c in row))
  s = "\n".join(seq for seq in seqs_list[:size])
  os.makedirs(os.path.join(args.out_loc, "sim_ground_truth_data"), exist_ok=True)
  with open(os.path.join(args.out_loc, "sim_ground_truth_data", name), "w") as f:
    f.write("Ground truth from model saved at {}:{}\n".format(socket.gethostname(), model_loc))
    f.write(s)
  
  
def save_preds(dataset, batch_size, name):
  data_feed = lib.feed(dataset, batch_size, reuse=False)
  data = next(data_feed)
  preds = []
  while data is not None:
    seqs = data
    preds.append(session.run(predictions, {inputs: seqs}))
    data = next(data_feed)
  preds = np.hstack(preds)
  os.makedirs(os.path.join(args.out_loc, "sim_ground_truth_data"), exist_ok=True)
  with open(os.path.join(args.out_loc, "sim_ground_truth_data", name), "w") as f:
    f.write("Ground truth from model saved at {}:{}\n".format(socket.gethostname(), model_loc))
    f.write("\n".join(str(p) for p in preds))
  return len(preds)

train_preds = save_preds(train_data, batch_size, "train_vals.txt")
save_data(train_data, "train_data.txt", size=train_preds)
valid_preds = save_preds(valid_data, batch_size, "valid_vals.txt")
save_data(valid_data, "valid_data.txt", size=valid_preds)
test_preds = save_preds(test_data, batch_size, "test_vals.txt")
save_data(test_data, "test_data.txt", size=test_preds)

print("Done")