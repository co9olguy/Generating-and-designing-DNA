import heapq
import editdistance as ed

set = "test"
if set=="gen":
  seqs_file = "/home/nkilloran/Desktop/Max_pipeline/2017.05.30-17h52m42s_pcnathan/Max_optim/2017.05.30-18h11m30s_pcnathan/samples/samples_100000.txt"
  out_file = "/home/nkilloran/Desktop/Max_pipeline/2017.05.30-17h52m42s_pcnathan/gen_edit_distance.txt"
elif set=="test":
  seqs_file = "/home/nkilloran/datasets/leo_binding_data/Max/test_data.txt"
  out_file = "/home/nkilloran/Desktop/Max_pipeline/2017.05.30-17h52m42s_pcnathan/test_edit_distance.txt"
elif set=="train":
  seqs_file = "/home/nkilloran/datasets/leo_binding_data/Max/train_data.txt"
  out_file = "/home/nkilloran/Desktop/Max_pipeline/2017.05.30-17h52m42s_pcnathan/train_edit_distance.txt"

train_file = "/home/nkilloran/datasets/leo_binding_data/Max/train_data.txt"

with open(seqs_file, "r") as f:
  seqs = [seq.strip() for seq in f.readlines()]

with open(train_file, "r") as f:
  train_seqs = [seq.strip() for seq in f.readlines()]

def min_edit_dist(gen_seq):
  h = []
  for train_seq in train_seqs:
    d = ed.eval(gen_seq, train_seq)
    heapq.heappush(h, d)
  if set=="train":
    return heapq.nsmallest(2, h)[-1]
  else:
    return heapq.nsmallest(1, h)[0]

num_seqs = len(seqs)
batch_size = 250
if batch_size < num_seqs:
  num_batches = num_seqs // batch_size
else:
  batch_size = num_seqs
  num_batches = 1

for idx in range(num_batches):
  batch = seqs[idx * batch_size : (idx + 1) * batch_size]
  d = [[g,min_edit_dist(g)] for g in batch]

  with open(out_file, "a") as f:
    out_str = "\n".join("{} {}".format(tuple_[0], tuple_[1]) for tuple_ in d) + "\n"
    f.write(out_str)
  print("Processed {} of {} sequences.".format((idx + 1) * batch_size, num_seqs))

print("Done")