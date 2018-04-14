import os
import numpy as np

#seq_file = "/home/nkilloran/datasets/human_genome/exon_min0_max50_win150/train_data.txt"
#align_file = "/home/nkilloran/datasets/human_genome/exon_min0_max50_win150/train_ann.txt"
seq_file = "/home/nkilloran/dna-gen/logs/exon_annotate_tests_new/2017.06.19-16h20m39s_pcnathan/samples/samples_7500"
align_file = "/home/nkilloran/dna-gen/logs/exon_annotate_tests_new/2017.06.19-16h20m39s_pcnathan/samples/samples_ann_7500"
out_loc = "/home/nkilloran/Desktop/"

flank_size = 40

with open(seq_file, "r") as f:
  seqs = np.vstack([np.expand_dims(np.array([c for c in line.strip()]),0) for line in f.readlines()])
a = np.loadtxt(align_file)

exon_starts = np.argmax(a>=0.5, 1)
exon_ends = -np.argmax((a>=0.5)[:,::-1], 1)

start_seqs = []
for idx, start in enumerate(exon_starts):
  start_seqs.append("".join(seqs[idx, start-flank_size//2:start+flank_size//2]))

end_seqs = []
for idx, end in enumerate(exon_ends):
  end_seqs.append("".join(seqs[idx, end-flank_size//2:end+flank_size//2]))

with open(os.path.join(out_loc, "exon_starts_seqs.txt"), "w") as f:
  f.write("\n".join(start_seqs))
with open(os.path.join(out_loc, "exon_ends_seqs.txt"), "w") as f:
  f.write("\n".join(end_seqs))

print("Done")