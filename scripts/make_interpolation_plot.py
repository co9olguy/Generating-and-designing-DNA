import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, help='File with pickled interpolation data')
#parser.add_argument('--seq_file', type=str, help='File with interpolation strings')
parser.add_argument('--num_plots', type=int, default=7, help='Number of subplots to show')
parser.add_argument('--out_file', type=str, default="interp_plot.svg", help='Filename to save created plot')
args = parser.parse_args()

colours = np.array([(0,213,0), # green (A)
                    (0,0,192), # blue (C)
                    (255,170,0), # yellow (G)
                    (213,0,0)] # red (T)
                   )/255

with open(args.data_file, "rb") as f:
  data = pickle.load(f)

# Six subplots sharing both x/y axes
f, axes = plt.subplots(1,args.num_plots, sharex=True, sharey=True);

for seq_idx, axis in enumerate(axes):
  for char_idx in range(4):
    plt_data = data[:, seq_idx, char_idx] # data has indices (latent_idx, seq_idx, vocab_idx)
    axis.plot(plt_data[::-1], range(len(plt_data)), color=colours[char_idx]);


# Fine-tune figure; make subplots close to each other
f.subplots_adjust(hspace=0);
plt.setp([a.get_yticklabels() for a in f.axes], visible=False);
plt.setp([a.get_yaxis() for a in f.axes], visible=False);

plt.savefig(args.out_file, format="svg")