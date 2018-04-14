import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='/home/nkilloran/datasets/leo_binding_data/Combined_Max_Myc_Mad_Mad_r_log_normalized.xlsx', help='Dataset location')
parser.add_argument('--output_dir', type=str, default='/home/nkilloran/datasets/leo_binding_data/', help='Dataset location')
parser.add_argument('--valid_frac', type=float, default=0.15, help='Fraction of data to use as validation data')
parser.add_argument('--test_frac', type=float, default=0.15, help='Fraction of data to use as test data')
parser.add_argument('--column_name', type=str, default='Max', help='Which column to get scores from')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

seed = args.seed

df = pd.read_excel(args.file)
df = df.sample(frac=1, random_state=seed)
rows = len(df)

if args.test_frac > 0:
  assert args.test_frac < 1
  test_size = int(rows * args.test_frac)
else:
  test_size = 0

if args.valid_frac > 0:
  assert args.valid_frac < 1
  valid_size = int(rows * args.valid_frac)
else:
  valid_size = 0

train_size = rows - test_size - valid_size

train = df[:train_size]
valid = df[train_size : train_size + valid_size]
test = df[train_size + valid_size:]

save_loc = os.path.join(args.output_dir, args.column_name)
os.makedirs(save_loc, exist_ok=True)

for name in ["train", "valid", "test"]:
  df = globals()[name]
  if len(df) > 0:
    seqs = df["Sequence"]
    seqs.to_csv(os.path.join(save_loc, "{}_data.txt".format(name)), index=False)
    vals = df[args.column_name]
    vals.to_csv(os.path.join(save_loc, "{}_vals.txt".format(name)), index=False)
