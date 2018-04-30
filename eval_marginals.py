import os
import pickle
import argparse
from functools import partial

import numpy as np

from rnnlm_marginal import reader

parser = argparse.ArgumentParser()
parser.add_argument('vocab_path', type=str)
parser.add_argument('exp_dir', type=str)
parser.add_argument('marginal_path', type=str)
args = vars(parser.parse_args())

exp_path = partial(os.path.join, args['exp_dir'])
vocab = reader.Vocabulary.from_vocab_file(args['vocab_path'])

with open(exp_path('rmar', 'total_tokens.txt')) as f:
    total_tokens = int(f.read())

with open(args['marginal_path'], mode='rb') as f:
    model_marginal = pickle.load(f)

data = []
with open(exp_path('rmar', 'sample.txt.count5-filter')) as lines:
    for line in lines:
        part = line.strip().split('\t')
        ngram = tuple(vocab.w2i(part[0].split(' ')))
        freq = int(part[1])
        model_ll = -sum(model_marginal[ngram])
        count_ll = np.log(freq / total_tokens)
        data.append((abs(model_ll - count_ll), freq, len(ngram)))

data = np.array(data)
print(data[:, 0].mean(), data[:, 0].shape[0]/1000, sep='\t')

for i in range(1, 6):
    print(
        data[data[:, 2] == i][:, 0].mean(),
        data[data[:, 2] == i][:, 0].shape[0]/1000,
        sep='\t')

ticks = [20, 50, 100, 200, 500, float('inf')]
for i in range(len(ticks) - 1):
    min_count, max_count = ticks[i], ticks[i+1]
    tick_eval_data = data[np.where(
        np.all(np.stack([
            data[:, 1] >= min_count,
            data[:, 1] < max_count], -1), -1))]
    print(tick_eval_data[:, 0].mean(), tick_eval_data[:, 0].shape[0]/1000, sep='\t')
