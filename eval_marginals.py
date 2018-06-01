import os
import pickle
import argparse
from functools import partial

import numpy as np

from rnnlm_marginal import reader


def log_sumexp(logit, axis=-1, keepdims=False):
    max_logit = np.max(logit, axis=axis, keepdims=True)
    exp_logit = np.exp(logit - max_logit)
    logsumexp = max_logit + np.log(exp_logit.sum(axis=axis, keepdims=True))
    if not keepdims:
        logsumexp = np.squeeze(logsumexp, axis=axis)
    return logsumexp


parser = argparse.ArgumentParser()
parser.add_argument('vocab_path', type=str)
parser.add_argument('exp_dir', type=str)
parser.add_argument('marginal_path', type=str)
parser.add_argument('--trace_iw', action='store_true')
parser.add_argument('--trace_rand', action='store_true')
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--progressive', action='store_true')
parser.add_argument('--quiet', action='store_true')
args = vars(parser.parse_args())

exp_path = partial(os.path.join, args['exp_dir'])
vocab = reader.Vocabulary.from_vocab_file(args['vocab_path'])

with open(exp_path('rmar', 'total_tokens.txt')) as f:
    total_tokens = int(f.read())

if args['trace_iw']:
    with open(exp_path('rmar', 'train_trace.size')) as f:
        trace_uniform_ll = np.log(1 / float(f.read()))

with open(args['marginal_path'], mode='rb') as f:
    model_marginal = pickle.load(f)

if args['num_samples'] > 0:
    idx_range = np.arange(args['num_samples'])

loop_iter = range(1)
if args['progressive']:
    loop_iter = range(args['num_samples'])

for _prog in loop_iter:
    data = []
    with open(exp_path('rmar', 'sample.txt.count5-filter')) as lines:
        for line in lines:
            part = line.strip().split('\t')
            ngram = tuple(vocab.w2i(part[0].split(' ')))
            freq = int(part[1])
            if not args['progressive']:
                _prog = args['num_samples'] - 1
            if args['trace_rand'] or args['trace_iw']:
                _model_ll = -np.array(model_marginal[ngram])
                if args['num_samples'] > 0:
                    # _model_ll = _model_ll[0:args['num_samples'] + 1]
                    _random_idx = np.random.choice(
                        idx_range, _prog + 1, replace=False)
                    _model_ll = _model_ll[_random_idx]
                if args['trace_iw']:
                    _model_ll[:, -1] = trace_uniform_ll - _model_ll[:, -1]
                _model_ll = np.sum(_model_ll, axis=1)
                model_ll = np.log(np.exp(log_sumexp(_model_ll)) / len(_model_ll))
            else:
                model_ll = -sum(model_marginal[ngram])
            count_ll = np.log(freq / total_tokens)
            data.append((abs(model_ll - count_ll), freq, len(ngram)))

    data = np.array(data)
    if args['quiet']:
        print(data[:, 0].mean())
    else:
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
            print(
                tick_eval_data[:, 0].mean(),
                tick_eval_data[:, 0].shape[0]/1000,
                sep='\t')
    if not args['progressive']:
        break
