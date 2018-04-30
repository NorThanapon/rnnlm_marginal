import re
import os
import argparse
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('exp_parent_dir', type=str)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--data', type=str, default='ptb')
parser.add_argument('--subdirs', type=str, default='v,reset,reset-uni')
parser.add_argument('--labels', type=str, default='LM,LM+Reset,LM+Reset+Uni')
args = vars(parser.parse_args())

get_exp_path = partial(os.path.join, args['exp_parent_dir'])
subdirs = args['subdirs'].split(',')
labels = args['labels'].split(',')
colors = ['green', 'blue', 'orange']
dataset = args['data']


def read_ppl(subdir):
    train_ppl = []
    valid_ppl = []
    ppl_regex = re.compile(r'(\w+) (ppl )?(\d+\.\d+)')
    with open(get_exp_path(f'{dataset}-{subdir}', 'train.log')) as lines:
        for line in lines:
            m = ppl_regex.search(line)
            if m is not None:
                split, __, ppl = m.groups()
                if split == 'train':
                    train_ppl.append(float(ppl))
                elif split == 'valid':
                    valid_ppl.append(float(ppl))
    return train_ppl, valid_ppl


train_ppls = []
valid_ppls = []
for subdir in subdirs:
    _tppl, _vppl = read_ppl(subdir)
    train_ppls.append(_tppl)
    valid_ppls.append(_vppl)

plt.figure(1, dpi=150)
for label, color, tppl, vppl in zip(labels, colors, train_ppls, valid_ppls):
    plt.plot(
        range(len(tppl)), np.log(tppl), label=f'{label}, train',
        alpha=0.6, lw=1.5, c=color, ls='--')
    plt.plot(
        range(len(vppl)), np.log(vppl), label=f'{label}, valid',
        alpha=0.6, lw=1.5, c=color)
plt.grid(True)
plt.xlabel('Epochs', fontsize=9)
plt.ylabel('Loss', fontsize=9)
plt.legend()
# plt.show()
