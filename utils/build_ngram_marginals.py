import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('count_file', type=str)
parser.add_argument('out_file', type=str)
parser.add_argument('num_tokens', type=int)
# 103227008 wikitext-103
# 2051910 wikitext-2
parser.add_argument('--min_ngram_count', type=int, default=50)
parser.add_argument('--max_ngram_len', type=int, default=5)
args = vars(parser.parse_args())

with open(args['count_file']) as lines, open(args['out_file'], 'w') as ofp:
    for i, line in enumerate(lines):
        if '<unk>' in line:
            continue
        ngram, count = line.strip().split('\t')
        if len(ngram.split()) > args['max_ngram_len']:
            continue
        if int(count) < args['min_ngram_count']:
            continue
        ll = np.log(float(count) / args['num_tokens'])
        ngram = ngram.replace('<s>', '</s>')
        ofp.write(f'{ngram}\t{ll}\n')
        if i % 1000 == 0:
            print(i)
