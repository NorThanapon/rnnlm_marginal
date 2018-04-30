import os
import json
import argparse
from functools import partial
from collections import ChainMap

import numpy as np
import tensorflow as tf

from rnnlm_marginal import reader
from rnnlm_marginal import util
from rnnlm_marginal import lm

parser = argparse.ArgumentParser()
parser.add_argument('vocab_path', type=str)
parser.add_argument('exp_dir', type=str)
parser.add_argument('output_path', type=str)
parser.add_argument('num_output_tokens', type=int)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--log_level', type=str, default='info')
args = vars(parser.parse_args())

exp_path = partial(os.path.join, args['exp_dir'])
logger = util.get_logger(log_file_path=exp_path('sample.log'), level=args['log_level'])

vocab = reader.Vocabulary.from_vocab_file(args['vocab_path'])

with open(exp_path('model_opt.json')) as fp:
    model_opt = json.load(fp)
    model_opt = ChainMap(model_opt, lm.default_rnnlm_opt())
with tf.variable_scope('model') as scope:
    eval_model = lm.create_rnnlm(model_opt, with_dropout=False)
    decoder = lm.create_sampler(eval_model)
util.dump_opt(args, logger, name='basic_opt')
util.dump_opt(model_opt, logger, name='model_opt')

for v in tf.trainable_variables():
    logger.info(f'{v.name}, {v.shape}')

# running model
sess_config = tf.ConfigProto(
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

texts = [[] for __ in range(args['batch_size'])]
num_tokens = 0
for samples in lm.sample(sess, decoder, args['batch_size'], exp_path('checkpoint')):
    lines = vocab.i2w(samples.T)
    for i, line in enumerate(lines):
        num_tokens += len(line)
        line = ' '.join(line)
        line = line.replace('</s>', '\n')
        line = line.replace(' \n ', '\n')
        texts[i].append(line)
    if num_tokens >= args['num_output_tokens']:
        break

with open(args['output_path'], mode='w') as ofp:
    for text in texts:
        ofp.write(' '.join(text))
        ofp.write('\n')
