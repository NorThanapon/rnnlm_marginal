import os
import json
import argparse
from functools import partial
from collections import ChainMap

import h5py
import numpy as np
import tensorflow as tf

from rnnlm_marginal import reader
from rnnlm_marginal import util
from rnnlm_marginal import lm


reader_opt = ChainMap({
    'sentences': False,
    # if sentences is False
    'min_seq_len': 100,
    'max_seq_len': 100,
    # fi
    'shuffle': False,
    'batch_size': 1,  # for accurate PPL, but less efficient
    'vocab_path': '',  # auto to arg
    'text_path': ''  # auto to arg
    }, reader.default_reader_opt())

parser = argparse.ArgumentParser()
parser.add_argument('vocab_path', type=str)
parser.add_argument('text_path', type=str)
parser.add_argument('exp_dir', type=str)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--trace_path', type=str, default='')
args = vars(parser.parse_args())


exp_path = partial(os.path.join, args['exp_dir'])
logger = util.get_logger(log_file_path=exp_path('test.log'), level=args['log_level'])

reader_opt['vocab_path'] = args['vocab_path']
reader_opt['text_path'] = args['text_path']
test_iter_wrapper = reader.get_batch_iter_from_file(reader_opt)

with open(exp_path('model_opt.json')) as fp:
    model_opt = json.load(fp)
    model_opt = ChainMap(model_opt, lm.default_rnnlm_opt())
get_trace = args['trace_path'] != ''
model_opt['rnn_get_all_states'] = get_trace
with tf.variable_scope('model'):
    eval_model = lm.create_rnnlm(model_opt, with_dropout=False)
util.dump_opt(reader_opt, logger, name='reader_opt')
util.dump_opt(model_opt, logger, name='model_opt')

for v in tf.trainable_variables():
    logger.info(f'{v.name}, {v.shape}')

# running model
sess_config = tf.ConfigProto(
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

trace_states = lm.test(
    sess, eval_model, test_iter_wrapper, logger, checkpoint_path=exp_path('checkpoint'),
    trace=get_trace)

if get_trace:
    path = args['trace_path']
    with h5py.File(f'{path}.hdf5', 'w') as f:
        # XXX: 2-layer LSTM
        names = ['c0', 'h0', 'c1', 'h1']
        for name, s in zip(names, zip(*trace_states)):
            s = np.squeeze(np.concatenate(s, axis=0))
            f.create_dataset(name, data=s)
#     trace_states = np.squeeze(np.concatenate(trace_states, axis=0))
#     logger.info(f'trace shape: {trace_states.shape}')
#     np.save(args['trace_path'], trace_states)
