import json
import argparse
import os
from functools import partial
from collections import ChainMap

import numpy as np
import tensorflow as tf

from rnnlm_marginal import reader
from rnnlm_marginal import util
from rnnlm_marginal import lm

# Training hyper-parameters are from Melis et al., 2017, except decay rate
# Model settings (medium) are from Zaremba et al., 2014

RESET_PROB = 0.05
TRAINABLE_STATE = False


def ptb_opt():
    reader_opt = ChainMap({
        'sentences': False,
        # if sentences is False
        'min_seq_len': 35,
        'max_seq_len': 35,
        # fi
        'shuffle': False,
        'batch_size': 64,
        'vocab_path': '',  # auto to data
        'text_path': ''  # auto to data
        }, reader.default_reader_opt())
    model_opt = ChainMap({
        'emb_dim': 650,
        'rnn_dim': 650,
        'rnn_layers': 2,
        'rnn_variational': True,
        'rnn_input_keep_prob': 0.7,
        'rnn_layer_keep_prob': 0.7,
        'rnn_output_keep_prob': 0.7,
        'rnn_state_keep_prob': 0.7,
        'logit_weight_tying': True,
        'vocab_size': -1  # auto to data
        }, lm.default_rnnlm_opt())
    train_opt = ChainMap({
        'loss_key': 'mean_token_nll',  # or sum_token_nll
        'init_learning_rate': 0.003,
        'decay_rate': 0.9,
        'staircase': True,
        'optim': 'tensorflow.train.AdamOptimizer',
        # if adam
        'optim_beta1': 0.0,
        'optim_beta2': 0.999,
        'optim_epsilon': 1e-8,
        # fi
        'clip_gradients': 5.0,
        'max_epochs': 40,
        'checkpoint_path': 'tmp',  # auto to exp_dir
        'decay_steps': -1  # if -1 auto to an epoch
        }, lm.default_train_opt())
    return reader_opt, model_opt, train_opt


def wt2_opt():
    reader_opt = ChainMap({
        'sentences': False,
        # if sentences is False
        'min_seq_len': 35,
        'max_seq_len': 35,
        # fi
        'shuffle': False,
        'batch_size': 64,
        'vocab_path': '',  # auto to data
        'text_path': ''  # auto to data
        }, reader.default_reader_opt())
    model_opt = ChainMap({
        'emb_dim': 650,
        'rnn_dim': 650,
        'rnn_layers': 2,
        'rnn_variational': True,
        'rnn_input_keep_prob': 0.5,
        'rnn_layer_keep_prob': 0.7,
        'rnn_output_keep_prob': 0.5,
        'rnn_state_keep_prob': 0.7,
        'logit_weight_tying': True,
        'vocab_size': -1  # auto to data
        }, lm.default_rnnlm_opt())
    train_opt = ChainMap({
        'loss_key': 'mean_token_nll',  # or sum_token_nll
        'init_learning_rate': 0.003,
        'decay_rate': 0.85,
        'staircase': True,
        'optim': 'tensorflow.train.AdamOptimizer',
        # if adam
        'optim_beta1': 0.0,
        'optim_beta2': 0.999,
        'optim_epsilon': 1e-8,
        # fi
        'clip_gradients': 5.0,
        'max_epochs': 40,
        'checkpoint_path': 'tmp',  # auto to exp_dir
        'decay_steps': -1  # if -1 auto to an epoch
        }, lm.default_train_opt())
    return reader_opt, model_opt, train_opt


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('exp_dir', type=str)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--delete', action='store_true')
parser.add_argument('--reset_state', action='store_true')
parser.add_argument('--unigram_reg', action='store_true')
args = vars(parser.parse_args())

exp_path = partial(os.path.join, args['exp_dir'])
data_path = partial(os.path.join, args['data_dir'])

if 'ptb' in args['data_dir']:
    reader_opt, model_opt, train_opt = ptb_opt()
elif 'wikitext-2' in args['data_dir']:
    reader_opt, model_opt, train_opt = wt2_opt()
else:
    raise ValueError('ptb or wikitext-2')

util.ensure_dir(args['exp_dir'], delete=args['delete'])
logger = util.get_logger(log_file_path=exp_path('train.log'), level=args['log_level'])
reader_opt.update({
    'vocab_path': data_path('vocab.txt'),
    'text_path': data_path('train.txt'),
})

# loading training data and validating data
train_iter_wrapper = reader.get_batch_iter_from_file(reader_opt)
reader_opt['text_path'] = data_path('valid.txt')
valid_iter_wrapper = reader.get_batch_iter_from_file(
    reader_opt, train_iter_wrapper.vocab)

# creating models
model_opt['vocab_size'] = train_iter_wrapper.vocab.vocab_size
if args['reset_state']:
    model_opt['rnn_state_reset_prob'] = RESET_PROB
    model_opt['rnn_init_state_trainable'] = TRAINABLE_STATE
with tf.variable_scope('model'):
    train_model = lm.create_rnnlm(model_opt, with_dropout=True)
with tf.variable_scope('model', reuse=True):
    eval_model = lm.create_rnnlm(model_opt, with_dropout=False)
if train_opt['decay_steps'] == -1:
    train_opt['decay_steps'] = train_iter_wrapper.num_batches
train_opt['checkpoint_path'] = exp_path('checkpoint')
optim_obj = train_model[train_opt['loss_key']]
if args['unigram_reg']:
    optim_obj += train_model['mean_unigram_token_nll']
optim_op, learning_rate = lm.create_optimizer(
    train_opt, optim_obj)

# dumping options for future reference
reader_opt['text_path'] = data_path('___.txt')
util.dump_opt(reader_opt, logger, name='reader_opt', fpath=exp_path('reader_opt.json'))
util.dump_opt(model_opt, logger, name='model_opt', fpath=exp_path('model_opt.json'))
util.dump_opt(train_opt, logger, name='train_opt', fpath=exp_path('train_opt.json'))

for v in tf.trainable_variables():
    logger.info(f'{v.name}, {v.shape}')

# running model
sess_config = tf.ConfigProto(
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
lm.train(
    train_opt, sess, train_model, optim_op, learning_rate,
    train_iter_wrapper, eval_model, valid_iter_wrapper, logger)

reader_opt['text_path'] = data_path('test.txt')
test_iter_wrapper = reader.get_batch_iter_from_file(
    reader_opt, train_iter_wrapper.vocab)
lm.test(sess, eval_model, test_iter_wrapper, logger, train_opt['checkpoint_path'])
