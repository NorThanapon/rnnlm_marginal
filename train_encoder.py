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

train_opt = ChainMap({
        'loss_key': 'mean_l2',
        'init_learning_rate': 0.001,
        'decay_rate': 0.8,
        'staircase': True,
        'optim': 'tensorflow.train.AdamOptimizer',
        # if adam
        'optim_beta1': 0.9,
        'optim_beta2': 0.999,
        'optim_epsilon': 1e-8,
        # fi
        'clip_gradients': 5.0,
        'max_epochs': 10,
        'checkpoint_path': 'tmp',  # auto to exp_dir
        'decay_steps': -1  # if -1 auto to an epoch
        }, lm.default_train_opt())


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('exp_dir', type=str)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--delete', action='store_true')
args = vars(parser.parse_args())

exp_path = partial(os.path.join, args['exp_dir'])
encoder_path = partial(
    os.path.join, os.path.join(args['exp_dir'], 'encoder'))
data_path = partial(os.path.join, args['data_dir'])
util.ensure_dir(os.path.join(args['exp_dir'], 'encoder'), delete=args['delete'])

with open(exp_path('reader_opt.json')) as fp:
    reader_opt = json.load(fp)
    reader_opt = ChainMap(reader_opt, reader.default_reader_opt())

logger = util.get_logger(
    log_file_path=encoder_path('train_encoder.log'), level=args['log_level'])
reader_opt.update({
    'vocab_path': data_path('vocab.txt'),
    'text_path': data_path('train.txt'),
    'min_seq_len': 2,
    'max_seq_len': 7,
})

# loading training data and validating data
train_iter_wrapper = reader.get_batch_iter_from_file(reader_opt)
reader_opt['text_path'] = data_path('valid.txt')
valid_iter_wrapper = reader.get_batch_iter_from_file(
    reader_opt, train_iter_wrapper.vocab)

# creating models
with open(exp_path('model_opt.json')) as fp:
    model_opt = json.load(fp)
    model_opt = ChainMap(model_opt, lm.default_rnnlm_opt())
model_opt['vocab_size'] = train_iter_wrapper.vocab.vocab_size
# model_opt['rnn_get_all_states'] = True
with tf.variable_scope('model'):
    eval_lm = lm.create_rnnlm(model_opt, with_dropout=False)
with tf.variable_scope('encoder'):
    train_enc = lm.create_encoder(model_opt, eval_lm, with_dropout=True)
with tf.variable_scope('encoder', reuse=True):
    eval_enc = lm.create_encoder(model_opt, eval_lm, with_dropout=False)

train_model = ChainMap(train_enc, eval_lm)
eval_model = ChainMap(eval_enc, eval_lm)

# creating optim
if train_opt['decay_steps'] == -1:
    train_opt['decay_steps'] = train_iter_wrapper.num_batches
train_opt['checkpoint_path'] = encoder_path('checkpoint')
optim_op, learning_rate = lm.create_optimizer(
    train_opt, train_enc['mean_l2'], var_list=tf.trainable_variables('encoder/*'))

# dumping options for future reference
reader_opt['text_path'] = data_path('___.txt')
util.dump_opt(
    reader_opt, logger, name='reader_opt', fpath=encoder_path('reader_opt.json'))
util.dump_opt(
    model_opt, logger, name='model_opt', fpath=encoder_path('model_opt.json'))
util.dump_opt(
    train_opt, logger, name='train_opt', fpath=encoder_path('train_opt.json'))

for v in tf.trainable_variables('encoder/*'):
    logger.info(f'{v.name}, {v.shape}')

# running model
sess_config = tf.ConfigProto(
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=tf.trainable_variables('model/*'))
saver.restore(sess, os.path.join(args["exp_dir"], 'checkpoint', 'best'))

lm.train(
    train_opt, sess, train_model, optim_op, learning_rate,
    train_iter_wrapper, eval_model, valid_iter_wrapper, logger,
    report_key='mean_l2', report_exp=False)
