import os
import json
import pickle
import argparse
from functools import partial
from collections import ChainMap

import h5py
import numpy as np
import tensorflow as tf

from rnnlm_marginal import reader
from rnnlm_marginal import util
from rnnlm_marginal import lm

parser = argparse.ArgumentParser()
parser.add_argument('vocab_path', type=str)
parser.add_argument('exp_dir', type=str)
parser.add_argument('ngram_path', type=str)
parser.add_argument('output_path', type=str)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_trace_splits', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--mini_sample_size', type=int, default=5)
parser.add_argument('--log_level', type=str, default='info')

args = vars(parser.parse_args())

exp_path = partial(os.path.join, args['exp_dir'])
logger = util.get_logger(log_file_path=exp_path('marginal.log'), level=args['log_level'])

vocab = reader.Vocabulary.from_vocab_file(args['vocab_path'])
in_data, out_data, unigrams = reader.read_ngrams(args['ngram_path'], vocab)


with open(exp_path('model_opt.json')) as fp:
    model_opt = json.load(fp)
    model_opt = ChainMap(model_opt, lm.default_rnnlm_opt())
with tf.variable_scope('model') as scope:
    model = lm.create_rnnlm(model_opt, with_dropout=False)
with tf.variable_scope('encoder'):
    encoder = lm.create_encoder(model_opt, model, with_dropout=False)
util.dump_opt(args, logger, name='basic_opt')
util.dump_opt(model_opt, logger, name='model_opt')

for v in tf.trainable_variables():
    logger.info(f'{v.name}, {v.shape}')

logger.info('loading trace...')
_path = os.path.join(args['exp_dir'], 'rmar', 'train_trace.hdf5')
with h5py.File(_path) as f:
    c0 = f['c0'][:, :]
    c1 = f['c1'][:, :]
    h0 = f['h0'][:, :]
    h1 = f['h1'][:, :]
state_size = h1.shape[-1]
trace_key = h1.T
tf_trace_assigns = []
tf_trace_q = tf.placeholder(dtype=tf.float32, shape=(None, state_size))
tf_trace_num = tf.placeholder(dtype=tf.int32, shape=None)
if args['num_trace_splits'] > 1:
    chunks = np.array_split(trace_key, args['num_trace_splits'], axis=-1)
    tf_scores = []
    for i, chunk in enumerate(chunks):
        # c_tf_trace_key = tf.constant(chunk, dtype=tf.float32)
        with tf.device('/cpu:0'):
            c_tf_trace_key = tf.get_variable(
                f'trace_{i}', shape=chunk.shape, trainable=False, dtype=tf.float32)
            c_tf_trace_ph = tf.placeholder(tf.float32, shape=chunk.shape)
            c_tf_assign = tf.assign(c_tf_trace_key, c_tf_trace_ph)
            tf_trace_assigns.append((c_tf_assign, c_tf_trace_ph, chunk))
        tf_scores.append(tf.matmul(tf_trace_q, c_tf_trace_key))
    tf_trace_scores = tf.concat(tf_scores, axis=-1)
else:
    tf_trace_key = tf.constant(trace_key, dtype=tf.float32)
    tf_trace_scores = tf.matmul(tf_trace_q, tf_trace_key)
tf_trace_log_scores = tf.nn.log_softmax(tf_trace_scores)
tf_trace_choices = tf.multinomial(
    tf_trace_scores, tf_trace_num, output_dtype=tf.int32)  # with replacement
tf_cached_scores = tf.get_variable(
    'cached_trace_scores', shape=(args['batch_size'], trace_key.shape[-1]),
    dtype=tf.float32, trainable=False)
tf_update_cache = tf.assign(tf_cached_scores, tf_trace_scores)
tf_cache_trace_choices = tf.multinomial(
    tf_cached_scores, tf_trace_num, output_dtype=tf.int32)

# running model
sess_config = tf.ConfigProto(
    intra_op_parallelism_threads=4,
    inter_op_parallelism_threads=4)
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(var_list=tf.trainable_variables())
saver.restore(sess, os.path.join(args["exp_dir"], 'encoder', 'checkpoint', 'best'))

# initialize trace graph
for assign_op, ph, chunk in tf_trace_assigns:
    sess.run(assign_op, feed_dict={ph: chunk})
del tf_trace_assigns

output = {}

# unigram
logger.info('computing unigram nlls...')
batches = np.array_split(unigrams, len(unigrams)//args['batch_size'] + 1)
if args['batch_size'] == 1:
    batches = batches[:-1]  # remove left over (empty)
for i in range(len(batches)):
    _pad_size = args['batch_size'] - len(batches[i])
    if _pad_size > 0:
        batches[i] = np.concatenate([batches[i], np.zeros((_pad_size, 1))])
        # XXX: there are some padded token in the result

for batch in batches:
    _row_range = np.arange(len(batch))
    batch_input = batch.T
    q_states = sess.run(
        encoder['enc_cell_output'],
        {encoder['full_seq']: batch_input,
         model['batch_size']: len(batch),
         model['seq_len_ph']: np.zeros((len(batch),))})
    log_PZ, __ = sess.run(
        [tf_trace_log_scores, tf_update_cache],
        {tf_trace_q: np.squeeze(q_states, 0)})
    choices = []
    for i_mini in range(args['num_samples'] // args['mini_sample_size']):
        choices.append(sess.run(
            tf_cache_trace_choices, {tf_trace_num: args['mini_sample_size']}))
    choices = np.concatenate(choices, -1)
    for i_sample in range(args['num_samples']):
        # states = ((c0[choices[:, i_sample]], h0[choices[:, i_sample]]),
        #           (c1[choices[:, i_sample]], h1[choices[:, i_sample]]))
        cell_output = h1[choices[:, i_sample]]
        result = sess.run(
            model['token_nll'],
            {model['cell_output']: cell_output[np.newaxis, :, :],
             model['label_token_ph']: batch_input,
             model['token_weight_ph']: np.ones_like(batch_input, dtype=np.float32),
             model['seq_weight_ph']: np.ones((len(batch), ))})
        log_Pz = log_PZ[_row_range, choices[:, i_sample]]
        for i_batch in range(len(batch)):
            _nll = output.setdefault(tuple(batch[i_batch]), [])
            _nll.append((result[0, i_batch], -log_Pz[i_batch]))

# # 1+ grams
logger.info('computing n-gram nlls...')
_row_range = np.arange(args['batch_size'])
feed_dict = {}
for batch in reader.get_batch_iter(in_data, out_data, batch_size=args['batch_size']):
    feed_dict[model['input_token_ph']] = batch.features.inputs
    feed_dict[model['seq_len_ph']] = batch.features.seq_len
    feed_dict[model['label_token_ph']] = batch.labels.label
    feed_dict[model['token_weight_ph']] = batch.labels.label_weight
    feed_dict[model['seq_weight_ph']] = batch.labels.seq_weight

    q_states = sess.run(encoder['enc_cell_output'], feed_dict)[0]  # enc runs backward
    log_PZ, __ = sess.run(
        [tf_trace_log_scores, tf_update_cache], {tf_trace_q: q_states})
    choices = []
    for i_mini in range(args['num_samples'] // args['mini_sample_size']):
        choices.append(sess.run(
            tf_cache_trace_choices, {tf_trace_num: args['mini_sample_size']}))
    choices = np.concatenate(choices, -1)
    for i_sample in range(args['num_samples']):
        states = ((c0[choices[:, i_sample]], h0[choices[:, i_sample]]),
                  (c1[choices[:, i_sample]], h1[choices[:, i_sample]]))
        feed_dict[model['init_state']] = states
        cell_output = h1[choices[:, i_sample]]
        first_input = batch.features.inputs[0:1, :]
        first_nll = sess.run(
            model['token_nll'],
            {model['cell_output']: cell_output[np.newaxis, :, :],
             model['label_token_ph']: first_input,
             model['token_weight_ph']: np.ones_like(first_input, dtype=np.float32),
             model['seq_weight_ph']: np.ones((first_input.shape[1], ))})
        token_nll = sess.run(model['token_nll'], feed_dict)
        log_Pz = log_PZ[_row_range, choices[:, i_sample]]
        for i_batch in range(token_nll.shape[1]):
            if batch.labels.seq_weight[i_batch] == 0:
                continue
            tokens = (batch.features.inputs[0, i_batch],
                      *batch.labels.label[:batch.features.seq_len[i_batch], i_batch])
            _nll = output.setdefault(tokens, [])
            i_token_nll = token_nll[:batch.features.seq_len[i_batch], i_batch]
            _nll.append((first_nll[0, i_batch], *i_token_nll, -log_Pz[i_batch]))

with open(args['output_path'] + '.pkl', 'wb') as f:
    pickle.dump(output, f)
