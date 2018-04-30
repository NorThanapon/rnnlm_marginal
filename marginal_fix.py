import os
import json
import pickle
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
parser.add_argument('ngram_path', type=str)
parser.add_argument('output_path', type=str)
parser.add_argument('--batch_size', type=int, default=1)
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

saver = tf.train.Saver(var_list=tf.trainable_variables())
saver.restore(sess, f'{args["exp_dir"]}/checkpoint/best')

output = {}

# unigram
tf_unigram_logits = tf.matmul(
    model['init_state'][-1].h, model['logit_w_var'], transpose_b=True)
tf_unigram_logits += model['logit_b_var']
unigram_nll = sess.run(-tf.nn.log_softmax(tf_unigram_logits), {model['batch_size']: 1})
for i, nll in enumerate(unigram_nll[0]):
    output[(i,)] = [nll]

# 1+ grams
feed_dict = {}
for batch in reader.get_batch_iter(in_data, out_data, batch_size=args['batch_size']):
    feed_dict[model['input_token_ph']] = batch.features.inputs
    feed_dict[model['seq_len_ph']] = batch.features.seq_len
    feed_dict[model['label_token_ph']] = batch.labels.label
    feed_dict[model['token_weight_ph']] = batch.labels.label_weight
    feed_dict[model['seq_weight_ph']] = batch.labels.seq_weight
    result = sess.run(model['token_nll'], feed_dict)
    token_nll = np.concatenate([unigram_nll[:, batch.features.inputs[0]], result])
    for i in range(token_nll.shape[1]):
        if batch.labels.seq_weight[i] == 0:
            continue
        tokens = (batch.features.inputs[0, i],
                  *batch.labels.label[:batch.features.seq_len[i], i])
        output[tokens] = list(token_nll[:, i])[:batch.features.seq_len[i] + 1]

with open(args['output_path'] + '.pkl', 'wb') as f:
    pickle.dump(output, f)
