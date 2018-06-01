import time
from functools import partial
from pydoc import locate

import numpy as np
import tensorflow as tf

from rnnlm_marginal import util
from rnnlm_marginal import cell


tfph = tf.placeholder
tfphdf = tf.placeholder_with_default


def _matmul(mat, mat2d, transpose_b=False):
    """return multiplication of 3D tensor and 2D tensor."""
    if len(mat.get_shape()) < 3:
        return tf.matmul(mat, mat2d, transpose_b=transpose_b)
    mat3d = mat
    mat3d_dim = int(mat3d.get_shape()[-1])
    if transpose_b:
        mat2d_dim = int(mat2d.get_shape()[0])
    else:
        mat2d_dim = int(mat2d.get_shape()[-1])
    output_shapes = tf.unstack(tf.shape(mat3d))
    output_shapes[-1] = mat2d_dim
    output_shape = tf.stack(output_shapes)
    flat_mat3d = tf.reshape(mat3d, [-1, mat3d_dim])
    outputs = tf.matmul(flat_mat3d, mat2d, transpose_b=transpose_b)
    return tf.reshape(outputs, output_shape)


def default_rnnlm_opt():
    return {
        'vocab_size': 10000,
        'emb_dim': 650,
        'rnn_dim': 650,
        'rnn_layers': 2,
        'rnn_variational': True,
        'rnn_input_keep_prob': 0.5,
        'rnn_layer_keep_prob': 0.3,
        'rnn_output_keep_prob': 0.5,
        'rnn_get_all_states': False,
        'rnn_state_keep_prob': 0.5,
        'rnn_state_reset_prob': 0.0,
        'rnn_init_state_trainable': False,
        'logit_weight_tying': True
    }


def create_rnnlm(opt, with_dropout=False):
    # placeholders
    input_token_ph = tfph(tf.int32, shape=[None, None], name='input_tokens')
    label_token_ph = tfph(tf.int32, shape=[None, None], name='label_tokens')
    seq_len_ph = tfph(tf.int32, shape=[None], name='seq_lens')
    token_weight_ph = tfph(tf.float32, shape=[None, None], name='token_weights')
    seq_weight_ph = tfph(tf.float32, shape=[None], name='seq_weights')

    # embeddings
    emb_var = tf.get_variable(
        'embeddings', shape=[opt['vocab_size'], opt['emb_dim']], dtype=tf.float32)
    input_token_emb = tf.nn.embedding_lookup(emb_var, input_token_ph)

    # rnn cell
    _rnn_input_size = opt['emb_dim']
    _rnn_cells = []
    _ikp = opt['rnn_input_keep_prob']
    _lkp = opt['rnn_layer_keep_prob']
    _okp = opt['rnn_output_keep_prob']
    _skp = opt['rnn_state_keep_prob']
    _tfdropoutwrapper = partial(
        tf.nn.rnn_cell.DropoutWrapper,
        state_keep_prob=_skp, variational_recurrent=opt['rnn_variational'],
        dtype=tf.float32)
    for layer in range(opt['rnn_layers']):
        _rnn_cell = tf.nn.rnn_cell.LSTMCell(opt['rnn_dim'], cell_clip=1.0)
        if with_dropout and any([_kp < 1.0 for _kp in (_ikp, _lkp, _skp)]):
            _rnn_cell = _tfdropoutwrapper(
                _rnn_cell, input_keep_prob=_ikp, output_keep_prob=_lkp,
                input_size=_rnn_input_size)
        _ikp = 1.0
        if layer == opt['rnn_layers'] - 2:
            _lkp = _okp
        _rnn_input_size = opt['rnn_dim']
        _rnn_cells.append(_rnn_cell)
    final_cell = tf.nn.rnn_cell.MultiRNNCell(_rnn_cells)
    rnn_cell = final_cell

    # state reset
    if opt['rnn_state_reset_prob'] > 0 or opt['rnn_init_state_trainable']:
        if with_dropout:
            _reset_prob = opt['rnn_state_reset_prob']
        else:
            _reset_prob = 0.0
        final_cell = cell.InitStateCellWrapper(
            final_cell, state_reset_prob=_reset_prob,
            trainable=opt['rnn_init_state_trainable'],
            dtype=tf.float32, actvn=tf.nn.tanh, output_reset=False)

    # get all states
    if opt['rnn_get_all_states']:
        final_cell = cell.StateOutputCellWrapper(final_cell)

    # rnn unroll states
    _input_shape = tf.shape(input_token_ph, out_type=tf.int32)
    seq_len = _input_shape[0]
    batch_size = _input_shape[1]
    init_state = final_cell.zero_state(batch_size, tf.float32)
    cell_output, final_state = tf.nn.dynamic_rnn(
        final_cell, input_token_emb, sequence_length=seq_len_ph, initial_state=init_state,
        dtype=tf.float32, time_major=True)

    if opt['rnn_get_all_states']:
        cell_output, all_states = cell_output

    # logit
    logit_b_var = tf.get_variable(
        'logit_b', shape=[opt['vocab_size']], dtype=tf.float32,
        initializer=tf.zeros_initializer())
    logit_w_var = emb_var
    if not opt['logit_weight_tying']:
        logit_w_var = tf.get_variable(
            'logit_w', shape=[opt['vocab_size'], opt['emb_dim']], dtype=tf.float32)
    logit = _matmul(cell_output, logit_w_var, transpose_b=True) + logit_b_var

    # loss
    token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_token_ph, logits=logit)
    token_nll = token_nll * token_weight_ph * seq_weight_ph
    sum_token_nll = tf.reduce_sum(token_nll)
    mean_token_nll = sum_token_nll / tf.reduce_sum(token_weight_ph)

    # unigram loss
    _start_state = final_cell.zero_state(batch_size, tf.float32)
    # XXX: 2-layer LSTM
    first_output = tf.tile(_start_state[-1].h[tf.newaxis, :, :], [seq_len, 1, 1])
    unigram_token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_token_ph,
        logits=_matmul(first_output, logit_w_var, transpose_b=True) + logit_b_var)
    unigram_token_nll * token_weight_ph * seq_weight_ph
    sum_unigram_token_nll = tf.reduce_sum(unigram_token_nll)
    mean_unigram_token_nll = sum_unigram_token_nll / tf.reduce_sum(token_weight_ph)
    # mean_token_nll = tf.Print(mean_token_nll, [mean_token_nll], 'll')
    return {k: v for k, v in locals().items() if not k.startswith('_')}


def create_marginal_rnnlm(lm):
    ngram_token_ph = tfph(tf.int32, shape=[None, None], name='ngram_tokens')
    ngram_logprob_ph = tfph(tf.float32, shape=[None], name='ngram_log_probs')
    ngram_len_ph = tfph(tf.int32, shape=[None], name='ngram_lens')
    ngram_token_weight_ph = tfph(
        tf.float32, shape=[None, None], name='ngram_token_weights')
    ngram_weight_ph = tfph(tf.float32, shape=[None], name='ngram_weights')
    ngram_token_emb = tf.nn.embedding_lookup(lm['emb_var'], ngram_token_ph)
    rnn_cell = lm['rnn_cell']
    _input_shape = tf.shape(ngram_token_ph, out_type=tf.int32)
    seq_len = _input_shape[0]
    batch_size = _input_shape[1]
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    cell_output, final_state = tf.nn.dynamic_rnn(
        rnn_cell, ngram_token_emb, sequence_length=ngram_len_ph-1,
        initial_state=init_state,
        dtype=tf.float32, time_major=True)
    # XXX: 2-layer LSTM Cell
    first_output = init_state[-1].h[tf.newaxis, :, :]
    cell_output = tf.concat([first_output, cell_output[:-1, :, :]], 0)
    logit = _matmul(cell_output, lm['logit_w_var'], transpose_b=True) + lm['logit_b_var']
    token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=ngram_token_ph, logits=logit)
    token_nll = token_nll * ngram_token_weight_ph * ngram_weight_ph
    ngram_logprob_model = -tf.reduce_sum(token_nll, axis=0)
    ngram_prob_model = tf.exp(ngram_logprob_model)
    # log_ratio = tf.stop_gradient(ngram_logprob_model - ngram_logprob_ph)
    # ngram_loss = ngram_prob_model * log_ratio

    ngram_loss = tf.squared_difference(ngram_logprob_model, ngram_logprob_ph) / 2
    mean_ngram_loss = tf.reduce_mean(ngram_loss)
    # mean_ngram_loss = tf.Print(mean_ngram_loss, [mean_ngram_loss], 'ngram')
    return {k: v for k, v in locals().items() if not k.startswith('_')}


def _sample_from_logit(logit):
    idx = tf.cast(tf.multinomial(logit, 1), tf.int32)
    gather_idx = tf.expand_dims(
        tf.range(start=0, limit=tf.shape(idx)[0]), axis=-1)
    gather_idx = tf.concat([gather_idx, idx], axis=-1)
    score = tf.gather_nd(tf.nn.log_softmax(logit), gather_idx)
    idx = tf.squeeze(idx, axis=(1, ))
    return idx, score


def create_sampler(lm, max_len=35, start_id=0):
    temperature_ph = tfphdf(1.0, shape=[], name='temperature')
    batch_size_ph = tfphdf(1, shape=[], name='batch_size')
    init_state = lm['final_cell'].zero_state(batch_size_ph, tf.float32)
    _init_input = tf.tile([start_id], [batch_size_ph])
    _gen_ta = tf.TensorArray(tf.int32, size=max_len)
    _init_values = (tf.constant(0), _init_input, init_state, _gen_ta)

    def _cond(t, _input_token, _state, _output_token):
        return tf.less(t, max_len)

    def _step(t, input_token, state, output_token):
        input_emb = tf.nn.embedding_lookup(lm['emb_var'], input_token)
        with tf.variable_scope('rnn', reuse=True):
            output, new_state = lm['final_cell'](input_emb, state)
        logit = tf.matmul(output, lm['logit_w_var'], transpose_b=True)
        logit += lm['logit_b_var']
        logit /= temperature_ph
        next_token, __ = _sample_from_logit(logit)
        output_token = output_token.write(t, next_token)
        return t + 1, next_token, new_state, output_token

    _t, _i, final_state, samples = tf.while_loop(
        _cond, _step, _init_values, back_prop=False)
    samples = samples.stack()
    return {k: v for k, v in locals().items() if not k.startswith('_')}


def create_encoder(opt, lm, with_dropout=False):
    full_seq = tf.concat([lm['input_token_ph'][0:1], lm['label_token_ph']], 0)
    rev_full_seq = tf.reverse_sequence(
        full_seq, seq_lengths=lm['seq_len_ph'] + 1, seq_axis=0, batch_axis=1)
    enc_input_token_emb = tf.nn.embedding_lookup(lm['emb_var'], rev_full_seq)
    # rnn cell
    _rnn_input_size = opt['emb_dim']
    _rnn_cells = []
    _ikp = opt['rnn_input_keep_prob']
    _lkp = opt['rnn_layer_keep_prob']
    _okp = opt['rnn_output_keep_prob']
    _skp = opt['rnn_state_keep_prob']
    _tfdropoutwrapper = partial(
        tf.nn.rnn_cell.DropoutWrapper,
        state_keep_prob=_skp, variational_recurrent=opt['rnn_variational'],
        dtype=tf.float32)
    for _layer in range(opt['rnn_layers']):
        _rnn_cell = tf.nn.rnn_cell.LSTMCell(opt['rnn_dim'], cell_clip=1.0)
        if with_dropout and any([_kp < 1.0 for _kp in (_ikp, _lkp, _skp)]):
            _rnn_cell = _tfdropoutwrapper(
                _rnn_cell, input_keep_prob=_ikp, output_keep_prob=_lkp,
                input_size=_rnn_input_size)
        _ikp = 1.0
        if _layer == opt['rnn_layers'] - 2:
            _lkp = _okp
        if _layer == opt['rnn_layers'] - 1:
            _okp = 1.0
        _rnn_input_size = opt['rnn_dim']
        _rnn_cells.append(_rnn_cell)
    enc_final_cell = tf.nn.rnn_cell.MultiRNNCell(_rnn_cells)

    # rnn unroll states
    enc_init_state = enc_final_cell.zero_state(lm['batch_size'], tf.float32)
    enc_cell_output, enc_final_state = tf.nn.dynamic_rnn(
        enc_final_cell, enc_input_token_emb, sequence_length=lm['seq_len_ph'] + 1,
        initial_state=enc_init_state, dtype=tf.float32, time_major=True)
    enc_cell_output = tf.reverse_sequence(
        enc_cell_output, seq_lengths=lm['seq_len_ph'] + 1, seq_axis=0, batch_axis=1)
    enc_cell_output = tf.layers.dense(
        enc_cell_output, enc_cell_output.shape[-1], activation=tf.nn.elu)
    enc_cell_output = tf.layers.dense(
        enc_cell_output, enc_cell_output.shape[-1], activation=tf.nn.elu)
    enc_cell_output = tf.layers.dense(
        enc_cell_output, enc_cell_output.shape[-1], activation=tf.nn.tanh)
    # loss
    # XXX: 2-layer LSTM
    lm_full_output = tf.concat(
        (tf.expand_dims(lm['init_state'][-1].h, 0), lm['cell_output']), 0)

    l2 = tf.reduce_sum(
        tf.squared_difference(lm_full_output, enc_cell_output) / 2, axis=-1)
    _init_w_shape = (1, lm['batch_size'])
    _weight = tf.concat(
        [tf.ones(_init_w_shape, dtype=tf.float32), lm['token_weight_ph']], 0)
    l2 = l2 * _weight
    mean_l2 = tf.reduce_sum(l2) / tf.reduce_sum(_weight)
    return {k: v for k, v in locals().items() if not k.startswith('_')}


def create_optimizer(opt, loss, var_list=None):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(
        opt['init_learning_rate'], global_step, opt['decay_steps'], opt['decay_rate'],
        opt['staircase'])
    _optim_opt = {k[len('optim_'):]: v for k, v in opt.items() if k.startswith('optim_')}
    Optim = locate(opt['optim'])
    optim = Optim(learning_rate, **_optim_opt)
    if var_list is None:
        var_list = tf.trainable_variables()
    g_v_pairs = optim.compute_gradients(loss, var_list=var_list)
    grads, tvars = [], []
    for g, v in g_v_pairs:
        if g is None:
            continue
        tvars.append(v)
        grads.append(g)
    clipped_grads, _norm = tf.clip_by_global_norm(grads, opt['clip_gradients'])
    optim_op = optim.apply_gradients(zip(clipped_grads, tvars), global_step=global_step)
    return optim_op, learning_rate


def default_train_opt():
    return {
        'init_learning_rate': 1.0,
        'decay_steps': -1,
        'decay_rate': 0.8,
        'staircase': True,
        'optim': 'tensorflow.train.GradientDescentOptimizer',
        'clip_gradients': 5.0,
        'max_epochs': 20,
        'checkpoint_path': 'tmp'
    }


def _feed_lm(feed_dict, lm, batch):
    feed_dict[lm['input_token_ph']] = batch.features.inputs
    feed_dict[lm['seq_len_ph']] = batch.features.seq_len
    feed_dict[lm['label_token_ph']] = batch.labels.label
    feed_dict[lm['token_weight_ph']] = batch.labels.label_weight
    feed_dict[lm['seq_weight_ph']] = batch.labels.seq_weight


def _run_epoch(sess, model, batch_wrapper, *fetch, feed_dict=None):
    if len(fetch) == 0:
        raise ValueError('`fetch` needs to have at least one element.')
    state = None
    if batch_wrapper.keep_state:
        state = sess.run(
            model['init_state'], {model['batch_size']: batch_wrapper.batch_size})
        fetch = [fetch, model['final_state']]
    else:
        fetch = [fetch, tf.no_op()]
    if feed_dict is None:
        feed_dict = {}
    total_time = 0.0
    for batch in batch_wrapper.iter():
        _feed_lm(feed_dict, model, batch)
        if state is not None:
            feed_dict[model['init_state']] = state
        x = time.time()
        results, state = sess.run(fetch, feed_dict=feed_dict)
        total_time += time.time() - x
        yield results, batch


def train(
        opt, sess, train_model, optim_op, learning_rate, train_batches,
        eval_model, valid_batches, logger, report_key='mean_token_nll', report_exp=True):
    checkpoint_path = opt['checkpoint_path']
    best_saver = tf.train.Saver(var_list=tf.trainable_variables())
    latest_saver = tf.train.Saver(var_list=tf.trainable_variables())
    epoch_fn = partial(_run_epoch, sess)
    best_loss = float('inf')
    steps = 0
    for epoch in range(opt['max_epochs']):
        # train
        ep_train_loss = 0.0
        ep_train_num_tokens = 0
        ep_steps = 0
        _start_time = time.time()
        for results, batch in epoch_fn(
                train_model, train_batches,
                optim_op, train_model[report_key], learning_rate):
            ep_train_loss += results[1] * batch.num_tokens
            ep_train_num_tokens += batch.num_tokens
            steps += 1
            ep_steps += 1
        avg_ep_train_loss = ep_train_loss / ep_train_num_tokens
        train_loss = avg_ep_train_loss
        if report_exp:
            train_loss = np.exp(avg_ep_train_loss)
        _seconds = time.time() - _start_time
        logger.info(
            f'{epoch + 1}: train {train_loss:.5f}, lr {results[2]:.5f}, '
            f'{ep_steps} steps in {_seconds:.1f}s')
        # valid
        ep_valid_loss = 0.0
        ep_valid_num_tokens = 0
        for results, batch in epoch_fn(
                eval_model, valid_batches, eval_model[report_key]):
            ep_valid_loss += results[0] * batch.num_tokens
            ep_valid_num_tokens += batch.num_tokens
        avg_ep_valid_loss = ep_valid_loss / ep_valid_num_tokens
        valid_loss = avg_ep_valid_loss
        if report_exp:
            valid_loss = np.exp(avg_ep_valid_loss)
        logger.info(f'{epoch + 1}: valid {valid_loss:.5f}')
        if avg_ep_valid_loss < best_loss:
            avg_ep_valid_loss = best_loss
            best_saver.save(sess, f'{checkpoint_path}/best')
        latest_saver.save(sess, f'{checkpoint_path}/latest')


def test(
        sess, eval_model, test_batches, logger, checkpoint_path=None, trace=False,
        report_key='mean_token_nll', report_exp=True):
    if checkpoint_path is not None:
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, f'{checkpoint_path}/best')
    epoch_fn = partial(
        _run_epoch, sess, eval_model, test_batches, eval_model[report_key])
    if trace:
        epoch_fn = partial(epoch_fn, eval_model['all_states'])
        states = []
    # valid
    ep_test_loss = 0.0
    ep_test_num_tokens = 0

    for results, batch in epoch_fn():
        ep_test_loss += results[0] * batch.num_tokens
        ep_test_num_tokens += batch.num_tokens
        if trace:
            # weight = batch.labels.label_weight[:, :, np.newaxis]
            # states.append([_s * weight for _s in util.flatten(results[1])])
            # no need to mask if batch size is 1
            states.append(util.flatten(results[1]))

    avg_ep_test_loss = ep_test_loss / ep_test_num_tokens
    test_loss = avg_ep_test_loss
    if report_exp:
        test_loss = np.exp(avg_ep_test_loss)
    logger.info(f'test {test_loss:.5f}')
    if trace:
        return states


def sample(sess, decoder, batch_size, checkpoint_path=None):
    if checkpoint_path is not None:
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, f'{checkpoint_path}/best')
    state = sess.run(decoder['init_state'], {decoder['batch_size_ph']: batch_size})
    while True:
        samples, state = sess.run(
            [decoder['samples'], decoder['final_state']],
            {decoder['batch_size_ph']: batch_size, decoder['init_state']: state})
        yield samples


def _feed_ngram_kl(feed_dict, mm, batch):
    feed_dict[mm['ngram_token_ph']] = batch.features.inputs
    feed_dict[mm['ngram_len_ph']] = batch.features.seq_len
    feed_dict[mm['ngram_logprob_ph']] = batch.labels.label
    feed_dict[mm['ngram_token_weight_ph']] = batch.labels.label_weight
    feed_dict[mm['ngram_weight_ph']] = batch.labels.seq_weight


def _run_epoch_ngram_kl(
        sess, model, mar_model, batch_wrapper, ngram_batch_wrapper,
        *fetch, feed_dict=None):
    if len(fetch) == 0:
        raise ValueError('`fetch` needs to have at least one element.')
    state = None
    if batch_wrapper.keep_state:
        state = sess.run(
            model['init_state'], {model['batch_size']: batch_wrapper.batch_size})
        fetch = [fetch, model['final_state']]
    else:
        fetch = [fetch, tf.no_op()]
    if feed_dict is None:
        feed_dict = {}
    total_time = 0.0
    ngram_batches = ngram_batch_wrapper.iter()
    for batch in batch_wrapper.iter():
        ngram_batch = next(ngram_batches, None)
        if ngram_batch is None:
            ngram_batches = ngram_batch_wrapper.iter()
            ngram_batch = next(ngram_batches, None)
        _feed_lm(feed_dict, model, batch)
        _feed_ngram_kl(feed_dict, mar_model, ngram_batch)
        if state is not None:
            feed_dict[model['init_state']] = state
        x = time.time()
        results, state = sess.run(fetch, feed_dict=feed_dict)
        total_time += time.time() - x
        yield results, batch


def train_ngram_kl(
        opt, sess, train_model, optim_op, learning_rate, train_batches,
        eval_model, valid_batches, logger,
        mar_model, ngram_batch_wrapper,
        report_key='mean_token_nll', report_exp=True):
    checkpoint_path = opt['checkpoint_path']
    best_saver = tf.train.Saver(var_list=tf.trainable_variables())
    latest_saver = tf.train.Saver(var_list=tf.trainable_variables())
    best_loss = float('inf')
    steps = 0
    for epoch in range(opt['max_epochs']):
        # train
        ep_train_loss = 0.0
        ep_train_num_tokens = 0
        ep_steps = 0
        _start_time = time.time()
        for results, batch in _run_epoch_ngram_kl(
                sess, train_model, mar_model, train_batches, ngram_batch_wrapper,
                optim_op, train_model[report_key], learning_rate):
            ep_train_loss += results[1] * batch.num_tokens
            ep_train_num_tokens += batch.num_tokens
            steps += 1
            ep_steps += 1
        avg_ep_train_loss = ep_train_loss / ep_train_num_tokens
        train_loss = avg_ep_train_loss
        if report_exp:
            train_loss = np.exp(avg_ep_train_loss)
        _seconds = time.time() - _start_time
        logger.info(
            f'{epoch + 1}: train {train_loss:.5f}, lr {results[2]:.5f}, '
            f'{ep_steps} steps in {_seconds:.1f}s')
        # valid
        ep_valid_loss = 0.0
        ep_valid_num_tokens = 0
        for results, batch in _run_epoch(
                sess, eval_model, valid_batches, eval_model[report_key]):
            ep_valid_loss += results[0] * batch.num_tokens
            ep_valid_num_tokens += batch.num_tokens
        avg_ep_valid_loss = ep_valid_loss / ep_valid_num_tokens
        valid_loss = avg_ep_valid_loss
        if report_exp:
            valid_loss = np.exp(avg_ep_valid_loss)
        logger.info(f'{epoch + 1}: valid {valid_loss:.5f}')
        if avg_ep_valid_loss < best_loss:
            avg_ep_valid_loss = best_loss
            best_saver.save(sess, f'{checkpoint_path}/best')
        latest_saver.save(sess, f'{checkpoint_path}/latest')


