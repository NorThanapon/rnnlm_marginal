import collections

import tensorflow as tf

from rnnlm_marginal.util import nested_map

OutputStateTuple = collections.namedtuple('OutputStateTuple', ('output', 'state'))


class StateOutputCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell):
        self._cell = cell

    @property
    def output_size(self):
        return OutputStateTuple(self._cell.output_size, self._cell.state_size)

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope=scope)
        return OutputStateTuple(output, new_state), new_state


ResetCellOutput = collections.namedtuple('ResetCellOutput', 'output reset')


class InitStateCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(
            self, cell, state_reset_prob=0.0, trainable=False,
            dtype=tf.float32, actvn=tf.nn.tanh, output_reset=False):
        self._cell = cell
        self._init_vars = self._create_init_vars(trainable, dtype, actvn)
        self._dtype = dtype
        self._reset_prob = state_reset_prob
        self._actvn = actvn
        self._output_reset = output_reset

    @property
    def output_size(self):
        if self._output_reset:
            return ResetCellOutput(self._cell.output_size, 1)
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def init_vars(self):
        return self._init_vars

    def _create_init_vars(self, trainable, dtype, actvn=None):
        self._i = 0
        with tf.variable_scope('init_state'):
            def create_init_var(size):
                var = tf.get_variable(
                    f'init_{self._i}', shape=(size, ), dtype=dtype,
                    initializer=tf.zeros_initializer(), trainable=trainable)
                if actvn is not None:
                    var = actvn(var)
                self._i = self._i + 1
                return var
            if (isinstance(self.state_size[0], tf.nn.rnn_cell.LSTMStateTuple) and
               trainable):
                return self._create_lstm_init_vars(trainable, dtype)
            return nested_map(create_init_var, self.state_size)

    def _create_lstm_init_vars(self, trainable, dtype):
        num_layers = len(self.state_size)
        states = []
        for i in range(num_layers):
            state_size = self.state_size[i]
            assert isinstance(state_size, tf.nn.rnn_cell.LSTMStateTuple), \
                '`state_size` is not LSTMStateTuple'
            c = tf.get_variable(
                f'init_{i}_c', shape=(state_size.c, ), dtype=dtype,
                trainable=trainable)
            h = tf.get_variable(
                f'init_{i}_h', shape=(state_size.c, ), dtype=dtype,
                trainable=trainable)
            c = tf.clip_by_value(c, -1.0, 1.0)
            h = tf.tanh(h)
            # h = tf.Print(h, [tf.reduce_mean(h)])
            states.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        return tuple(states)

    def _get_reset(self, inputs):
        # TODO: better way to figure out the batch size
        batch_size = tf.shape(inputs)[0]
        rand = tf.random_uniform((batch_size, ))
        r = tf.cast(tf.less(rand, self._reset_prob), tf.float32)
        r = r[:, tf.newaxis]
        return r, batch_size

    def _get_zero_reset(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros((batch_size, 1), dtype=tf.float32)

    def __call__(self, inputs, state, scope=None):
        r = None
        if self._reset_prob > 0.0:
            r, batch_size = self._get_reset(inputs)
            # r = tf.Print(r, [r])
            state = self.select_state(state, r, batch_size)
        cell_output, new_state = self._cell(inputs, state)
        if self._output_reset:
            if self._reset_prob <= 0.0:
                r = self._get_zero_reset(inputs)
            return ResetCellOutput(cell_output, r), new_state
        else:
            return cell_output, new_state

    def select_state(self, state, r, batch_size):
        def _select(cur_state, init_var):
            return r * (init_var[tf.newaxis, :] - cur_state) + cur_state
        return nested_map(_select, state, self._init_vars)

    def tiled_init_state(self, batch_size, seq_len=None):
        def _tile(var):
            if seq_len is not None:
                return tf.tile(var[tf.newaxis, tf.newaxis, :], (seq_len, batch_size, 1))
            return tf.tile(var[tf.newaxis, :], (batch_size, 1))
        return nested_map(_tile, self._init_vars)

    def zero_state(self, batch_size, dtype):
        assert dtype == self._dtype, \
            'dtype must be the same as dtype during the cell construction'
        return self.tiled_init_state(batch_size)
