import six
import random
import collections
from functools import partial
import numpy as np


def read_ngram_lm(filepath, vocab):
    ngram_data, logp_data = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip().split('\t')
            ngram = line[0].split(' ')
            logp = float(line[1])
            token_ids = vocab.w2i(ngram)
            ngram_data.append(token_ids)
            logp_data.append(logp)
    return ngram_data, logp_data


def read_ngrams(filepath, vocab):
    in_data, out_data, unigrams = [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip().split('\t')[0].split(' ')
            token_ids = vocab.w2i(line)
            if len(token_ids) == 1:
                unigrams.append(token_ids)
            else:
                in_data.append(token_ids[0:-1])
                out_data.append(token_ids[1:])
    return in_data, out_data, unigrams


def read_sentences(filepath, vocab):
    eos = [vocab[Vocabulary.special_symbols['end_seq']]]
    in_data, out_data = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip().split(' ')
            token_ids = vocab.w2i(line)
            in_data.append(eos + token_ids)
            out_data.append(token_ids + eos)
    return in_data, out_data


def read_text(filepath, vocab, min_seq_len=20, max_seq_len=20):
    eos = vocab[Vocabulary.special_symbols['end_seq']]
    data = [eos]
    with open(filepath) as f:
        for line in f:
            line = line.strip().split(' ')
            token_ids = vocab.w2i(line)
            data.extend(vocab.w2i(line))
            data.append(eos)
        chunk_in_data, chunk_out_data = [], []
        i = 0
        seq_len = min_seq_len
        while i < len(data) - 1:
            if min_seq_len != max_seq_len:
                seq_len = np.random.randint(low=min_seq_len, high=max_seq_len+1)
            end = i + seq_len
            if end > len(data) - 1:
                end = len(data) - 1
            chunk_in_data.append(data[i: end])
            chunk_out_data.append(data[i + 1: end + 1])
            i += seq_len
    return chunk_in_data, chunk_out_data


def _batch_iter(batch_size, shuffle, data, *more_data, pad=[[]]):
    """iterate over data using equally distant pointers. Left overs are always at the
    last sequences of the last batch.
    """
    all_data = [data] + list(more_data)
    pos = list(range(len(data)))
    num_batch = len(data) // batch_size
    left_over = len(data) % batch_size
    pointers = [0]
    for ibatch in range(1, batch_size):
        pointers.append(pointers[-1] + num_batch)
        if left_over - ibatch >= 0:
            pointers[ibatch] += 1
    if shuffle:
        random.shuffle(pos)
    for i in range(num_batch):
        yield ([d_[pos[p + i]] for p in pointers] for d_ in all_data)
    if left_over > 0:
        if pad is not None:
            # add empty as a pad
            yield ([d_[pos[p + num_batch]] for p in pointers[:left_over]] +
                   [pad[j] for __ in range(batch_size - left_over)]
                   for j, d_ in enumerate(all_data))
        else:
            yield ([d_[pos[p + num_batch]] for p in pointers[:left_over]]
                   for d_ in all_data)


def _hstack_list(data, padding=0, dtype=np.int32):
    lengths = list(map(len, data))
    max_len = max(lengths)
    arr = np.full((max_len, len(data)), padding, dtype=dtype)
    for i, row in enumerate(data):
        arr[0:len(row), i] = row  # assign row of data to a column
    return arr, np.array(lengths, dtype=np.int32)


def _masked_full_like(np_data, value, num_non_padding=None, padding=0, dtype=np.float32):
    arr = np.full_like(np_data, value, dtype=dtype)
    total_non_pad = sum(num_non_padding)
    if num_non_padding is not None and total_non_pad < np_data.size:
        # is there a way to avoid this for loop?
        for i, last in enumerate(num_non_padding):
            arr[last:, i] = 0
    return arr, total_non_pad


BatchTuple = collections.namedtuple(
    'BatchTuple', ('features', 'labels', 'num_tokens'))
SeqFeatureTuple = collections.namedtuple('SeqFeatureTuple', ('inputs', 'seq_len'))
SeqLabelTuple = collections.namedtuple(
    'SeqLabelTuple', ('label', 'label_weight', 'seq_weight'))


def get_batch_iter(in_data, out_data, batch_size=1, shuffle=False):
    for x, y in _batch_iter(batch_size, shuffle, in_data, out_data, pad=[[], []]):
        x_arr, x_len = _hstack_list(x)
        y_arr, y_len = _hstack_list(y)
        seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
        token_weight, num_tokens = _masked_full_like(
            y_arr, 1, num_non_padding=y_len)
        features = SeqFeatureTuple(x_arr, x_len)
        labels = SeqLabelTuple(y_arr, token_weight, seq_weight)
        yield BatchTuple(features, labels, num_tokens)


def get_ngram_batch_iter(in_data, out_data, batch_size=1, shuffle=False):
    for x, y in _batch_iter(batch_size, shuffle, in_data, out_data, pad=[[], 0]):
        x_arr, x_len = _hstack_list(x)
        y_arr = np.array(y)
        seq_weight = np.where(x_len > 0, 1, 0).astype(np.float32)
        token_weight, num_tokens = _masked_full_like(
            x_arr, 1, num_non_padding=x_len)
        features = SeqFeatureTuple(x_arr, x_len)
        labels = SeqLabelTuple(y_arr, token_weight, seq_weight)
        yield BatchTuple(features, labels, num_tokens)


def default_reader_opt():
    return {
        'vocab_path': 'test_data/vocab.txt',
        'text_path': 'test_data/train.txt',
        'ngrams': False,
        'sentences': False,
        # if sentences is False
        'min_seq_len': 9,
        'max_seq_len': 9,
        # fi
        'shuffle': False,
        'batch_size': 3
    }


BatchIterWrapper = collections.namedtuple(
    'BatchIterWrapper', ('iter', 'vocab', 'keep_state', 'batch_size', 'num_batches'))


def get_batch_iter_from_file(opt, vocab=None):
    if vocab is None:
        vocab = Vocabulary.from_vocab_file(opt['vocab_path'])
    _text_path = opt['text_path']
    batch_fn = get_batch_iter
    if opt['ngrams']:
        in_data, out_data = read_ngram_lm(_text_path, vocab)
        batch_fn = get_ngram_batch_iter
    elif opt['sentences']:
        in_data, out_data = read_sentences(_text_path, vocab)
    else:
        in_data, out_data = read_text(
            _text_path, vocab, opt['min_seq_len'],  opt['max_seq_len'])
    num_batches = np.ceil(len(in_data) / opt['batch_size'])
    keep_state = not (opt['shuffle'] or opt['sentences'])
    batch_iter = partial(
            batch_fn,
            in_data, out_data, batch_size=opt['batch_size'], shuffle=opt['shuffle'])
    return BatchIterWrapper(
        batch_iter, vocab, keep_state, opt['batch_size'], num_batches)


class Vocabulary(object):

    special_symbols = {
        'end_seq': '</s>', 'start_seq': '<s>', 'end_encode': '</enc>',
        'unknown': '<unk>'}

    def __init__(self):
        self._w2i = {}
        self._i2w = []
        self._i2freq = {}
        self._vocab_size = 0

    def __getitem__(self, arg):
        if isinstance(arg, six.string_types):
            return self._w2i[arg]
        elif isinstance(arg, int):
            return self._i2w[arg]
        else:
            raise ValueError('Only support either integer or string')

    @property
    def vocab_size(self):
        return self._vocab_size

    def add(self, word, count):
        self._w2i[word] = self._vocab_size
        self._i2w.append(word)
        self._i2freq[self._vocab_size] = count
        self._vocab_size += 1

    def w2i(self, word, unk_id=None):
        if isinstance(word, six.string_types):
            if unk_id is None and self.special_symbols['unknown'] in self._w2i:
                unk_id = self._w2i[self.special_symbols['unknown']]
            if unk_id is not None:
                return self._w2i.get(word, unk_id)
            else:
                return self._w2i[word]
        if isinstance(word, collections.Iterable):
            return [self.w2i(_w) for _w in word]

    def i2w(self, index):
        if isinstance(index, six.string_types):
            raise ValueError(
                ('index must be an integer, recieved `{}`. '
                 'Call `w2i()` for converting word to id').format(index))
        if isinstance(index, collections.Iterable):
            return [self.i2w(_idx) for _idx in index]
        return self._i2w[index]

    def word_set(self):
        return set(self._w2i.keys())

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def from_vocab_file(filepath):
        vocab = Vocabulary()
        with open(filepath) as ifp:
            for line in ifp:
                parts = line.strip().split()
                count = 0
                word = parts[0]
                if len(parts) > 1:
                    count = int(parts[1])
                vocab.add(word, count)
        return vocab


if __name__ == '__main__':
    batchs = get_batch_iter_from_file(default_reader_opt())
    for batch in batchs.iter():
        x = batch
    print(x)
