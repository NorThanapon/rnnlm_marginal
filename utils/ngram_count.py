import sys
import subprocess

input_file_path = sys.argv[1]
max_ngram_len = int(sys.argv[2])
min_ngram_count = int(sys.argv[3])


def SRILM_ngram_count(
        text_filepath, out_filepath, order=5, ngram_count_path='ngram-count'):
    count_filepath = out_filepath + '.count' + str(order)
    command = [ngram_count_path, '-order', str(order), '-text', text_filepath]
    subprocess.call(command + ['-write', count_filepath])
    return count_filepath


count_file_path = SRILM_ngram_count(
    input_file_path, input_file_path, order=max_ngram_len,
    ngram_count_path='ngram-count')


with open(count_file_path) as i:
    with open(f'{count_file_path}-filter', 'w') as o1:
        for line in i:
            line = line.replace('<s>', '</s>')
            ngram, count = line.strip().split('\t')
            if len(ngram.split()) > max_ngram_len:
                continue
            if int(count) < min_ngram_count:
                continue
            o1.write(line)
