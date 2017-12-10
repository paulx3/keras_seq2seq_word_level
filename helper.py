'''
@author: wanzeyu

@contact: wan.zeyu@outlook.com

@file: helper.py

@time: 2017/12/4 2:19
'''

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from numpy import array

np.set_printoptions(threshold=np.nan)
padding_len = 20

UNK_TOKEN = 2
END_TOKEN = 1
START_TOKEN = 0


def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab


vocab = load_vocab("train_vocab.txt")
id_vocab = {value: key for key, value in vocab.items()}


def tokenize_and_map(line):
    return [0] + [vocab.get(token, UNK_TOKEN) for token in line.split(' ')]


def get_data_v2(file_name):
    res = []
    with open(file_name) as fp:
        for in_line in fp:
            tmp = tokenize_and_map(in_line)[:(padding_len - 1)] + [END_TOKEN]
            res.append(tmp)
    return res


def get_data_v2_offset(file_name):
    res = []
    with open(file_name) as fp:
        for in_line in fp:
            tmp = tokenize_and_map(in_line)[:(padding_len - 1)] + [END_TOKEN]
            tmp = tmp[1:]
            tmp = to_categorical(array(tmp), num_classes=3244)
            tmp = pad_sequences(tmp, padding="post", value=1.0, maxlen=3244)
            res.append(tmp)
    return pad_sequences(res, padding="post", maxlen=30)
