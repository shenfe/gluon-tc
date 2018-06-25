# coding: utf8

import collections
from helper import reads, try_gpu
from mxnet.contrib import text
from mxnet import nd
from random import shuffle
import random


"""
1. load train and test data
"""

train_data = reads(*['dataset/shenke_comment_%s.csv' % str(label) for label in [0, 1, 2]])

test_data = reads(*['dataset/sex_comment_12_tj_cozd_%s.csv' % str(label) for label in [2, 1, 0]])

_train_size = min(100000, len(train_data))
train_data = random.sample(train_data, _train_size)
shuffle(train_data)
shuffle(test_data)

print('data shuffled')
_ratio_test_size = 1000
print('neg ratio is about %d/%d' % ([i[2] for i in train_data[:_ratio_test_size]].count('0'), _ratio_test_size))


"""
2. get the tokenized
"""

train_tokenized = [row[1] for row in train_data]

test_tokenized = [row[1] for row in test_data]


"""
3. build a vocabulary
"""

_token_counter = collections.Counter()


def _count_token(train_tokenized, token_counter):
    for sample in train_tokenized:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1


_count_token(train_tokenized, _token_counter)

vocab = text.vocab.Vocabulary(_token_counter,
                              unknown_token='<ukn>',
                              reserved_tokens=['</s>', '，', '。'])

# print('xxxx')
# print(len(vocab.idx_to_token))


"""
4. pre-process data
"""


def encode_samples(tokenized_samples, vocab, pad_size=100):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in vocab.token_to_idx:
                feature.append(vocab.token_to_idx[token])
            else:
                feature.append(0)
        while len(feature) < pad_size:
            feature.append(1)  # padded with '</s>'
        if len(feature) > pad_size:
            feature = feature[:pad_size]  # cropped
        features.append(feature)
    return features


_sample_size = 50

ctx = try_gpu()
train_features = encode_samples(train_tokenized, vocab, _sample_size)
test_features = encode_samples(test_tokenized, vocab, _sample_size)
train_features = nd.array(train_features, ctx=ctx)
test_features = nd.array(test_features, ctx=ctx)
train_labels = nd.array([(0 if str(s[2]) == '0' else 1) for s in train_data], ctx=ctx)
test_labels = nd.array([(0 if str(s[2]) == '0' else 1) for s in test_data], ctx=ctx)
print('train sample 1:')
print(train_data[0][0])
print('train sample 1 feature vector:')
print(train_features[0])
print('train sample 1 label:')
print(train_labels[0])


"""
some test
"""

if __name__ == '__main__':
    pass

