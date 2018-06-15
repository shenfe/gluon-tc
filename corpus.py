# coding: utf-8

import collections
from helper import reads, try_gpu
from mxnet.contrib import text
from mxnet import nd


"""
1. load train and test data
"""

train_data = reads(*['dataset/sex_comment_12_tj_cozd_%s.csv' % str(label) for label in [2, 1, 0]])

test_data = reads(*['dataset/sex_comment_huoshan_fk_%s.csv' % str(label) for label in [0, 1, 2]])


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
                              unknown_token='</s>',
                              reserved_tokens=None)

# print('xxxx')
# print(len(vocab.idx_to_token))


"""
4. pre-process data
"""


def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in vocab.token_to_idx:
                feature.append(vocab.token_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=100, padding=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            # 添加 PAD 符号使每个序列等长（长度为 maxlen ）
            while len(padded_feature) < maxlen:
                padded_feature.append(padding)
        padded_features.append(padded_feature)
    return padded_features


ctx = try_gpu()
train_features = encode_samples(train_tokenized, vocab)
# print(train_features[0])
test_features = encode_samples(test_tokenized, vocab)
train_features = nd.array(pad_samples(train_features, 50, 0), ctx=ctx)
# print(train_features[0])
test_features = nd.array(pad_samples(test_features, 50, 0), ctx=ctx)
train_labels = nd.array([(0 if s[2] == '0' else '1') for s in train_data], ctx=ctx)
test_labels = nd.array([(0 if s[2] == '0' else '1') for s in test_data], ctx=ctx)


"""
some test
"""

if __name__ == '__main__':
    assert train_data[0][1][0] == '2018年'
    assert train_data[0][2] == '0'
    assert test_data[0][1][0] == '你'
    assert test_data[0][2] == '2'

