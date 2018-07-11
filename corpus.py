# coding: utf8

import collections
from mxnet.contrib import text
from mxnet import nd
from random import shuffle
import random

from helper import reads, try_gpu


class Corpus:
    def __init__(self, conf={}):
        """Get corpus data ready"""

        """
        Get configurations
        """
        train_data          = conf.get('train_data')
        test_data           = conf.get('test_data')
        train_prefix        = conf.get('train')
        test_prefix         = conf.get('test')
        train_size_limit    = conf.get('train_size_limit')
        unknown_token       = conf.get('unknown_token')
        use_gpu             = conf.get('use_gpu')
        gpu_index           = conf.get('gpu_index')

        """
        1. load train and test data
        """

        train_data = train_data or reads(*['dataset/%s_%s.csv' % (train_prefix, str(label)) for label in [0, 1, 2]])
        test_data = test_data or reads(*['dataset/%s_%s.csv' % (test_prefix, str(label)) for label in [2, 1, 0]])

        _train_size = min(train_size_limit, len(train_data))
        train_data = random.sample(train_data, _train_size)
        shuffle(train_data)
        shuffle(test_data)

        print('data shuffled')
        _ratio_test_size = 1000
        print('neg ratio is about %d/%d' % ([i[2] for i in train_data[:_ratio_test_size]].count('0'), _ratio_test_size))

        print('train sample 1:')
        print(train_data[0][0])

        """
        2. get the tokenized
        """

        train_tokenized = [row[1] for row in train_data]
        test_tokenized = [row[1] for row in test_data]

        print('train sample 1 tokenized:')
        print(train_tokenized[0])

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
        traindata_vocab = text.vocab.Vocabulary(_token_counter,
                                                unknown_token=unknown_token,
                                                # reserved_tokens=['</s>', '，', '。']
                                                )

        print('train data vocab:')
        print(traindata_vocab.idx_to_token[:20])
        print('train data vocab size: %d' % len(traindata_vocab.idx_to_token))


        """
        4. extract labels
        """

        ctx = try_gpu(use_gpu, index=gpu_index)
        train_labels = nd.array([(0 if str(s[2]) == '0' else 1) for s in train_data], ctx=ctx)
        test_labels = nd.array([(0 if str(s[2]) == '0' else 1) for s in test_data], ctx=ctx)

        print('train sample 1 label:')
        print(train_labels[0])


        self.unknown_token = unknown_token

        self.train_data = train_data
        self.test_data = test_data
        self.train_tokenized = train_tokenized
        self.test_tokenized = test_tokenized
        self.traindata_vocab = traindata_vocab

        self.ctx = ctx
        self.train_labels = train_labels
        self.test_labels = test_labels


    @staticmethod
    def embed(tokenized_samples, vocab, unknown_token, pad_size=100, replace_with_ukn=False):
        features = []
        for sample in tokenized_samples:
            feature = []
            for token in sample:
                token = token.decode('utf8')
                if token in vocab.token_to_idx:
                    feature.append(vocab.token_to_idx[token])
                else:
                    if replace_with_ukn:
                        feature.append(vocab.token_to_idx[unknown_token])
            while len(feature) < pad_size:
                feature.append(vocab.token_to_idx[unknown_token])  # padded
            if len(feature) > pad_size:
                feature = feature[:pad_size]  # cropped
            features.append(feature)
        return features


    def embed_features(self, vocab, unknown_token=None, seq_len=25, replace_with_ukn=False):
        """
        Feature embeddings
        """
        unknown_token = unknown_token or self.unknown_token
        train_features = self.embed(self.train_tokenized, vocab, unknown_token, seq_len, replace_with_ukn)
        self.train_features = nd.array(train_features, ctx=self.ctx)
        test_features = self.embed(self.test_tokenized, vocab, unknown_token, seq_len, replace_with_ukn)
        self.test_features = nd.array(test_features, ctx=self.ctx)

        print('train sample 1 features:')
        print(self.train_features[0])

