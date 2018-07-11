# coding: utf8

from mxnet import nd

from corpus import Corpus
from embedding import Embedding


def predict(net, conf, tokenized_samples):
    f = Corpus.embed(tokenized_samples, Embedding(conf).vocab, conf.get('unknown_token'),
                     pad_size=conf.get('seq_len'))
    return net(nd.array(f).T)
