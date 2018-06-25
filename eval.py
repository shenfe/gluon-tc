# coding: utf8

from mxnet import metric, nd

from corpus import ctx
from model import batch_size, net, loss


def eval_model(features, labels, is_train=False):
    l_sum = 0
    l_n = 0
    accuracy = metric.Accuracy()
    batch_count = features.shape[0] // batch_size
    for i in range(batch_count):
        X = features[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T  # batch_size * embed_size
        y = labels[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T  # batch_size * 1
        output = net(X)
        # if i % 100 == 0 and is_train: print(X, output, y)
        l = loss(output, y)
        l_sum += l.sum().asscalar()
        l_n += l.size
        accuracy.update(preds=nd.argmax(output, axis=1), labels=y)
    return l_sum / l_n, accuracy.get()[1]

