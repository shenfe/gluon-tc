# coding: utf8

from mxnet import metric, nd

from corpus import ctx
from model import batch_size, net, loss


def eval_model(features, labels, is_train=False):
    l_sum = 0
    l_n = 0
    accuracy = metric.Accuracy()
    for i in range(features.shape[0] // batch_size):
        X = features[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T  # batch_size * embed_size
        y = labels[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T  # batch_size * 1
        output = net(X)
        # print('X[0]', X[0])
        l = loss(output, y)
        l_sum += l.sum().asscalar()
        l_n += l.size
        # if is_train:
        #     print('output', output)
        accuracy.update(preds=nd.argmax(output, axis=1), labels=y)
    return l_sum / l_n, accuracy.get()[1]

