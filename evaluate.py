# coding: utf8

from collections import defaultdict
from mxnet import metric, nd
from mxnet.gluon import loss as gloss


loss = gloss.SoftmaxCrossEntropyLoss()


def eval_model(features, labels, net, batch_size):
    l_sum = 0
    l_n = 0
    accuracy = metric.Accuracy()
    batch_count = features.shape[0] // batch_size
    preds_all = None
    labels_all = None

    for i in range(batch_count):
        X = features[i*batch_size : (i+1)*batch_size].as_in_context(features.context).T  # batch_size * embed_size
        y = labels[i*batch_size : (i+1)*batch_size].as_in_context(labels.context).T  # batch_size * 1
        output = net(X)
        l = loss(output, y)
        l_sum += l.sum().asscalar()
        l_n += l.size

        preds = nd.argmax(output, axis=1)
        accuracy.update(preds=preds, labels=y)

        if preds_all is None:
            preds_all = preds
        preds_all = nd.concat(preds_all, preds, dim=0)
        if labels_all is None:
            labels_all = y
        labels_all = nd.concat(labels_all, y, dim=0)

    # tp = nd.sum((preds_all == 1) * (labels_all == 1)).asscalar()
    # fp = nd.sum((preds_all == 1) * (labels_all == 0)).asscalar()
    # fn = nd.sum((preds_all == 0) * (labels_all == 1)).asscalar()
    # precision = float(tp) / (tp + fp)
    # recall = float(tp) / (tp + fn)
    # f1 = 2 * (precision * recall) / (precision + recall)

    return l_sum / l_n, accuracy.get()[1], evaluate(preds_all, labels_all)


def accuracy(predictions, labels):
    true_count = 0
    for i in xrange(len(predictions)):
        if predictions[i] == labels[i]:
            true_count += 1
    return float(true_count) / len(predictions)


def _evaluate(predict_set, label_set):
    if len(predict_set) == 0:
        return 1, 0, 0
    if len(label_set) == 0:
        return 0, 1, 0

    true_count = len(label_set & predict_set)

    precision = true_count * 1.0 / len(predict_set)
    recall = true_count * 1.0 / len(label_set)

    if precision + recall == 0:
        return precision, recall, 0

    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate(preds, labels):
    predict_list = preds.asnumpy().tolist()
    label_list = labels.asnumpy().tolist()

    cat_label = defaultdict(set)
    cat_pred = defaultdict(set)

    for i, (pr, la) in enumerate(zip(predict_list, label_list)):
        cat_pred[int(pr)].add(i)
        cat_label[int(la)].add(i)

    return { cat: _evaluate(cat_pred[cat], res) for cat, res in cat_label.items()}

