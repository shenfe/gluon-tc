# coding: utf8

from __future__ import print_function

from mxnet import autograd, nd

from corpus import vocab, train_features, test_features, train_labels, test_labels, ctx
from model import num_epochs, batch_size, trainer, net, loss
from eval import eval_model

import time


def train():
    print('train_features.shape: %s, %s' % train_features.shape)
    for epoch in range(1, num_epochs + 1):
        time0 = int(time.time())
        batch_count = train_features.shape[0] // batch_size
        print('batch_count: %d' % batch_count)
        for i in range(batch_count):
            # if i % 100 == 0: print('+')
            X = train_features[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
            y = train_labels[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        time1 = int(time.time())
        train_loss, train_acc = eval_model(train_features, train_labels, is_train=True)
        test_loss, test_acc = eval_model(test_features, test_labels)
        print('time %ds, epoch %d, train loss %.6f, acc %.4f; test loss %.6f, acc %.4f'
              % (time1 - time0, epoch, train_loss, train_acc, test_loss, test_acc))
        net.save_params('model/net-epoch_%d-batch_size_%d.params' % (epoch, batch_size))


def test(review):
    return nd.argmax(net(nd.reshape(nd.array([vocab.token_to_idx[token] for token in review], ctx=ctx),
                                    shape=(-1, 1)
                                    )
                        ),
                     axis=1
                     ).asscalar()

if __name__ == '__main__':
    train()
    print(test(['我', '在', '换', '衣服']))
    print(test(['我', '操', '、', '无敌', '了']))
    print(test(['胸', '太', '小', '了']))
    print(test(['不让', '肏', '就', '别', '老', '瞎', '晃']))
