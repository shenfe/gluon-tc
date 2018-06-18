# coding: utf8

from mxnet import autograd, nd

from corpus import vocab, train_features, test_features, train_labels, test_labels, ctx
from model import num_epochs, batch_size, trainer, net, loss
from eval import eval_model


def train():
    print('train_features.shape: %s, %s' % train_features.shape)
    for epoch in range(1, num_epochs + 1):
        for i in range(train_features.shape[0] // batch_size):
            # print('epoch %d, i %d' % (epoch, i))
            X = train_features[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
            y = train_labels[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_loss, train_acc = eval_model(train_features, train_labels, is_train=True)
        test_loss, test_acc = eval_model(test_features, test_labels)
        print('epoch %d, train loss %.6f, acc %.4f; test loss %.6f, acc %.4f'
              % (epoch, train_loss, train_acc, test_loss, test_acc))


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
    print(test(['不让', '肏', '就', '别', '老', '瞎晃']))
