# coding: utf8

import os
import time
import logging
from mxnet import autograd, nd, metric

from helper import abs_file_path, nows
from config import conf as _conf
from evaluate import eval_model, loss

from corpus import Corpus
from embedding import Embedding
from model import Model, Trainer

from predict import predict


eval_train_interval = 1
eval_period = False
eval_period_interval = 10


class Train:

    save_params = True

    def __init__(self, conf={}, log_handler=None, model_dir=None):
        """Train a model, and evaluate it"""

        """
        Ensure the workspace, to save model parameters there
        """
        if not model_dir:
            model_dir = abs_file_path('models/%s' % nows())
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        """
        Get configurations
        """
        __conf = {}
        __conf.update(_conf())
        __conf.update(conf)
        conf = __conf
        batch_size = conf.get('batch_size')
        num_epochs = conf.get('num_epochs')

        """
        Get the corpus data
        """
        _corpus = Corpus(conf)
        _embedding = Embedding({
            'pretrained_file': conf.get('pretrained_file'),
            'unknown_token': conf.get('unknown_token')
        })
        _corpus.embed_features(_embedding.vocab,
                               unknown_token=conf.get('unknown_token'),
                               seq_len=conf.get('seq_len'))

        ctx = _corpus.ctx
        train_labels = _corpus.train_labels
        test_labels = _corpus.test_labels
        train_features = _corpus.train_features
        test_features = _corpus.test_features
        train_size = train_features.shape[0]
        print('train_features.shape: %s, %s' % train_features.shape)

        """
        Get the checkpoint
        """
        model_type = conf.get('model_type', 'cnn')
        checkpoint = conf.get('checkpoint')
        if checkpoint is not None:
            if not isinstance(checkpoint, int):
                checkpoint = int(checkpoint)
            checkpoint = os.path.join(model_dir, '%s-%04d.params' % (model_type, checkpoint))
            if not os.path.exists(checkpoint):
                logging.error('Invalid checkpoint: %s' % checkpoint)
                return

        """
        Initialize the net, and create a trainer
        """
        net = Model(conf, (_embedding.embed, _embedding.embed_size, _embedding.vocab_size), ctx, params_file=checkpoint)
        trainer = Trainer(net, conf)

        results = []

        batch_count = train_size // batch_size
        print('batch_count: %d' % batch_count)

        """
        Iterate
        """
        for epoch in range(1, num_epochs + 1):

            result = {'epoch': epoch}

            if eval_period:
                l_sum = 0
                l_n = 0
                accuracy = metric.Accuracy()

            time0 = int(time.time())

            """
            Train batch by batch
            """
            for i in range(batch_count):
                X = train_features[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
                y = train_labels[i*batch_size : (i+1)*batch_size].as_in_context(ctx).T
                output = None
                with autograd.record():
                    output = net(X)
                    l = loss(output, y)
                l.backward()
                trainer.step(batch_size)

                if eval_period:
                    l_sum += l.sum().asscalar()
                    l_n += l.size
                    accuracy.update(preds=nd.argmax(output, axis=1), labels=y)
                    if i % eval_period_interval == 0 and i > 0:
                        print('epoch %d, batch %d; train loss %.6f, acc %.4f' % (epoch, i, l_sum / l_n, accuracy.get()[1]))
                        l_sum = 0
                        l_n = 0
                        accuracy = metric.Accuracy()

            """
            Calculate the training time
            """
            time1 = int(time.time())
            print('epoch %d, time %ds:' % (epoch, time1 - time0))
            result['time'] = time1 - time0

            """
            Evaluate the model upon the testing dataset
            """
            time0 = int(time.time())
            test_loss, test_acc, test_prf = eval_model(test_features, test_labels, net, batch_size)
            time1 = int(time.time())
            print('    [test] loss %.6f, acc %.4f, time %ds' % (test_loss, test_acc, time1 - time0))
            result['test'] = {
                'loss': test_loss,
                'acc': test_acc,
                'time': time1 - time0,
                'prf': test_prf
            }

            if epoch % eval_train_interval == 0:
                """
                Evaluate the model upon the training dataset
                """
                time0 = int(time.time())
                train_loss, train_acc, train_prf = eval_model(train_features, train_labels, net, batch_size)
                time1 = int(time.time())
                print('    [train] loss %.6f, acc %.4f, time %ds' % (train_loss, train_acc, time1 - time0))
                result['train'] = {
                    'loss': train_loss,
                    'acc': train_acc,
                    'time': time1 - time0,
                    'prf': train_prf
                }

            if self.save_params:
                net.save_params(os.path.join(model_dir, '%s-%04d.params' % (net.model_type, epoch)))

            results.append(result)

            if log_handler:
                log_handler(result)

        self.conf = conf
        self.corpus = _corpus
        self.embedding = _embedding
        self.ctx = ctx
        self.net = net
        self.model_dir = model_dir


    def embed(self, *tokenized_samples):
        return self.corpus.embed(tokenized_samples, self.embedding.vocab, self.conf.get('unknown_token'),
                                 pad_size=self.conf.get('seq_len'))


if __name__ == '__main__':
    conf = _conf()
    t = Train(conf)
    net = t.net
    print(predict(net, conf, [
        ['就', '喜欢', '这种', '含骚待草', '的', '女人']
    ]))

