# coding: utf8

from mxnet import autograd, gluon, init, metric, nd
from mxnet.gluon import nn, rnn

from helper import try_gpu
from config import conf
from embedding import Embedding
from predict import predict


class Model(nn.Block):

    model_type = 'rnn'

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 num_outputs=2, mode='lstm', bidirectional=True, drop_prob=0.5, **kwargs):
        if not drop_prob or drop_prob <= 0:
            drop_prob = 0.0
        if not num_hiddens or num_hiddens <= 0:
            num_hiddens = embed_size

        super(Model, self).__init__(**kwargs)

        with self.name_scope():
            self.dropout = nn.Dropout(drop_prob) if drop_prob else None

            self.embedding = nn.Embedding(vocab_size, embed_size)

            if mode == 'rnn-relu':
                self.encoder = rnn.RNN(num_hiddens,
                                       num_layers,
                                       activation='relu',
                                       dropout=drop_prob,
                                       input_size=embed_size)
            elif mode == 'rnn-tanh':
                self.encoder = rnn.RNN(num_hiddens,
                                       num_layers,
                                       activation='tanh',
                                       dropout=drop_prob,
                                       input_size=embed_size)
            elif mode == 'gru':
                self.encoder = rnn.GRU(num_hiddens,
                                       num_layers,
                                       dropout=drop_prob,
                                       input_size=embed_size)
            else:
                self.encoder = rnn.LSTM(num_hiddens,
                                        num_layers,
                                        dropout=drop_prob,
                                        bidirectional=bidirectional,
                                        input_size=embed_size)

            self.decoder = nn.Dense(num_outputs, flatten=False)


    def forward(self, inputs, begin_state=None):
        embeddings = self.embedding(inputs)
        if self.dropout:
            embeddings = self.dropout(embeddings)

        states = self.encoder(embeddings)
        outputs = states[0]
        if self.dropout:
            outputs = self.dropout(outputs)

        # 连结初始时间步和最终时间步的隐藏状态
        encoding = nd.concat(outputs, states[-1])
        return self.decoder(encoding)


def create_model(config, embedding=None, ctx=None, net=None, params_file=None):

    """
    Get model configurations
    """
    num_hiddens = config.get('hidden_size')
    num_layers  = config.get('num_layers')
    mode        = config.get('mode')
    drop_prob   = config.get('drop_prob')

    """
    Get the embedding
    """
    if not embedding:
        embedding = config.get('pretrained_file')
    if isinstance(embedding, tuple):
        embed, embed_size, vocab_size = embedding
    else:
        if isinstance(embedding, str):
            embed = Embedding(embedding)
        else:
            embed = embedding
        embed_size = embed.vec_len
        vocab_size = len(embed)

    """
    Get the context
    """
    if not ctx:
        ctx = try_gpu(gpu=config.get('use_gpu', True), index=config.get('gpu_index', 0))

    """
    Get the net
    """
    net = net or Model(vocab_size, embed_size, num_hiddens, num_layers,
                       num_outputs=2, mode=mode, drop_prob=drop_prob)
    net.initialize(init.Xavier(), ctx=ctx)

    # 设置 embedding 层的 weight 为预训练的词向量
    net.embedding.weight.set_data(embed.idx_to_vec.as_in_context(ctx))

    # 训练中不迭代词向量（net.embedding中的模型参数）
    net.embedding.collect_params().setattr('grad_req', 'null')

    if params_file:
        net.load_params(params_file, ctx)

    return net


def create_trainer(net, config={}):
    optimiz = config.get('optimiz', 'adam')
    lr = config.get('lr', .001)
    trainer = gluon.Trainer(net.collect_params(), optimiz, {'learning_rate': lr})
    return trainer


class Scorer:
    def __init__(self, seq_len, vec_dim, location_vec, location_model, location_symbol, batch_size,
                 is_multilabel=False):
        self.conf = {}
        self.conf.update(conf())
        self.conf.update({
            'batch_size': batch_size,
            'seq_len': seq_len
        })
        self.net = create_model(conf, embedding=location_vec, params_file=location_model)

    def process(self, tokenized_samples, max_cnt=-1):
        if len(tokenized_samples) > max_cnt > 0:
            tokenized_samples = tokenized_samples[:max_cnt]
        return predict(self.net, self.conf, tokenized_samples)


class Highlighter:
    def __init__(self, seq_len, vec_dim, location_vec, location_model, location_symbol, batch_size,
                 is_multilabel=False):
        self.conf = {}
        self.conf.update(conf())
        self.conf.update({
            'batch_size': batch_size,
            'seq_len': seq_len
        })
        self.net = create_model(conf, embedding=location_vec, params_file=location_model)

    def process(self, tokenized_samples, range=(-255, 255), max_cnt=-1):
        pass
