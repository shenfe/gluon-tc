# coding: utf8

from mxnet import autograd, gluon, init, metric, nd
from mxnet.gluon import loss as gloss, nn, rnn


class MyModel(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 mode='lstm', bidirectional=True, drop_prob=0.5, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        with self.name_scope():
            self.dropout = nn.Dropout(drop_prob)
            self.embedding = nn.Embedding(vocab_size, embed_size)

            if mode == 'relu':
                self.encoder = rnn.RNN(num_hiddens,
                                       num_layers,
                                       activation='relu',
                                       dropout=drop_prob,
                                       input_size=embed_size)
            elif mode == 'tanh':
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
            else:  # lstm
                self.encoder = rnn.LSTM(num_hiddens,
                                        num_layers,
                                        dropout=drop_prob,
                                        bidirectional=bidirectional,
                                        input_size=embed_size)

            self.decoder = nn.Dense(num_outputs, flatten=False)

    def forward(self, inputs, begin_state=None):
        embeddings = self.dropout(self.embedding(inputs))
        states = self.encoder(embeddings)
        outputs = self.dropout(states[0])
        # 连结初始时间步和最终时间步的隐藏状态
        encoding = nd.concat(outputs, states[-1])
        return self.decoder(encoding)


from corpus import vocab, ctx
from embedding import embed


num_outputs = 2
lr = 1
num_epochs = 200
batch_size = 16
embed_size = embed.vec_len
num_hiddens = 1000
num_layers = 2
vocab_size = len(vocab)
mode = 'lstm'

net = MyModel(vocab_size, embed_size, num_hiddens, num_layers, mode=mode)
net.initialize(init.Xavier(), ctx=ctx)

# 设置 embedding 层的 weight 为预训练的词向量
net.embedding.weight.set_data(embed.idx_to_vec.as_in_context(ctx))

# 训练中不迭代词向量（net.embedding中的模型参数）
net.embedding.collect_params().setattr('grad_req', 'null')

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
# trainer = gluon.Trainer(net.collect_params(), 'adam')

loss = gloss.SoftmaxCrossEntropyLoss()
