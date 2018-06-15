# coding: utf-8

from mxnet import autograd, gluon, init, metric, nd
from mxnet.gluon import loss as gloss, nn, rnn


class MyModel(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,
                 bidirectional, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(len(vocab), embed_size)
            self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    input_size=embed_size)
            self.decoder = nn.Dense(num_outputs, flatten=False)

    def forward(self, inputs, begin_state=None):
        embeddings = self.embedding(inputs)
        states = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态。
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs


num_outputs = 2
lr = 0.1
num_epochs = 20
batch_size = 10
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True

from corpus import vocab, ctx
from embedding import embed

embed_size = embed.vec_len

net = MyModel(vocab, embed_size, num_hiddens, num_layers, bidirectional)
net.initialize(init.Xavier(), ctx=ctx)

# 设置 embedding 层的 weight 为预训练的词向量
net.embedding.weight.set_data(embed.idx_to_vec.as_in_context(ctx))

# 训练中不迭代词向量（net.embedding中的模型参数）
net.embedding.collect_params().setattr('grad_req', 'null')

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

loss = gloss.SoftmaxCrossEntropyLoss()
