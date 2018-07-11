# coding: utf8

"""
gpu
"""
use_gpu = True
gpu_index = 3

"""
corpus
"""
seq_len = 25
train_size_limit = 100000
train = 'shenke_comment'
test = [
    'sex_comment_12_tj_cozd',
    'sex_comment_huoshan_fk'
][0]

"""
embedding
"""
unknown_token = '</s>'
pretrained_file = 'w2v/comment.vec.dat'

"""
model
"""
lr = .001
optimiz = 'adam'  # 'sgd', 'adam'
hidden_size = 80  # same as embed_dim
num_layers = 2
mode = 'lstm'
drop_prob = 0.5

"""
training
"""
num_epochs = 512
batch_size = 128


def conf(prop={}, *args):
    """
    Get or set global variables in this module, to read/write configurations
    """

    if isinstance(prop, str):
        if len(args) == 0:
            return globals().get(prop)
        globals()[prop] = args[0]
        return args[0]

    if isinstance(prop, (list, set)):
        res = []
        for p in prop:
            res.append(globals().get(p))
        return res

    if isinstance(prop, dict):
        for k, v in prop.items():
            globals()[k] = v

    return globals()
