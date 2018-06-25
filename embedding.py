# coding: utf8

import mxnet as mx
from mxnet.contrib import text

from corpus import vocab


_pretrained_file_path = 'w2v/comment.vec.dat'
_elem_delim = ' '

embed = text.embedding.CustomEmbedding(
    pretrained_file_path=_pretrained_file_path,
    elem_delim=_elem_delim,
    vocabulary=vocab
    # vocabulary=None
)


"""
some test
"""

if __name__ == '__main__':
    print('embed size: %d' % len(embed))


    def vec_of_token(t):
        return embed.idx_to_vec[embed.token_to_idx[t]]
