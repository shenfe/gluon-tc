# coding: utf8

from mxnet.contrib import text

from corpus import vocab


_pretrained_file_path = 'w2v/comment.vec.dat'
_elem_delim = ' '

embed = text.embedding.CustomEmbedding(
    pretrained_file_path=_pretrained_file_path,
    elem_delim=_elem_delim,
    vocabulary=vocab
)


"""
some test
"""

if __name__ == '__main__':
    assert len(embed) == 389419
    assert embed.vec_len == 80
    assert embed.token_to_idx[u'美好'] == 1958
    assert embed.token_to_idx['美好'] == 1958
    assert embed.idx_to_token[1958] == u'美好'
    print(embed.idx_to_vec[1958])
