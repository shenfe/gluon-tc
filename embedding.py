# coding: utf8

import os
import io
import mxnet as mx
from mxnet import nd
from mxnet.contrib import text

from helper import abs_file_path, count_col


class Embedding:
    def __init__(self, conf={}):
        """Get embeddings ready.

        Args:
            conf: dict or str, required, specifying `pretrained_file` and `unknown_token`
        """

        """
        Get configurations
        """
        if isinstance(conf, str):
            conf = {
                'pretrained_file': conf,
                'unknown_token': '</s>'
            }
        pretrained_file_path = abs_file_path(conf.get('pretrained_file'))
        unknown_token = conf.get('unknown_token', '</s>')

        embed = _Embedding(pretrained_file_path, unknown_token)

        self.embed = embed
        self.vec_len = embed.vec_len
        self.embed_dim = embed.vec_len
        self.embed_size = embed.vec_len
        self.vocab = self.embed
        self.vocab_size = len(embed._idx_to_vec)

    def __len__(self):
        return self.vocab_size


class _Embedding:
    def __init__(self, pretrained_file_path, unknown_token):
        self._idx_to_token = []
        self._token_to_idx = {}
        self._load_embedding(pretrained_file_path, unknown_token)

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_vec(self):
        return self._idx_to_vec

    def _load_embedding(self, pretrained_file_path, unknown_token):

        pretrained_file_path = os.path.expanduser(pretrained_file_path)

        if not os.path.isfile(pretrained_file_path):
            raise ValueError('`pretrained_file_path` must be a valid path to '
                             'the pre-trained token embedding file.')

        vec_len = count_col(pretrained_file_path) - 1
        print('embedding vec_len: %d' % vec_len)


        def split_line(line):
            s = line.strip().split()
            if len(s) - 1 < vec_len:
                return None, None
            v = s[len(s) - vec_len:]
            elems = [float(i) for i in v]
            p = line.find(v[0])
            if p < 1:
                return None, None
            token = line[:p].strip()
            return token, elems


        all_elems = []
        all_elems.extend([0] * vec_len)
        tokens = set()
        loaded_unknown_vec = None
        line_num = 0

        self._idx_to_token.append(unknown_token)
        self._token_to_idx[unknown_token] = 0

        with io.open(pretrained_file_path, 'r', encoding='utf8') as f:
            for line in f:
                line_num += 1

                token, elems = split_line(line)

                if token is None or (token in tokens):
                    continue

                tokens.add(token)

                if token == unknown_token:
                    if loaded_unknown_vec is None:
                        loaded_unknown_vec = elems
                else:
                    all_elems.extend(elems)
                    self._idx_to_token.append(token)
                    self._token_to_idx[token] = len(self._idx_to_token) - 1

        self.vec_len = vec_len
        self._idx_to_vec = nd.array(all_elems).reshape((-1, self.vec_len))

        if loaded_unknown_vec is None:
            self._idx_to_vec[0] = nd.zeros(shape=self.vec_len)
        else:
            self._idx_to_vec[0] = nd.array(loaded_unknown_vec)
