# coding: utf8

import csv
import sys
py3 = False
if sys.version_info > (3, 0):
    py3 = True
    from functools import reduce

import mxnet as mx
from mxnet import nd


def read(path):
    data = []
    with open(path, 'r') if not py3 else open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([row['text'], row['data'].split(' '), row['label']])
    return data


def reads(*paths):
    return reduce(lambda x, y: x + y, list(map(read, paths)))


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
        print('using gpu')
    except:
        ctx = mx.cpu()
        print('using cpu')
    return ctx


if __name__ == '__main__':
    # cor = read('dataset/test.csv')
    cor = reads('dataset/test.csv', 'dataset/test.csv')
    assert cor[0][0] == '我要'
    assert cor[0][1] == ['我', '要']
    assert cor[0][2] == '1'
