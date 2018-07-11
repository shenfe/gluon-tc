# coding: utf8

import os
import sys
import time
import csv
py3 = False
if sys.version_info > (3, 0):
    py3 = True
    from functools import reduce

import mxnet as mx
from mxnet import nd

script_dir = os.path.dirname(__file__)  # absolute dir this script is in


def abs_file_path(location):
    return os.path.join(script_dir, location)


def nows():
    """
    Seconds now
    """
    return str(time.time()).split('.')[0]


def read(path):
    data = []
    with open(path, 'r') if not py3 else open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([row['text'], row['data'].split(' '), row['label']])
    return data


def reads(*paths):
    return reduce(lambda x, y: x + y, list(map(read, paths)))


def try_gpu(gpu=True, index=0):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""

    if not gpu:
        print('using cpu (by force)')
        return mx.cpu()

    try:
        ctx = mx.gpu(index)
        _ = nd.array([0], ctx=ctx)
        print('using gpu')
    except:
        ctx = mx.cpu()
        print('using cpu')
    return ctx


def count_col(location):
    with open(location) as f:
        f.readline()
        r = {}
        for i in range(20):
            line = f.readline()
            if not line:
                break
            n = len(line.strip().split())
            r[n] = r.get(n, 0) + 1

        max_n = 0
        for n, count in r.items():
            if count > r.get(max_n, 0):
                max_n = n

        return max_n


if __name__ == '__main__':
    # cor = read('dataset/test.csv')
    cor = reads('dataset/test.csv', 'dataset/test.csv')
    assert cor[0][0] == '我要'
    assert cor[0][1] == ['我', '要']
    assert cor[0][2] == '1'
