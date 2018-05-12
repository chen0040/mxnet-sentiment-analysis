import mxnet as mx
from mxnet import ndarray as nd


def pad_sequences(x, max_seq_length, ctx=mx.cpu()):
    row_count = len(x)
    result = nd.zeros(shape=(row_count, max_seq_length), ctx=ctx)
    for i, row in enumerate(x):
        if len(row) >= max_seq_length:
            for j, w in enumerate(row[:max_seq_length]):
                result[i, j] = w
        else:
            for j, w in enumerate(row):
                result[i, max_seq_length - len(row) + j] = w
    return result
