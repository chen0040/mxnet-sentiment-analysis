from __future__ import print_function
import mxnet as mx
import os
import sys

from sklearn.model_selection import train_test_split


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def main():
    sys.path.append(patch_path('..'))

    mx.random.seed(1)

    model_dir_path = os.path.join(os.path.dirname(__file__), 'models')
    data_file_path = patch_path('data/umich-sentiment-train.txt')

    from mxnet_sentiment.library.lstm import SentimentAnalyserWithSoftMaxBidrectionalLSTM
    from mxnet_sentiment.utility.simple_data_loader import load_text_label_pairs

    text_label_pairs = load_text_label_pairs(data_file_path)

    train_data, validation_data = train_test_split(text_label_pairs, test_size=0.3, random_state=42)

    rnn = SentimentAnalyserWithSoftMaxBidrectionalLSTM(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))
    rnn.load_model(model_dir_path)

    for text, label in validation_data:
        predicted_label = rnn.predict_class(text)
        print('predicted: ', predicted_label, 'actual: ', label)


if __name__ == '__main__':
    main()
