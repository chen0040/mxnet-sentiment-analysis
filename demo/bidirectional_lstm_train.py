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

    output_dir_path = os.path.join(os.path.dirname(__file__), 'models')
    data_file_path = patch_path('data/umich-sentiment-train.txt')

    from mxnet_sentiment.library.lstm import SentimentAnalyserWithSoftMaxBidrectionalLSTM
    from mxnet_sentiment.utility.simple_data_loader import load_text_label_pairs
    from mxnet_sentiment.utility.text_utils import fit_text

    text_data_model = fit_text(data_file_path)
    text_label_pairs = load_text_label_pairs(data_file_path)

    train_data, validation_data = train_test_split(text_label_pairs, test_size=0.3, random_state=42)

    rnn = SentimentAnalyserWithSoftMaxBidrectionalLSTM(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))
    history = rnn.fit(text_data_model, text_label_pairs=train_data, model_dir_path=output_dir_path,
                      checkpoint_interval=10, batch_size=64, epochs=20, test_text_label_pairs=validation_data)


if __name__ == '__main__':
    main()
