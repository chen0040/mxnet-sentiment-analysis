import mxnet as mx
from mxnet.gluon.nn import Embedding, Sequential, Dropout, Dense
from mxnet.gluon.rnn import LSTM

from mxnet_sentiment.library.base import MultiClassTextClassifier


class SentimentAnalyserWithSoftMaxLSTM(MultiClassTextClassifier):
    model_name = 'lstm-softmax'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        super(SentimentAnalyserWithSoftMaxLSTM, self).__init__(model_ctx, data_ctx)
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.sequence_length = None
        self.config = None
        self.vocab_size = None
        self.labels = None
        self.data_ctx = data_ctx
        self.model_ctx = model_ctx

    def get_params_file_path(self, model_dir_path) -> str:
        return model_dir_path + '/' + SentimentAnalyserWithSoftMaxLSTM.model_name + '-net.params'

    def get_config_file_path(self, model_dir_path) -> str:
        return model_dir_path + '/' + SentimentAnalyserWithSoftMaxLSTM.model_name + '_config.npy'

    def get_history_file_path(self, model_dir_path) -> str:
        return model_dir_path + '/' + SentimentAnalyserWithSoftMaxLSTM.model_name + '-history.npy'

    def create_model(self) -> Sequential:
        embedding_size = 100
        model = Sequential()
        with model.name_scope():
            # input shape is (batch_size,), output shape is (batch_size, embedding_size)
            model.add(Embedding(input_dim=self.vocab_size, output_dim=embedding_size))
            model.add(Dropout(0.2))
            # layout : str, default 'TNC'
            # The format of input and output tensors.
            # T, N and C stand for sequence length, batch size, and feature dimensions respectively.
            # Change it to NTC so that the input shape can be (batch_size, sequence_length, embedding_size)
            model.add(LSTM(hidden_size=64, layout='NTC'))
            model.add(Dense(len(self.labels)))

        return model


class SentimentAnalyserWithSoftMaxBidrectionalLSTM(MultiClassTextClassifier):
    model_name = 'bi-directional-lstm-softmax'

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        super(SentimentAnalyserWithSoftMaxBidrectionalLSTM, self).__init__(model_ctx, data_ctx)
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.sequence_length = None
        self.config = None
        self.vocab_size = None
        self.labels = None
        self.data_ctx = data_ctx
        self.model_ctx = model_ctx

    def get_params_file_path(self, model_dir_path) -> str:
        return model_dir_path + '/' + SentimentAnalyserWithSoftMaxBidrectionalLSTM.model_name + '-net.params'

    def get_config_file_path(self, model_dir_path) -> str:
        return model_dir_path + '/' + SentimentAnalyserWithSoftMaxBidrectionalLSTM.model_name + '_config.npy'

    def get_history_file_path(self, model_dir_path) -> str:
        return model_dir_path + '/' + SentimentAnalyserWithSoftMaxBidrectionalLSTM.model_name + '-history.npy'

    def create_model(self) -> Sequential:
        embedding_size = 100
        model = Sequential()
        with model.name_scope():
            # input shape is (batch_size,), output shape is (batch_size, embedding_size)
            model.add(Embedding(input_dim=self.vocab_size, output_dim=embedding_size))
            model.add(Dropout(0.2))
            # layout : str, default 'TNC'
            # The format of input and output tensors.
            # T, N and C stand for sequence length, batch size, and feature dimensions respectively.
            # Change it to NTC so that the input shape can be (batch_size, sequence_length, embedding_size)
            model.add(LSTM(hidden_size=64, layout='NTC', bidirectional=True))
            model.add(Dense(len(self.labels)))

        return model

