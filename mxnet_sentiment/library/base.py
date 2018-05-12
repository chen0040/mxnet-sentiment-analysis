from abc import abstractmethod, ABC

import mxnet as mx
from mxnet import autograd, nd, gluon
import numpy as np
from mxnet_sentiment.utility.sequences import pad_sequences
from mxnet_sentiment.utility.tokenizer_utils import word_tokenize
from random import shuffle


class MultiClassTextClassifier(ABC):

    def __init__(self, model_ctx=mx.cpu(), data_ctx=mx.cpu()):
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.sequence_length = None
        self.config = None
        self.vocab_size = None
        self.labels = None
        self.data_ctx = data_ctx
        self.model_ctx = model_ctx

    @abstractmethod
    def get_params_file_path(self, model_dir_path) -> str:
        pass

    @abstractmethod
    def get_config_file_path(self, model_dir_path) -> str:
        pass

    @abstractmethod
    def get_history_file_path(self, model_dir_path) -> str:
        pass

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()

        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.sequence_length = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']

        self.model = self.create_model()
        self.model.load_params(self.get_params_file_path(model_dir_path))

    @abstractmethod
    def create_model(self) -> gluon.nn.Sequential:
        pass

    def checkpoint(self, model_dir_path):
        self.model.save_params(self.get_params_file_path(model_dir_path))

    def evaluate_accuracy(self, text_label_pairs, batch_size=64):
        X, Y = self.encode_text(text_label_pairs, batch_size)
        softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()

        metric = mx.metric.Accuracy()
        loss_avg = 0.
        for i, (data, label) in enumerate(zip(X, Y)):
            data = data.as_in_context(self.model_ctx)
            label = label.as_in_context(self.model_ctx)
            predictions = self.model(data)
            loss = softmax_loss(predictions, label)
            metric.update(preds=predictions, labels=label)
            loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)
        return metric.get()[1], loss_avg

    def encode_text(self, text_label_pairs, batch_size=64):
        num_samples = len(text_label_pairs)
        num_batches = num_samples // batch_size
        xs = []
        ys = []
        for text, label in text_label_pairs[:num_batches * batch_size]:
            tokens = [x.lower() for x in word_tokenize(text)]
            wid_list = list()
            for w in tokens:
                wid = 0
                if w in self.word2idx:
                    wid = self.word2idx[w]
                wid_list.append(wid)
            xs.append(wid_list)
            ys.append(self.labels[label])

        # shape of X at this point is (num_batches * batch_size, sequence_length)
        X = pad_sequences(xs, max_seq_length=self.sequence_length, ctx=self.data_ctx)
        # reshape X to (num_batches, batch_size, sequence_length)
        X = X.reshape((num_batches, batch_size, self.sequence_length))
        Y = nd.array(ys, ctx=self.data_ctx).reshape((num_batches, batch_size))

        return X, Y

    def fit(self, text_data_model, text_label_pairs, model_dir_path, batch_size=64, epochs=20,
            learning_rate=0.001, checkpoint_interval=10, test_text_label_pairs=None):

        self.config = text_data_model
        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.sequence_length = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']

        np.save(self.get_config_file_path(model_dir_path), self.config)

        self.model = self.create_model()
        self.model.collect_params().initialize(init=mx.init.Xavier(magnitude=2.24), ctx=self.model_ctx)

        trainer = gluon.Trainer(self.model.collect_params(), optimizer='adam', optimizer_params={
            'learning_rate': learning_rate
        })

        shuffle(text_label_pairs)
        X, Y = self.encode_text(text_label_pairs, batch_size)

        history = dict()
        loss_train_seq = []
        acc_train_seq = []
        loss_test_seq = []
        acc_test_seq = []

        num_samples = len(text_label_pairs)
        num_batches = num_samples // batch_size

        softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()

        for e in range(epochs):
            cumulative_loss = .0
            accuracy = mx.metric.Accuracy()
            for i, (data, label) in enumerate(zip(X, Y)):
                data = data.as_in_context(self.model_ctx)
                label = label.as_in_context(self.model_ctx)
                with autograd.record():
                    output = self.model(data)
                    prediction = nd.argmax(output, axis=1)
                    accuracy.update(preds=prediction, labels=label)
                    loss = softmax_loss(output, label)
                    loss.backward()
                trainer.step(batch_size)
                batch_loss = nd.sum(loss).asscalar()
                batch_avg_loss = batch_loss / data.shape[0]
                cumulative_loss += batch_loss
                print("Epoch %s / %s, Batch %s / %s. Loss: %s, Accuracy : %s" %
                      (e + 1, epochs, i + 1, num_batches, batch_avg_loss, accuracy.get()[1]))

            train_acc = accuracy.get()[1]
            acc_train_seq.append(train_acc)
            if test_text_label_pairs is None:
                print("Epoch %s / %s. Loss: %s. Accuracy: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_acc))
            else:
                test_acc, test_avg_loss = self.evaluate_accuracy(test_text_label_pairs,
                                                                 batch_size=batch_size)
                acc_test_seq.append(test_acc)
                loss_test_seq.append(test_avg_loss)

                print("Epoch %s / %s. Loss: %s. Accuracy: %s. Test Accuracy: %s." %
                      (e + 1, epochs, cumulative_loss / num_samples, train_acc, test_acc))

            if e % checkpoint_interval == 0:
                self.checkpoint(model_dir_path)
            loss_train_seq.append(cumulative_loss)

        self.checkpoint(model_dir_path)

        np.save(self.get_history_file_path(model_dir_path), history)

        return history

    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else 1 for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.sequence_length)
        output = self.model(x)
        return output.asnumpy()[0]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

    def test_run(self, sentence):
        print(self.predict(sentence))
