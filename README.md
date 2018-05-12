# mxnet-sentiment-analysis

Sentiment Analysis implemented using Gluon and MXNet

Currently the available models include:

* Sentiment Analysis
    * LSTM
    * bi-directional LSTM
    
# Usage

To train the [SentimentAnalyserWithSoftMaxLSTM](mxnet_sentiment/library/lstm.py) using [umich](demo/data/umich-sentiment-train.txt), run the following
command:

```bash
python demo/lstm_train.py
```

The codes for [lstm_train.py](demo/lstm_train.py) is shown below:

```python
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

    from mxnet_sentiment.library.lstm import SentimentAnalyserWithSoftMaxLSTM
    from mxnet_sentiment.utility.simple_data_loader import load_text_label_pairs
    from mxnet_sentiment.utility.text_utils import fit_text

    text_data_model = fit_text(data_file_path)
    text_label_pairs = load_text_label_pairs(data_file_path)

    train_data, validation_data = train_test_split(text_label_pairs, test_size=0.3, random_state=42)

    rnn = SentimentAnalyserWithSoftMaxLSTM(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))
    history = rnn.fit(text_data_model, text_label_pairs=train_data, model_dir_path=output_dir_path,
                      checkpoint_interval=10, batch_size=64, epochs=20, test_text_label_pairs=validation_data)


if __name__ == '__main__':
    main()

```

The trained models are stored in the [demo/models](demo/models) directory with prefix "lstm-softmax-*"

To test the trained models run the following command:

```bash
python demo/lstm_predict.py
```

The codes for [lstm_predict.py](demo/lstm_predict.py) are shown below:

```python
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

    from mxnet_sentiment.library.lstm import SentimentAnalyserWithSoftMaxLSTM
    from mxnet_sentiment.utility.simple_data_loader import load_text_label_pairs

    text_label_pairs = load_text_label_pairs(data_file_path)

    train_data, validation_data = train_test_split(text_label_pairs, test_size=0.3, random_state=42)

    rnn = SentimentAnalyserWithSoftMaxLSTM(model_ctx=mx.gpu(0), data_ctx=mx.gpu(0))
    rnn.load_model(model_dir_path)

    for text, label in validation_data:
        predicted_label = rnn.predict_class(text)
        print('predicted: ', predicted_label, 'actual: ', label)


if __name__ == '__main__':
    main()
```


# Note

Note that the default training scripts in the [demo](demo) folder use GPU for training, therefore, you must configure your
graphic card for this (or remove the "model_ctx=mxnet.gpu(0)" in the training scripts). 

* Step 1: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (you should download CUDA® Toolkit 9.0)
* Step 2: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 

