# neural-sentiment
A sentiment analyzer using deep rnns, built with TensorFlow.

### Installation & Dependency Instructions

1. Install [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html)
2. `sudo pip install nltk`

### Usage Instructions

To run with preset hyper-parameters just run:

`python train.py`

It will download the data set, unzip, and process it. This will take several minutes. The training will begin automatically after this is done.

### Model

The model is: embedding layer -> LSTMCells -> logistic regression output layer. I'll provide a picture in the future.

### Results

coming soon

### Attribution

The dataset used was the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/).
I also referred to this [tutorial](http://deeplearning.net/tutorial/lstm.html) as a starting point
