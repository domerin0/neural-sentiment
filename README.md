# neural-sentiment
A sentiment analyzer using deep rnns, built with TensorFlow.

### Installation & Dependency Instructions

1. Install [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html)
2. `sudo pip install nltk`

### Usage Instructions

To run with preset hyper-parameters just run:

`python train.py`

It will download the data set, unzip, and process it. This will take several minutes. The training will begin automatically after this is done.

After your model is trained you can run:

`python sample.py text "Your text to sample here"` 

It will return some number in the interval [0,1]. This number corresponds to a positive (1), or a negative (0) score.

### Model

The model is: embedding layer -> LSTMCells -> logistic regression output layer. I'll provide a picture in the future.

### Tensorboard Usage

I've begun to implement tensorboard variables and histograms.

You can access tensorboard with this application by using:

`tensorboard --logfile="tensorboard --logdir=/tmp/tb_logs/"`

Then copy and paste the localhost url your terminal window gives you, into your browser of choice.

There isn't much to see right no except an overview of the graph. I will include screenshots of this down the road.

### Results

coming soon

### Attribution

The dataset used was the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/).
I also referred to this [tutorial](http://deeplearning.net/tutorial/lstm.html) as a starting point
