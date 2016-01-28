'''
I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and semi-based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html

Written by: Dominik Kaukinen
'''
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
from tensorflow.python.platform import gfile
import numpy as np
import sys
import os
import nltk
from six.moves import xrange
import util.dataprocessor
import models.sentiment
import util.vocabmapping

#Defaults for network parameters
hidden_size = 128
max_seq_length = 500
num_layers = 1
batch_size = 25
max_epoch = 150
learning_rate = 0.001
lr_decay_factor = 0.01
steps_per_checkpoint = 10
checkpoint_dir = "data/checkpoints/"
text = "Hello World!"

x = '''
cmd line args will be:
text: the text you want to test against
checkpoint_dir: directory to store/restore checkpoints
'''


def main():
    setNetworkParameters()
    vocab_mapping = util.vocabmapping.VocabMapping()
    with tf.Session() as sess:
        model = loadModel(sess, vocab_mapping.getSize())
        if model == None:
            return
        tokens = tokenize(text.lower())
        if len(tokens) > model.max_seq_length:
            tokens = tokens[0:model.max_seq_length]
        indices = [vocab_mapping.getIndex(j) for j in tokens]
        seq_lengths = [len(indices)]
        inputs = []
        for length_idx in xrange(model.max_seq_length):
              inputs.append(
              np.array([indices[0][length_idx]], dtype=np.int32))
        #dummy target, we don't really need this.
        targets = [1]
        indices = np.array(indices + [vocab_mapping.getIndex("<PAD>") for j in range(model.max_seq_length - len(tokens))])
        assert len(targets) == len(inputs), "input len: {0}, target len: {1}".format(len(inputs), len(targets))
        _, _, output = model.step(sess, inputs, targets, seq_lengths, True)
        print(len(output[0]))
        print len(output)
        print "Value of sentiment: {0}".format(output[-1])


def loadModel(session, vocab_size):
    hParams = restoreHyperParameters()
    model = models.sentiment.SentimentModel(int(hParams[0]), int(hParams[1]),
    float(hParams[2]), int(hParams[3]), int(hParams[4]), int(hParams[5]),
    1, float(hParams[7]),float(hParams[8]) ,True)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "Double check you got the checkpoint_dir right..."
        print "Model not found..."
        model = None
    return model

'''
Restore training hyper parameters.
This is a hack mostly, but I couldn't find another way to do this.
Ultimately, I don't think is that bad.
'''
def restoreHyperParameters():
    path = os.path.join(checkpoint_dir, "hyperparams.npy")
    return np.load(path)


def setNetworkParameters():
    try:
        if "checkpoint_dir" in sys.argv:
            global checkpoint_dir
            checkpoint_dir = sys.argv[sys.argv.index("checkpoint_dir") + 1]
        if "text" in sys.argv:
            global text
            text = sys.argv[sys.argv.index("text") + 1]
    except Exception as a:
        print "Problem with cmd args " + a
        print x

def tokenize(text):
    text = text.decode('utf-8')
    return nltk.word_tokenize(text)

if __name__ == "__main__":
    main()
