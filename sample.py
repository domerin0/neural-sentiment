'''
This is an example of sentiment analysis using tensorflow
 with variable sequence length input

I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and loosely based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html
Most notably, I changed the embedding methodology, and of course
did it in tensorflow instead of theano.


Written by: Dominik Kaukinen
'''
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
import numpy as np
import sys
import os
import nltk
from six.moves import xrange
import util.dataprocessor
import models.sentiment
import util.vocabmapping

checkpoint_dir = "data/checkpoints/"
text = "Hello World!"

x = '''
cmd line args will be:
text: the text you want to test against
checkpoint_dir: directory to store/restore checkpoints
'''


def main():
    vocab_mapping = util.vocab_mapping.VocabMapping()
    with tf.Session() as sess:
        model = loadModel()
        if model == None:
            return
        tokens = tokenize(review.read().lower())
        if len(tokens) > model.max_seq_length:
            tokens = tokens[0:model.max_seq_length]
        indices = [vocab_mapping.getIndex(j) for j in tokens]
        indices = [vocab_mapping.getIndex("<PAD>") for j in range(model.max_seq_length - len(tokens))]
        assert len(indices) == model.max_seq_length,"Error! length is"
        indices.append(1)
        indices.append(len(tokens))
        batch_inputs, targets, seq_lengths = model.getBatch(data, train_data=False)
        _, _, output = model.step(sess, batch_inputs, targets, seq_lengths, True)
        print "Value of sentiment: {0}".format(output)


def loadModel():
    model = models.sentiment.SentimentModel(vocab_size, hidden_size,
    num_layers, 5, max_seq_length, 1,
    learning_rate, lr_decay_factor)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "Double check you got the checkpoint_dir right..."
        print "Model not found..."
        model = None
    return model

def setNetworkParameters():
    try:
        if "checkpoint_dir" in sys.argv:
            global checkpoint_dir
            checkpoint_dir = sys.argv[sys.argv.index("checkpoint_dir") + 1]
    except Exception as a:
        print "Problem with cmd args " + a
        print x


if __name__ == "__main__":
    main()
