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
from tensorflow.python.platform import gfile
import numpy as np
import sys
import math
import os
import random
import time
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


x = '''
cmd line args will be:
hidden_size: number of hidden units in hidden layers
num_layers: number of hidden layers
batch_size: size of batchs in training
max_epoch: max number of epochs to train for
learning_rate: beggining learning rate
steps_per_checkpoint: number of steps before running test set
lr_decay_factor: factor by which to decay learning rate
max_seq_length: maximum length of input token sequence
checkpoint_dir: directory to store/restore checkpoints

'''
def main():
    setNetworkParameters()
    util.dataprocessor.run(max_seq_length)

    #create model
    print "Creating model with..."
    print "Number of hidden layers: " + str(num_layers)
    print "Number of units per layer: " + str(hidden_size)
    print ""
    vocabmapping = util.vocabmapping.VocabMapping()

    vocab_size = vocabmapping.getSize()

    path = "data/processed/"
    infile = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    #randomize data order
    data = np.load(os.path.join(path, infile[0]))
    for i in range(1, len(infile)):
        data = np.vstack((data, np.load(os.path.join(path, infile[i]))))
    np.random.shuffle(data)
    num_batches = len(data) / batch_size
    # 70/30 splir for train/test
    train_start_end_index = [0, int(0.7 * len(data))]
    test_start_end_index = [int(0.7 * len(data)) + 1, len(data) - 1]

    with tf.Session() as sess:
        model = createModel(sess, vocab_size)
    #train model and save to checkpoint
        print "Beggining training..."
        print "Maximum number of epochs to train for: " + str(max_epoch)
        print "Batch size: " + str(batch_size)
        print "Starting learning rate: " + str(learning_rate)
        print "Learning rate decay factor: " + str(lr_decay_factor)

        step_time, loss = 0.0, 0.0
        previous_losses = []
        for step in xrange(num_batches * max_epoch):
            # Get a batch and make a step.
            start_time = time.time()
            inputs, targets, seq_lengths = model.getBatch(data[train_start_end_index[0]:train_start_end_index[1]])
            _, step_loss, _ = model.step(sess, inputs, targets, seq_lengths)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if step % steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                print ("global step %d learning rate %.4f step-time %.2f loss %.4f"
                % (model.global_step.eval(), model.learning_rate.eval(),
                step_time, loss))
                         # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(checkpoint_dir, "sentiment.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                inputs, targets, seq_lengths = model.getBatch(data[test_start_end_index[0]:test_start_end_index[1]])
                _, test_loss, _ = model.step(sess, inputs, targets, seq_lengths, True)
                print "Test loss: {0}".format(test_loss)
                sys.stdout.flush()

def createModel(session, vocab_size):
    model = models.sentiment.SentimentModel(vocab_size, hidden_size,
    num_layers, 5, max_seq_length, batch_size,
    learning_rate, lr_decay_factor)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def setNetworkParameters():
    try:
        if "hidden_size" in sys.argv:
            global hidden_size
            hidden_size = int(sys.argv[sys.argv.index("hidden_size") + 1])
        if "num_layers" in sys.argv:
            global num_layers
            num_layers = int(sys.argv[sys.argv.index("num_layers") + 1])
        if "batch_size" in sys.argv:
            global batch_size
            batch_size = int(sys.argv[sys.argv.index("batch_size") + 1])
        if "max_epoch" in sys.argv:
            global max_epoch
            max_epoch = int(sys.argv[sys.argv.index("max_epoch") + 1])
        if "learning_rate" in sys.argv:
            global learning_rate
            max_epoch = int(sys.argv[sys.argv.index("learning_rate") + 1])
        if "lr_decay_factor" in sys.argv:
            global lr_decay_factor
            max_epoch = int(sys.argv[sys.argv.index("lr_decay_factor") + 1])
        if "max_seq_length" in sys.argv:
            global max_seq_length
            max_seq_length = int(sys.argv[sys.argv.index("max_seq_length") + 1])
        if "checkpoint_dir" in sys.argv:
            global checkpoint_dir
            checkpoint_dir = sys.argv[sys.argv.index("checkpoint_dir") + 1]
        if "steps_per_checkpoint" in sys.argv:
            global steps_per_checkpoint
            checkpoint_dir = sys.argv[sys.argv.index("steps_per_checkpoint") + 1]
    except Exception as a:
        print "Problem with cmd args " + a
        print x

if __name__ == '__main__':
    main()
