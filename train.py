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
import math
import os
import random
import time
from six.moves import xrange
import util.dataprocessor
import models.sentiment
import util.vocabmapping

#Defaults for network parameters
hidden_size = 100
max_seq_length = 100
num_layers = 1
batch_size = 35
max_epoch = 5000
learning_rate = 0.0001
lr_decay_factor = 0.97
steps_per_checkpoint = 100
checkpoint_dir = "data/checkpoints/"
dropout = 0.9
grad_clip = 5
max_vocab_size = 20000
string_args = [("hidden_size", "int"), ("num_layers", "int"), ("batch_size", "int"),
("max_epoch", "int"),("learning_rate", "float"), ("steps_per_checkpoint", "int"),
("lr_decay_factor", "float"), ("max_seq_length", "int"),
("checkpoint_dir","string"), ("dropout", "float"), ("grad_clip", "int"),
("max_vocab_size", "int")]

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
dropout: probability of hidden inputs being removed
grad_clip: max gradient norm
max_vocab_size: maximum size of source vocab
'''
def main():
    setNetworkParameters()
    util.dataprocessor.run(max_seq_length, max_vocab_size)

    #create model
    print "Creating model with..."
    print "Number of hidden layers: {0}".format(num_layers)
    print "Number of units per layer: {0}".format(hidden_size)
    print "Dropout: {0}".format(dropout)
    vocabmapping = util.vocabmapping.VocabMapping()

    vocab_size = vocabmapping.getSize()
    print "Vocab size is: {0}".format(vocab_size)
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
    print "Number of training examples per batch: {0}, \
    number of batches be epoch: {1}".format(batch_size,num_batches)
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("/tmp/tb_logs", sess.graph_def)
        model = createModel(sess, vocab_size)
    #train model and save to checkpoint
        print "Beggining training..."
        print "Maximum number of epochs to train for: " + str(max_epoch)
        print "Batch size: " + str(batch_size)
        print "Starting learning rate: " + str(learning_rate)
        print "Learning rate decay factor: " + str(lr_decay_factor)

        step_time, loss = 0.0, 0.0
        previous_losses = []
        tot_steps = num_batches * max_epoch
        #starting at step 1 to prevent test set from running after first batch
        for step in xrange(1, tot_steps):
            # Get a batch and make a step.
            start_time = time.time()

            inputs, targets, seq_lengths = model.getBatch(data[train_start_end_index[0]:train_start_end_index[1]])

            str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths)

            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if step % steps_per_checkpoint == 0:
                writer.add_summary(str_summary, step)

                # Print statistics for the previous epoch.
                print ("global step %d learning rate %.7f step-time %.2f loss %.4f"
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
                inputs, targets, seq_lengths = model.getBatch(data[test_start_end_index[0]:test_start_end_index[1]], True)
                print "Running test set"
                accuracy, test_loss, _ = model.step(sess, inputs, targets, seq_lengths, True)
                print "Test loss: {0} Accuracy: {1}".format(test_loss, accuracy)
                print "-------Step {0}/{1}------".format(step,tot_steps)
                sys.stdout.flush()

def createModel(session, vocab_size):
    model = models.sentiment.SentimentModel(vocab_size, hidden_size,
    dropout, num_layers, grad_clip, max_seq_length, batch_size,
    learning_rate, lr_decay_factor)
    saveHyperParameters(vocab_size)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

'''
This function is sort of silly, but I don't know how else to restore the model
from a checkpoint without giving it the same hyper parameters. This method
makes it easier to store (and restore) the hyper parameters of the model.

This only works because they are all numerical types.
'''
def saveHyperParameters(vocab_size):
    hParams = np.array([vocab_size, hidden_size,
    dropout, num_layers, grad_clip, max_seq_length, batch_size,
    learning_rate, lr_decay_factor])
    path = os.path.join(checkpoint_dir, "hyperparams.npy")
    np.save(path, hParams)


#to remove code duplication I decided to do some reflection-type parsing
def setNetworkParameters():
    try:
        for arg in string_args:
            exec("if \"{0}\" in sys.argv:\n\
                \tglobal {0}\n\
                \t{0} = {1}(sys.argv[sys.argv.index({0}) + 1])".format(arg[0], arg[1]))
    except Exception as a:
        print "Problem with cmd args " + a
        print x

if __name__ == '__main__':
    main()
