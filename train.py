'''
I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and semi-based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html

Written by: Dominik Kaukinen
'''
import tensorflow as tf
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

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_epoch', 50, 'Max number of epochs to train for.')
flags.DEFINE_integer('num_layers', 2, 'Number of hidden layers.')
flags.DEFINE_integer('hidden_size', 50, 'Number of hidden units in hidden layers')
flags.DEFINE_integer('batch_size', 200, 'Size of minibatches.')
flags.DEFINE_integer('steps_per_checkpoint', 50, 'Number of steps before running test set.')
flags.DEFINE_float('lr_decay_factor', 0.97, 'Factor by which to decay learning rate.')
flags.DEFINE_integer('max_seq_length', 200, 'Maximum length of input token sequence')
flags.DEFINE_integer('grad_clip', 5, 'Max gradient norm')
flags.DEFINE_integer('max_vocab_size', 20000, 'Maximum size of source vocab')
flags.DEFINE_float('dropout', 0.5, 'Probability of hidden inputs being removed')
flags.DEFINE_float('train_frac', 0.7, 'Number between 0 and 1 indicating percentage of\
 data to use for training (rest goes to test set)')
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')

def main():
    util.dataprocessor.run(FLAGS.max_seq_length, FLAGS.max_vocab_size)

    #create model
    print "Creating model with..."
    print "Number of hidden layers: {0}".format(FLAGS.num_layers)
    print "Number of units per layer: {0}".format(FLAGS.hidden_size)
    print "Dropout: {0}".format(FLAGS.dropout)
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
    #data = data[:3000]
    num_batches = len(data) / FLAGS.batch_size
    # 70/30 splir for train/test
    train_start_end_index = [0, int(FLAGS.train_frac * len(data))]
    test_start_end_index = [int(FLAGS.train_frac * len(data)) + 1, len(data) - 1]
    print "Number of training examples per batch: {0}, \
    \nNumber of batches per epoch: {1}".format(FLAGS.batch_size,num_batches)
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("/tmp/tb_logs", sess.graph)
        path = getCheckpointPath()
        model = createModel(sess, vocab_size, path)
    #train model and save to checkpoint
        print "Beggining training..."
        print "Maximum number of epochs to train for: {0}".format(FLAGS.max_epoch)
        print "Batch size: {0}".format(FLAGS.batch_size)
        print "Starting learning rate: {0}".format(FLAGS.learning_rate)
        print "Learning rate decay factor: {0}".format(FLAGS.lr_decay_factor)

        step_time, loss = 0.0, 0.0
        previous_losses = []
        tot_steps = num_batches * FLAGS.max_epoch
        model.initData(data, train_start_end_index, test_start_end_index)
        #starting at step 1 to prevent test set from running after first batch
        for step in xrange(1, tot_steps):
            # Get a batch and make a step.
            start_time = time.time()

            inputs, targets, seq_lengths = model.getBatch()
            str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths, False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if step % FLAGS.steps_per_checkpoint == 0:
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
                step_time, loss, test_accuracy = 0.0, 0.0, 0.0
                # Run evals on test set and print their accuracy.
                print "Running test set"
                for test_step in xrange(len(model.test_data)):
                    inputs, targets, seq_lengths = model.getBatch(True)
                    str_summary, test_loss, _, accuracy = model.step(sess, inputs, targets, seq_lengths, True)
                    loss += test_loss
                    test_accuracy += accuracy
                normalized_test_loss, normalized_test_accuracy = loss / len(model.test_data), test_accuracy / len(model.test_data)
                checkpoint_path = os.path.join(path, "sentiment{0}.ckpt".format(normalized_test_accuracy))
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                writer.add_summary(str_summary, step)
                print "Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(normalized_test_loss, normalized_test_accuracy)
                print "-------Step {0}/{1}------".format(step,tot_steps)
                loss = 0.0
                sys.stdout.flush()

def createModel(session, vocab_size, path):
    model = models.sentiment.SentimentModel(vocab_size, FLAGS.hidden_size,
    FLAGS.dropout, FLAGS.num_layers, FLAGS.grad_clip, FLAGS.max_seq_length,
    FLAGS.learning_rate, FLAGS.lr_decay_factor, FLAGS.batch_size)
    saveHyperParameters(vocab_size, path)
    print path
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "Created model with fresh parameters."
        session.run(tf.initialize_all_variables())
    return model

def getCheckpointPath():
    '''
    Check if new hyper params match with old ones
    if not, then create a new model in a new Directory
    Returns:
    path to checkpoint directory
    '''
    old_path = os.path.join(FLAGS.checkpoint_dir, "hyperparams.npy")
    if os.path.exists(old_path):
        params = np.load(old_path)
        ok = \
        params[3] == FLAGS.num_layers and \
        params[1] == FLAGS.hidden_size and \
        params[2] == FLAGS.dropout and \
        params[5] == FLAGS.max_seq_length and \
        params[0] == FLAGS.max_vocab_size
        if ok:
            return FLAGS.checkpoint_dir
        else:
            infostring = "hiddensize_{0}_dropout_{1}_numlayers_{2}".format(FLAGS.hidden_size,
            FLAGS.dropout, FLAGS.num_layers)
            path = os.path.join("data/checkpoints/", str(int(time.time())) + infostring)
            if not os.path.exists(path):
                os.makedirs(path)
            print "hyper parameters changed, training new model at {0}".format(path)
            return path
    else:
        return FLAGS.checkpoint_dir


'''
This function is sort of silly, but I don't know how else to restore the model
from a checkpoint without giving it the same hyper parameters. This method
makes it easier to store (and restore) the hyper parameters of the model.

This only works because they are all numerical types.
'''
def saveHyperParameters(vocab_size, path):
    hParams = np.array([vocab_size, FLAGS.hidden_size,
    FLAGS.dropout, FLAGS.num_layers, FLAGS.grad_clip, FLAGS.max_seq_length,
    FLAGS.learning_rate, FLAGS.lr_decay_factor])
    path = os.path.join(path, "hyperparams.npy")
    np.save(path, hParams)

if __name__ == '__main__':
    main()
