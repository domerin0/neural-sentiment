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

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')

def main():
	vocab_mapping = util.vocabmapping.VocabMapping()
	with tf.Session() as sess:
		model = loadModel(sess, vocab_mapping.getSize())
		if model == None:
			return
		tokens = tokenize(FLAGS.text.lower())
		if len(tokens) > model.max_seq_length:
			tokens = tokens[0:model.max_seq_length]
		indices = [vocab_mapping.getIndex(j) for j in tokens]
		seq_lengths = [len(indices)]
		inputs = []
		indices = np.array(indices + [vocab_mapping.getIndex("<PAD>") for j in range(model.max_seq_length - len(tokens))])
		for length_idx in xrange(model.max_seq_length):
			  inputs.append(
			  np.array([indices[length_idx]], dtype=np.int32))
		#dummy target, we don't really need this.
		targets = [1]
		onehot = np.zeros((len(targets), 2))
		onehot[np.arange(len(targets)), targets] = 1

		input_feed = {}
		for i in xrange(model.max_seq_length):
			input_feed[model.seq_input[i].name] = inputs[i]
		input_feed[model.target.name] = onehot
		input_feed[model.seq_lengths.name] = seq_lengths
		output_feed = [model.y]

		outputs = sess.run(output_feed, input_feed)
		score = np.argmax(outputs[0])
		probability = outputs[0].max(axis=1)[0]
		print "Value of sentiment: {0} with probability: {1}".format(score , probability)


def loadModel(session, vocab_size):
	hParams = restoreHyperParameters()
	model = models.sentiment.SentimentModel(int(hParams[0]), int(hParams[1]),
	float(hParams[2]), int(hParams[3]), int(hParams[4]), int(hParams[5]),
	float(hParams[6]),float(hParams[7]) ,True)
	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
		print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
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
	path = os.path.join(FLAGS.checkpoint_dir, "hyperparams.npy")
	return np.load(path)

def tokenize(text):
	text = text.decode('utf-8')
	return nltk.word_tokenize(text)

if __name__ == "__main__":
	main()
