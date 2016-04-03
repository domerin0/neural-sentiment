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
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/1459487748hiddensize_128_dropout_0.8_numlayers_1', 'Directory to store/restore checkpoints')
flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')

def main():
	vocab_mapping = util.vocabmapping.VocabMapping()
	with tf.Session() as sess:
		model = loadModel(sess, vocab_mapping.getSize())
		if model == None:
			return
		max_seq_length = model.max_seq_length
		test_data  = [FLAGS.text.lower(), "second test etxt", "third one to test"]
		for text in test_data:
			data, seq_lengths, targets = prepareText(text, max_seq_length, vocab_mapping)
			input_feed = {}
			input_feed[model.seq_input.name] = data
			input_feed[model.target.name] = targets
			input_feed[model.seq_lengths.name] = seq_lengths
			output_feed = [model.y]
			outputs = sess.run(output_feed, input_feed)
			score = np.argmax(outputs[0])
			probability = outputs[0].max(axis=1)[0]
			print "Value of sentiment: {0} with probability: {1}".format(score , probability)

def prepareText(text, max_seq_length, vocab_mapping):
	'''
	Input:
	text_list: a list of strings

	Returns:
	inputs, seq_lengths, targets
	'''
	data = np.array([i for i in range(max_seq_length)])
	targets = []
	seq_lengths = []
	tokens = tokenize(text)
	if len(tokens) > max_seq_length:
		tokens = tokens[0:max_seq_length]
	inputs = []

	indices = [vocab_mapping.getIndex(j) for j in tokens]
	if len(indices) < max_seq_length:
		indices = indices + [vocab_mapping.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
	else:
		indices = indices[0:max_seq_length]
	seq_lengths.append(len(tokens))

	data = np.vstack((data, indices))
	targets.append(1)

	onehot = np.zeros((len(targets), 2))
	onehot[np.arange(len(targets)), targets] = 1
	return data[1::], np.array(seq_lengths), onehot


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
