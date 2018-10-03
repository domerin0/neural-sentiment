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
import os
import nltk
from six.moves import xrange
import util.dataprocessor
import models.sentiment
import util.vocabmapping
import ConfigParser
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')
flags.DEFINE_string('config_file', 'config.ini', 'Path to configuration file.')

def main():
	vocab_mapping = util.vocabmapping.VocabMapping()
	with tf.Session() as sess:
		model = load_model(sess, vocab_mapping.get_size())
		if model == None:
			return
		max_seq_length = model.max_seq_length
		test_data  = [FLAGS.text.lower()]
		for text in test_data:
			data, seq_lengths, targets = prepare_text(text, max_seq_length, vocab_mapping)
			input_feed = {}
			input_feed[model.seq_input.name] = data
			input_feed[model.target.name] = targets
			input_feed[model.seq_lengths.name] = seq_lengths
			input_feed[model.dropout_keep_prob_embedding.name] = model.dropout
			input_feed[model.dropout_keep_prob_lstm_input.name] = model.dropout
			input_feed[model.dropout_keep_prob_lstm_output.name] = model.dropout
			output_feed = [model.y]
			outputs = sess.run(output_feed, input_feed)
			score = np.argmax(outputs[0])
			probability = outputs[0].max(axis=1)[0]
			print("Value of sentiment: {0} with probability: {1}".format(score , probability))

def prepare_text(text, max_seq_length, vocab_mapping):
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

	indices = [vocab_mapping.get_index(j) for j in tokens]
	if len(indices) < max_seq_length:
		indices = indices + [vocab_mapping.get_index("<PAD>") for i in range(max_seq_length - len(indices))]
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
		print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Double check you got the checkpoint_dir right...")
		print("Model not found...")
		model = None
	return model

def load_model(session, vocab_size):
	hyper_params_path = os.path.join(FLAGS.checkpoint_dir, 'hyperparams.p')
	with open(hyper_params_path, 'rb') as hp:
		hyper_params = pickle.load(hp)
	#hyper_params = read_config_file()
	model = models.sentiment.SentimentModel(vocab_size = vocab_size,
											hidden_size = hyper_params["hidden_size"],
											dropout = 1.0,
											num_layers = hyper_params["num_layers"],
											max_gradient_norm = hyper_params["grad_clip"],
											max_seq_length = hyper_params["max_seq_length"],
											learning_rate = hyper_params["learning_rate"],
											lr_decay = hyper_params["lr_decay_factor"],
											batch_size = 1,
											forward_only=True)
	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
		model.saver.restore(session, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
	else:
		print("Double check you got the checkpoint_dir right...")
		print("Model not found...")
		model = None
	return model

def read_config_file():
	'''
	Reads in config file, returns dictionary of network params
	'''
	config = ConfigParser.ConfigParser()
	config.read(FLAGS.config_file)
	dic = {}
	sentiment_section = "sentiment_network_params"
	general_section = "general"
	dic["num_layers"] = config.getint(sentiment_section, "num_layers")
	dic["hidden_size"] = config.getint(sentiment_section, "hidden_size")
	dic["dropout"] = config.getfloat(sentiment_section, "dropout")
	dic["batch_size"] = config.getint(sentiment_section, "batch_size")
	dic["train_frac"] = config.getfloat(sentiment_section, "train_frac")
	dic["learning_rate"] = config.getfloat(sentiment_section, "learning_rate")
	dic["lr_decay_factor"] = config.getfloat(sentiment_section, "lr_decay_factor")
	dic["grad_clip"] = config.getint(sentiment_section, "grad_clip")
	dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
		"use_config_file_if_checkpoint_exists")
	dic["max_epoch"] = config.getint(sentiment_section, "max_epoch")
	dic ["max_vocab_size"] = config.getint(sentiment_section, "max_vocab_size")
	dic["max_seq_length"] = config.getint(general_section,
		"max_seq_length")
	dic["steps_per_checkpoint"] = config.getint(general_section,
		"steps_per_checkpoint")
	return dic

def tokenize(text):
	text = text.decode('utf-8')
	return nltk.word_tokenize(text)

if __name__ == "__main__":
	main()
