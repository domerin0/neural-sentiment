
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

class SentimentModel(object):
	'''
	Sentiment Model
	params:
	vocab_size: size of vocabulary
	hidden_size: number of units in a hidden layer
	num_layers: number of hidden lstm layers
	max_gradient_norm: maximum size of gradient
	max_seq_length: the maximum length of the input sequence
	batch_size:the size of a batch being passed to the model
	learning_rate: the learning rate to use in param adjustment
	lr_decay:rate at which to decayse learning rate
	forward_only: whether to run backward pass or not
	'''
	def __init__(self, vocab_size, hidden_size, dropout,
	num_layers, max_gradient_norm, max_seq_length, batch_size,
	learning_rate, lr_decay, forward_only=False):

		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
		self.learning_rate * lr_decay)
		initializer = tf.random_uniform_initializer(-1,1)
		self.batch_pointer = 0
		self.seq_input = []
		self.seq_lengths = []
		self.targets = []
		self.dropout = dropout

		self.global_step = tf.Variable(0, trainable=False)
		self.max_seq_length = max_seq_length

		#seq_input: list of tensors, each tensor is size max_seq_length
		#target: a list of values betweeen 0 and 1 indicating target scores
		#seq_lengths:the early stop lengths of each input tensor
		for i in range(batch_size):
			self.seq_input.append(tf.placeholder(tf.int32, shape=[max_seq_length],
			name="input{0}".format(i)))
			self.targets.append(tf.placeholder(tf.float32,
			name="targets{0}".format(i)))
		self.seq_lengths = tf.placeholder(tf.int32, shape=[batch_size],
		name="early_stop")

		#create input embedding layer
		#there is some sort of issue with using gpu for this layer
		#which is why ive forced it to be run on the cpu
		with tf.device("/cpu:0"):
			embeddings = tf.Variable(tf.random_uniform((self.vocab_size, hidden_size), -1.0, 1.0))
			emb =[tf.nn.embedding_lookup(embeddings, inp) for inp in self.seq_input]

		#create hidden lstm layers
		cell = rnn_cell.LSTMCell(hidden_size, hidden_size, initializer=initializer)
		cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
		self.cell = rnn_cell.MultiRNNCell([cell] * num_layers)
		self.initial_state = self.cell.zero_state(batch_size, tf.float32)
		self.hidden_outputs, self.states = rnn.rnn(self.cell, emb, dtype=tf.float32)

		#output logistic regression layer

		weights = tf.Variable(tf.random_normal([hidden_size,1], stddev=0.01))
		bias = tf.Variable(tf.random_normal([1], stddev=0.01))

		with tf.name_scope("output_proj") as scope:
			self.outputs = [tf.nn.sigmoid(tf.matmul(m, weights)) for m in self.hidden_outputs]
		w_hist = tf.histogram_summary("weights", weights)
		b_hist = tf.histogram_summary("biases", bias)
		#compute losses
		self.losses = [tf.reduce_mean((-self.targets[i] * tf.log(self.outputs[i]))
		- ((1 - self.targets[i]) * tf.log(1 - self.outputs[i])))
		for i in range(len(self.targets))]
		#self.losses = [tf.reduce_mean(tf.square(self.targets[i] - self.outputs[i])) for i in range(len(self.targets))]

		params = tf.trainable_variables()
		if not forward_only:
			#self.gradient_norms = []
			with tf.name_scope("train") as scope:
				opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			gradients = tf.gradients(self.losses, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients,
			max_gradient_norm)
			self.gradient_norms = norm
			self.updates = opt.apply_gradients(
			zip(clipped_gradients, params), global_step=self.global_step)
		self.saver = tf.train.Saver(tf.all_variables())

	def getBatch(self, data, train_data=False):
		'''
		Get a random batch of data to preprocess for a step
		not sure how efficient this is...

		Input:
		data: shuffled nxm numpy array of data (last 2 columns are target & seq length)
		train_data: flag indicating whether or not to increment batch pointer, in other
			word whether to return the next training batch, or cross val data

		Returns:
		A numpy arrays for inputs, target, and seq_lengths

		'''
		seq_lengths = (data.transpose()[-1]).transpose()
		targets = (data.transpose()[-2]).transpose()
		#cut off last two columns (score and seq length)
		data = (data.transpose()[0:-2]).transpose()
		batch_inputs = []
		if not train_data:
			start = self.batch_pointer * self.batch_size
			for j in range(start, start + self.batch_size):
				vec = []
				for i in range(self.max_seq_length):
					vec.append(data[j][i])
				batch_inputs.append(vec)
			self.batch_pointer += 1
			self.batch_pointer = (len(data) / self.batch_size) % self.batch_pointer
			targets = targets[start: start + self.batch_size]
			targets = [(0.6 >= targets[i] and 1) or 0 for i in range(len(targets))]
			seq_lengths = seq_lengths[start: start + self.batch_size]
		else:
			for j in range(len(targets)):
				vec = []
				for i in range(min(seq_lengths[j], self.max_seq_length)):
					vec.append(data[j][i])
				batch_inputs.append(vec)
		return batch_inputs, targets, seq_lengths


	def step(self, session, inputs, targets, seq_lengths, forward_only=False):
		'''
		Inputs:
		session: tensorflow session
		inputs: list of list of ints representing tokens in review of batch_size
		output: list of sentiment scores
		seq_lengths: list of sequence lengths, provided at runtime to prevent need for padding

		Returns:
		gradient norm, loss, outputs
		'''
		assert len(inputs) == len(targets)
		assert len(inputs) == len(seq_lengths)
		input_feed = {}
		import util.vocabmapping
		vocab = util.vocabmapping.VocabMapping()
		for i in xrange(len(inputs)):
			assert len(inputs[i]) == self.max_seq_length, "length of seq: {0}".format(str(len(inputs[i])))
			input_feed[self.seq_input[i].name] = inputs[i]
			input_feed[self.targets[i].name] = float(targets[i] / 10.0)
		if not forward_only:
			output_feed = [self.updates,self.gradient_norms, sum(self.losses) / len(inputs)]
		else:
			output_feed = [sum(self.losses) / len(inputs)]
		input_feed[self.seq_lengths.name] = seq_lengths
		for i in xrange(len(self.outputs)):
			output_feed.append(self.outputs[i])

		outputs = session.run(output_feed, input_feed)

		if not forward_only:
			return outputs[1], outputs[2], None
		else:
			return None, outputs[0], outputs[1:]
