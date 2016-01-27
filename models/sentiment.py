
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
		num_classes = 11
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
		self.learning_rate * lr_decay)
		initializer = tf.random_uniform_initializer(-1,1)
		self.batch_pointer = 0
		self.seq_input = []
		self.seq_lengths = []
		self.dropout = dropout

		self.global_step = tf.Variable(0, trainable=False)
		self.max_seq_length = max_seq_length

		#seq_input: list of tensors, each tensor is size max_seq_length
		#target: a list of values betweeen 0 and 1 indicating target scores
		#seq_lengths:the early stop lengths of each input tensor
		for i in range(max_seq_length):
			self.seq_input.append(tf.placeholder(tf.int32, shape=[None],
			name="input{0}".format(i)))
		self.target = tf.placeholder(tf.float32, name="target", shape=[None,num_classes])
		self.seq_lengths = tf.placeholder(tf.int32, shape=[None],
		name="early_stop")

		#create input embedding layer
		#there is some sort of issue with using gpu for this layer
		#which is why ive forced it to be run on the cpu
		with tf.device("/cpu:0"):
			embeddings = tf.Variable(tf.random_uniform((self.vocab_size, hidden_size), -1.0, 1.0))
			emb =[tf.nn.embedding_lookup(embeddings, inp) for inp in self.seq_input]

		#create hidden lstm layers
		cell = rnn_cell.LSTMCell(hidden_size, hidden_size, initializer=initializer)
		if not forward_only and dropout < 1.0:
			cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
		self.cell = rnn_cell.MultiRNNCell([cell] * num_layers)
		self.initial_state = self.cell.zero_state(batch_size, tf.float32)
		self.hidden_outputs, self.states = rnn.rnn(self.cell, emb, dtype=tf.float32)

		#output logistic regression layer

		weights = tf.Variable(tf.random_normal([hidden_size,num_classes], stddev=0.01))
		bias = tf.Variable(tf.random_normal([num_classes], stddev=0.01))

		with tf.name_scope("output_proj") as scope:
			self.y = tf.matmul(self.hidden_outputs[-1], weights) + bias
		w_hist = tf.histogram_summary("weights", weights)
		b_hist = tf.histogram_summary("biases", bias)
		#compute losses
		with tf.name_scope("loss") as scope:
			self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.target))

		params = tf.trainable_variables()
		if not forward_only:
			#self.gradient_norms = []
			with tf.name_scope("train") as scope:
				opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.target,1))
			with tf.name_scope("accuracy") as scope:
				self.accuracy = tf.equal(tf.argmax(self.y,1), tf.argmax(self.target,1))
			gradients = tf.gradients(self.losses, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients,
			max_gradient_norm)
			self.gradient_norms = norm
			self.updates = opt.apply_gradients(
			zip(clipped_gradients, params), global_step=self.global_step)
		self.saver = tf.train.Saver(tf.all_variables())

	def getBatch(self, data, test_data=False):
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
		start = self.batch_pointer * self.batch_size
		seq_lengths = (data.transpose()[-1]).transpose()
		targets = (data.transpose()[-2]).transpose()
		onehot = np.zeros((len(targets), 11))
		onehot[np.arange(len(targets)), targets] = 1
		#cut off last two columns (score and seq length)
		data = (data.transpose()[0:-2]).transpose()
		batch_inputs = []
		if not test_data:
			for length_idx in xrange(self.max_seq_length):
				  batch_inputs.append(np.array([data[batch_idx][length_idx]
				for batch_idx in xrange(start, start + self.batch_size)], dtype=np.int32))
			self.batch_pointer += 1
			self.batch_pointer = self.batch_pointer % (len(data) / self.batch_size)
			onehot = onehot[start: start + self.batch_size]
			assert len(batch_inputs[0]) == self.batch_size
			seq_lengths = seq_lengths[start: start + self.batch_size]
		else:
			for length_idx in xrange(self.max_seq_length):
				  batch_inputs.append(
				  np.array([data[batch_idx][length_idx]
				for batch_idx in xrange(len(targets))], dtype=np.int32))
		return batch_inputs, onehot, seq_lengths


	def step(self, session, inputs, targets, seq_lengths, forward_only=False):
		'''
		Inputs:
		session: tensorflow session
		inputs: list of list of ints representing tokens in review of batch_size
		output: list of sentiment scores
		seq_lengths: list of sequence lengths, provided at runtime to prevent need for padding

		Returns:
		accuracy, loss, gradient norms
		or (in forward only):
		accuracy, loss, outputs
		'''
		input_feed = {}
		for i in xrange(self.max_seq_length):
			input_feed[self.seq_input[i].name] = inputs[i]
		input_feed[self.target.name] = targets
		if not forward_only:
			output_feed = [self.accuracy, self.losses, self.gradient_norms]
		else:
			output_feed = [self.accuracy, self.losses, self.y]
		input_feed[self.seq_lengths.name] = seq_lengths
		outputs = session.run(output_feed, input_feed)

		if not forward_only:
			return outputs[0], outputs[1], outputs[2]
		else:
			return outputs[0], outputs[1], outputs[2]
