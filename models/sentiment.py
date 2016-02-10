
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
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
	learning_rate: the learning rate to use in param adjustment
	lr_decay:rate at which to decayse learning rate
	forward_only: whether to run backward pass or not
	'''
	def __init__(self, vocab_size, hidden_size, dropout,
	num_layers, max_gradient_norm, max_seq_length,
	learning_rate, lr_decay, forward_only=False):
		self.num_classes =2
		self.vocab_size = vocab_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
		self.learning_rate * lr_decay)
		initializer = tf.random_uniform_initializer(-1,1)
		self.batch_pointer = 0
		self.seq_input = []
		self.seq_lengths = []
		self.dropout = dropout
		self.max_gradient_norm = max_gradient_norm
		self.global_step = tf.Variable(0, trainable=False)
		self.max_seq_length = max_seq_length

		#seq_input: list of tensors, each tensor is size max_seq_length
		#target: a list of values betweeen 0 and 1 indicating target scores
		#seq_lengths:the early stop lengths of each input tensor
		for i in range(max_seq_length):
			self.seq_input.append(tf.placeholder(tf.int32, shape=[None],
			name="input{0}".format(i)))
		self.target = tf.placeholder(tf.float32, name="target", shape=[None,self.num_classes])
		self.seq_lengths = tf.placeholder(tf.int32, shape=[None],
		name="early_stop")

		cell = rnn_cell.LSTMCell(hidden_size, hidden_size, initializer=initializer)
		#If multiple layers are wanted
		if num_layers >1:
			cell = rnn_cell.MultiRNNCell([cell] * num_layers)
		if not forward_only and dropout < 1.0:
			cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
		#create input embedding layer
		cell = rnn_cell.EmbeddingWrapper(cell, vocab_size)

		encoder_outputs, encoder_state = rnn.rnn(cell, self.seq_input, dtype=tf.float32)

		#Get average last hidden state over time
		# size of concat_states = (batch_size * seq_length) x hidden_size
		concat_states =  tf.pack(encoder_state)

		#size of avg_states = batch_size x hidden_size*2
		avg_states = tf.reduce_mean(concat_states, 0)

		#size of avg_last_state = batch_size x hidden_size
		avg_last_state = tf.slice(avg_states, [0, hidden_size], [-1, hidden_size])

		#output logistic regression layer
		weights = tf.Variable(tf.random_normal([hidden_size,self.num_classes], stddev=0.01))
		bias = tf.Variable(tf.random_normal([self.num_classes], stddev=0.01))

		with tf.name_scope("output_proj") as scope:
			self.y = tf.matmul(avg_last_state, weights) + bias
		#compute losses, minimize cross entropy
		with tf.name_scope("loss") as scope:
			self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.target))
			loss_summ = tf.scalar_summary("loss", self.losses)
		self.y = tf.nn.softmax(self.y)
		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.target,1))
		with tf.name_scope("accuracy") as scope:
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			acc_summ = tf.scalar_summary("accuracy", self.accuracy)

		params = tf.trainable_variables()
		if not forward_only:
			with tf.name_scope("train") as scope:
				opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			gradients = tf.gradients(self.losses, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
			with tf.name_scope("grad_norms") as scope:
				grad_summ = tf.scalar_summary("grad_norms", norm)
			self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
			self.saver = tf.train.Saver(tf.all_variables())
			self.merged = tf.merge_all_summaries()

	def getBatch(self, test_data=False):
		'''
		Get a random batch of data to preprocess for a step
		not sure how efficient this is...

		Input:
		data: shuffled batchxnxm numpy array of data
		train_data: flag indicating whether or not to increment batch pointer, in other
			word whether to return the next training batch, or cross val data

		Returns:
		A numpy arrays for inputs, target, and seq_lengths

		'''
		batch_inputs = []
		if not test_data:
			temp = self.train_data[self.train_batch_pointer].transpose()
			for i in range(self.max_seq_length):
				batch_inputs.append(temp[i])
			targets = self.train_targets[self.train_batch_pointer]
			seq_lengths = self.train_sequence_lengths[self.train_batch_pointer]
			self.train_batch_pointer += 1
			self.train_batch_pointer = self.train_batch_pointer % len(self.train_data)
			return batch_inputs, targets, seq_lengths
		else:
			temp = self.test_data[self.test_batch_pointer].transpose()
			for i in range(self.max_seq_length):
				batch_inputs.append(temp[i])
			targets = self.test_targets[self.test_batch_pointer]
			seq_lengths = self.test_sequence_lengths[self.test_batch_pointer]
			self.test_batch_pointer += 1
			self.test_batch_pointer = self.test_batch_pointer % len(self.test_data)
			return batch_inputs, targets, seq_lengths

	def initData(self, data, batch_size, train_start_end_index, test_start_end_index):
		'''
		Split data into train/test sets and load into memory
		'''
		self.batch_size = batch_size
		self.train_batch_pointer = 0
		self.test_batch_pointer = 0
		#cutoff non even number of batches
		targets = (data.transpose()[-2]).transpose()
		onehot = np.zeros((len(targets), 2))
		onehot[np.arange(len(targets)), targets] = 1
		sequence_lengths = (data.transpose()[-1]).transpose()
		data = (data.transpose()[0:-2]).transpose()

		self.train_data = data[train_start_end_index[0]: train_start_end_index[1]]
		self.test_data = data[test_start_end_index[0]:test_start_end_index[1]]
		self.test_num_batch = len(self.test_data) / batch_size

		num_train_batches = len(self.train_data) / batch_size
	 	num_test_batches = len(self.test_data) / batch_size
		train_cutoff = len(self.train_data) - (len(self.train_data) % batch_size)
		test_cutoff = len(self.test_data) - (len(self.test_data) % batch_size)
		self.train_data = self.train_data[:train_cutoff]
		self.test_data = self.test_data[:test_cutoff]

		self.train_sequence_lengths = sequence_lengths[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
		self.train_sequence_lengths = np.split(self.train_sequence_lengths, num_train_batches)
		self.train_targets = onehot[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
		self.train_targets = np.split(self.train_targets, num_train_batches)
		self.train_data = np.split(self.train_data, num_train_batches)

		print "test size is: {0}, splitting into {1} batches".format(len(self.test_data), num_test_batches)
		self.test_data = np.split(self.test_data, num_test_batches)
		self.test_targets = onehot[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
		self.test_targets = np.split(self.test_targets, num_test_batches)
		self.test_sequence_lengths = sequence_lengths[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
		self.test_sequence_lengths = np.split(self.test_sequence_lengths, num_test_batches)

	def step(self, session, inputs, targets, seq_lengths, forward_only=False):
		'''
		Inputs:
		session: tensorflow session
		inputs: list of list of ints representing tokens in review of batch_size
		output: list of sentiment scores
		seq_lengths: list of sequence lengths, provided at runtime to prevent need for padding

		Returns:
		merged_tb_vars, loss, none
		or (in forward only):
		accuracy, loss, outputs
		'''
		input_feed = {}
		for i in xrange(self.max_seq_length):
			input_feed[self.seq_input[i].name] = inputs[i]
		input_feed[self.target.name] = targets
		input_feed[self.seq_lengths.name] = seq_lengths
		if not forward_only:
			output_feed = [self.merged, self.losses, self.update]
		else:
			output_feed = [self.accuracy, self.losses, self.y]
		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[0], outputs[1], None
		else:
			return outputs[0], outputs[1], outputs[2]
