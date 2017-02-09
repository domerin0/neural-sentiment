
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
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
	learning_rate, lr_decay,batch_size, forward_only=False):
		self.num_classes =2
		self.dropout = dropout
		self.vocab_size = vocab_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
		self.learning_rate * lr_decay)
		initializer = tf.random_uniform_initializer(-1,1)
		self.batch_pointer = 0
		self.seq_input = []
		self.batch_size = batch_size
		self.seq_lengths = []
		self.projection_dim = hidden_size
		self.dropout = dropout
		self.max_gradient_norm = max_gradient_norm
		self.global_step = tf.Variable(0, trainable=False)
		self.max_seq_length = max_seq_length

		#seq_input: list of tensors, each tensor is size max_seq_length
		#target: a list of values betweeen 0 and 1 indicating target scores
		#seq_lengths:the early stop lengths of each input tensor
		self.str_summary_type = tf.placeholder(tf.string,name="str_summary_type")
		self.seq_input = tf.placeholder(tf.int32, shape=[None, max_seq_length],
		name="input")
		self.target = tf.placeholder(tf.float32, name="target", shape=[None,self.num_classes])
		self.seq_lengths = tf.placeholder(tf.int32, shape=[None],
		name="early_stop")

		self.dropout_keep_prob_embedding = tf.placeholder(tf.float32,
														  name="dropout_keep_prob_embedding")
		self.dropout_keep_prob_lstm_input = tf.placeholder(tf.float32,
														   name="dropout_keep_prob_lstm_input")
		self.dropout_keep_prob_lstm_output = tf.placeholder(tf.float32,
															name="dropout_keep_prob_lstm_output")

		with tf.variable_scope("embedding"), tf.device("/cpu:0"):
			W = tf.get_variable(
				"W",
				[self.vocab_size, hidden_size],
				initializer=tf.random_uniform_initializer(-1.0, 1.0))
			embedded_tokens = tf.nn.embedding_lookup(W, self.seq_input)
			embedded_tokens_drop = tf.nn.dropout(embedded_tokens, self.dropout_keep_prob_embedding)

		rnn_input = [embedded_tokens_drop[:, i, :] for i in range(self.max_seq_length)]
		with tf.variable_scope("lstm") as scope:
			single_cell = rnn_cell.DropoutWrapper(
				rnn_cell.LSTMCell(hidden_size,
								  initializer=tf.random_uniform_initializer(-1.0, 1.0),
								  state_is_tuple=True),
								  input_keep_prob=self.dropout_keep_prob_lstm_input,
								  output_keep_prob=self.dropout_keep_prob_lstm_output)
			cell = rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)

			initial_state = cell.zero_state(self.batch_size, tf.float32)

			rnn_output, rnn_state = rnn.rnn(cell, rnn_input,
											initial_state=initial_state,
											sequence_length=self.seq_lengths)

		with tf.variable_scope("output_projection"):
			W = tf.get_variable(
				"W",
				[hidden_size, self.num_classes],
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.get_variable(
				"b",
				[self.num_classes],
				initializer=tf.constant_initializer(0.1))
			#we use the cell memory state for information on sentence embedding
			self.scores = tf.nn.xw_plus_b(rnn_state[-1][0], W, b)
			self.y = tf.nn.softmax(self.scores)
			self.predictions = tf.argmax(self.scores, 1)

		with tf.variable_scope("loss"):
			self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.target, name="ce_losses")
			self.total_loss = tf.reduce_sum(self.losses)
			self.mean_loss = tf.reduce_mean(self.losses)

		with tf.variable_scope("accuracy"):
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.target, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		params = tf.trainable_variables()
		if not forward_only:
			with tf.name_scope("train") as scope:
				opt = tf.train.AdamOptimizer(self.learning_rate)
			gradients = tf.gradients(self.losses, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
			with tf.name_scope("grad_norms") as scope:
				grad_summ = tf.scalar_summary("grad_norms", norm)
			self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
			loss_summ = tf.scalar_summary("{0}_loss".format(self.str_summary_type), self.mean_loss)
			acc_summ = tf.scalar_summary("{0}_accuracy".format(self.str_summary_type), self.accuracy)
			self.merged = tf.merge_summary([loss_summ, acc_summ])
		self.saver = tf.train.Saver(tf.all_variables())

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
		#batch_inputs = []
		if not test_data:
			batch_inputs = self.train_data[self.train_batch_pointer]#.transpose()
			#for i in range(self.max_seq_length):
			#	batch_inputs.append(temp[i])
			targets = self.train_targets[self.train_batch_pointer]
			seq_lengths = self.train_sequence_lengths[self.train_batch_pointer]
			self.train_batch_pointer += 1
			self.train_batch_pointer = self.train_batch_pointer % len(self.train_data)
			return batch_inputs, targets, seq_lengths
		else:
			batch_inputs = self.test_data[self.test_batch_pointer]#.transpose()
			#for i in range(self.max_seq_length):
			#	batch_inputs.append(temp[i])
			targets = self.test_targets[self.test_batch_pointer]
			seq_lengths = self.test_sequence_lengths[self.test_batch_pointer]
			self.test_batch_pointer += 1
			self.test_batch_pointer = self.test_batch_pointer % len(self.test_data)
			return batch_inputs, targets, seq_lengths

	def initData(self, data, train_start_end_index, test_start_end_index):
		'''
		Split data into train/test sets and load into memory
		'''
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
		self.test_num_batch = len(self.test_data) / self.batch_size

		num_train_batches = len(self.train_data) / self.batch_size
		num_test_batches = len(self.test_data) / self.batch_size
		train_cutoff = len(self.train_data) - (len(self.train_data) % self.batch_size)
		test_cutoff = len(self.test_data) - (len(self.test_data) % self.batch_size)
		self.train_data = self.train_data[:train_cutoff]
		self.test_data = self.test_data[:test_cutoff]

		self.train_sequence_lengths = sequence_lengths[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
		self.train_sequence_lengths = np.split(self.train_sequence_lengths, num_train_batches)
		self.train_targets = onehot[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
		self.train_targets = np.split(self.train_targets, num_train_batches)
		self.train_data = np.split(self.train_data, num_train_batches)

		print "Test size is: {0}, splitting into {1} batches".format(len(self.test_data), num_test_batches)
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
		merged_tb_vars, loss, outputs
		'''
		input_feed = {}
		#for i in xrange(self.max_seq_length):
		input_feed[self.seq_input.name] = inputs
		input_feed[self.target.name] = targets
		input_feed[self.seq_lengths.name] = seq_lengths
		input_feed[self.dropout_keep_prob_embedding.name] = self.dropout
		input_feed[self.dropout_keep_prob_lstm_input.name] = self.dropout
		input_feed[self.dropout_keep_prob_lstm_output.name] = self.dropout

		if not forward_only:
			input_feed[self.str_summary_type.name] = "train"
			output_feed = [self.merged, self.mean_loss, self.update]
		else:
			input_feed[self.str_summary_type.name] = "test"
			output_feed = [self.merged, self.mean_loss, self.y, self.accuracy]
		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[0], outputs[1], None
		else:
			return outputs[0], outputs[1], outputs[2], outputs[3]
