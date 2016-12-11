import tensorflow as tf


class RNNEncoder:
    def __init__(self, params, scope):
        self.scope = scope
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        with tf.variable_scope(scope):
            if 'word_dim' in params:
                self.word_rnn_dim = None
                self.word_embeddings = tf.get_variable('word_embedding', [params['num_word'], params['word_dim']],
                                                       initializer=initializer)
                self.word_rnn_dim = params['word_rnn_dim']
                self.word_bidirect = params['word_bidirect']
            self.char_rnn_dim = None
            if 'char_dim' in params:
                self.char_embeddings = tf.get_variable('char_embedding', [params['num_char'], params['char_dim']],
                                                       initializer=initializer)
                self.char_rnn_dim = params['char_rnn_dim']
                self.char_dim = params['char_dim']
                self.char_bidirect = params['char_bidirect']

    def rnn_layer(self, embedded_seq, rnn_dim, lengths, is_bidirect):
        with tf.variable_scope('forward'):
            cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
            forward_outputs, forward_state = tf.nn.dynamic_rnn(cell, inputs=embedded_seq, sequence_length=lengths, dtype=embedded_seq.dtype)
        if is_bidirect:
            with tf.variable_scope('backward'):
                cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
                if lengths != None:
                    rev_embedded_seq = tf.reverse_sequence(embedded_seq, lengths, seq_dim=1, batch_dim=None)
                    backward_outputs, backward_state = tf.nn.dynamic_rnn(cell, inputs=rev_embedded_seq,
                                                                         sequence_length=lengths,
                                                                         dtype=embedded_seq.dtype)
                    backward_outputs = tf.reverse_sequence(backward_outputs, lengths, seq_dim=1, batch_dim=None)
                else:
                    rev_embedded_seq = tf.reverse(embedded_seq, [False, True, False])
                    backward_outputs, backward_state = tf.nn.dynamic_rnn(cell, inputs=rev_embedded_seq,
                                                                         sequence_length=lengths,
                                                                         dtype=embedded_seq.dtype)
                    backward_outputs = tf.reverse(backward_outputs, [False, True, False])

            return tf.concat(2, [forward_outputs, backward_outputs]), tf.concat(1, [forward_state, backward_state])
        else:
            return forward_outputs, forward_state

    def encode(self, word_input_ids, sentence_lengths, char_input_ids, word_lengths, reuse):
        inputs = []
        input_dim = 0
        with tf.variable_scope(self.scope, reuse=reuse):
            if self.char_rnn_dim:
                with tf.variable_scope('char_rnn', reuse=reuse):
                    max_word_len = char_input_ids.get_shape()[2]
                    max_seq_len = char_input_ids.get_shape()[1]
                    char_embedded = tf.reshape(
                        tf.nn.embedding_lookup(self.char_embeddings, char_input_ids),
                        tf.pack([-1, max_word_len, self.char_dim])
                    )
                    word_lengths = tf.reshape(word_lengths, [-1])
                    _, char_state = self.rnn_layer(char_embedded, self.char_rnn_dim, word_lengths, self.char_bidirect)

                    char_rnn_dim_ = self.char_rnn_dim * (2 if self.char_bidirect else 1)
                    char_out = tf.reshape(char_state, tf.pack([-1, max_seq_len, char_rnn_dim_]))
                    inputs.append(char_out)
                    input_dim += char_rnn_dim_
            if self.word_rnn_dim:
                word_embedded = tf.nn.embedding_lookup(self.word_embeddings, word_input_ids)
                inputs.append(word_embedded)
                input_dim += self.word_rnn_dim
            inputs = tf.concat(2, inputs)
            with tf.variable_scope('overall_rnn', reuse=reuse):
                word_outputs, _ = self.rnn_layer(inputs, self.word_rnn_dim, sentence_lengths, self.word_bidirect)
                final_outputs = tf.reduce_max(word_outputs, reduction_indices=1)

            return final_outputs  # or return word_state

class CNNEncoder:
    def __init__(self, params, scope):
        self.scope = scope
        self.word_filter_sizes = params.get('word_filter_sizes', None)
        self.word_num_filters = params.get('word_num_filters', None)
        self.char_filter_sizes = params.get('char_filter_sizes', None)
        self.char_num_filters = params.get('char_num_filters', None)
        if self.char_num_filters:
            self._output_dim = len(self.char_filter_sizes) * self.char_num_filters
        else:
            self._output_dim = len(self.word_filter_sizes) * self.word_num_filters
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        with tf.variable_scope(scope):
            if 'word_dim' in params:
                self.embeddings = tf.get_variable('word_embedding', [params['num_word'], params['word_dim']], initializer=initializer)
            if 'char_dim' in params:
                self.embeddings = tf.get_variable('char_embedding', [params['num_char'], params['char_dim']], initializer=initializer)

    @staticmethod
    def cnn_layer(inputs, filter_sizes, num_filters, scope):
        sequence_len = inputs.get_shape()[-2]
        embedding_size = inputs.get_shape()[-1]
        inputs_expanded = tf.expand_dims(inputs, -1)

        pooled_outputs = []
        with tf.variable_scope(scope):
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    w = tf.get_variable(name="w", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                    # w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        inputs_expanded,
                        w,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv"
                    )
                    h = tf.nn.tanh(tf.nn.bias_add(conv, b), name='sigmoid')
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool'
                    )
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(3, pooled_outputs)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat

    @property
    def output_dim(self):
        return self._output_dim

    def encode(self, input_ids, reuse=False):
        inputs = tf.nn.embedding_lookup(self.embeddings, input_ids)
        shape = inputs.get_shape()
        assert len(shape) == 3 or self.char_num_filters
        sentence_len = shape[1]
        with tf.variable_scope(self.scope, reuse=reuse):
            # character-level CNN
            if self.char_num_filters:
                word_len = shape[-2]
                char_dim = shape[-1]
                char_inputs = tf.reshape(inputs, tf.pack([-1, word_len, char_dim]))
                word_inputs = self.cnn_layer(char_inputs, self.char_filter_sizes, self.char_num_filters, "char_cnn_layer")
                word_inputs = tf.reshape(word_inputs, tf.pack([-1, sentence_len, word_inputs.get_shape()[-1]]))
            else:
                word_inputs = inputs
            # word-level CNN
            outputs = self.cnn_layer(word_inputs, self.word_filter_sizes, self.word_num_filters, "word_cnn_layer")  # [batch_size, output_dim]
        return outputs


class AdditionEncoder:
    def __init__(self, params, scope):
        assert 'word_dim' in params
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        with tf.variable_scope(scope):
            self.embeddings = tf.get_variable('word_embedding', [params['num_word'], params['word_dim']], initializer=initializer)

    @staticmethod
    def sequence_mask(lengths, maxlen=None, dtype=tf.bool, name=None):
        """Same as sequence_mask in version 1.1
        """
        with tf.name_scope(name or "SequenceMask"):
            lengths = tf.convert_to_tensor(lengths)
            if lengths.get_shape().ndims != 1:
                raise ValueError("lengths must be 1D for sequence_mask")

            if maxlen is None:
                maxlen = tf.max(lengths, [0])
            else:
                maxlen = tf.convert_to_tensor(maxlen)
            if maxlen.get_shape().ndims != 0:
                raise ValueError("maxlen must be scalar for sequence_mask")

            # The basic idea is to compare a range row vector of size maxlen:
            # [0, 1, 2, 3, 4]
            # to length as a matrix with 1 column: [[1], [3], [2]].
            # Because of broadcasting on both arguments this comparison results
            # in a matrix of size (len(lengths), maxlen)
            result = tf.range(0, maxlen, 1) < tf.expand_dims(lengths, 1)
            if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
                return result
            else:
                return tf.cast(result, dtype)

    def encode(self, input_ids, lengths, reuse=False):
        # with tf.variable_scope(self.scope):
        max_length = input_ids.get_shape()[1]

        if lengths != None:
            mask = self.sequence_mask(tf.to_int32(lengths), max_length, dtype=tf.float32)   # [batch_size, sentence_length]
            mask = tf.expand_dims(mask, -1)
            return tf.reduce_sum((tf.nn.embedding_lookup(self.embeddings, input_ids) * mask), reduction_indices=1)
        else:
            return tf.reduce_sum(tf.nn.embedding_lookup(self.embeddings, input_ids), reduction_indices=1)


class RelationMatcherModel:
    def __init__(self, params):
        self.pos_relation_ids = tf.placeholder(tf.int32, [None, 3])
        self.neg_relation_ids = tf.placeholder(tf.int32, [None, 3])
        self.q_word_ids = tf.placeholder(tf.int32, [None, params['max_sentence_len']], name='q_word_ids')
        self.q_sentence_lengths = tf.placeholder(tf.int64, [None], name="q_sentence_lengths")
        self.q_char_ids = tf.placeholder(tf.int32, [None, params['max_sentence_len'], params['max_word_len']], name='q_char_ids')
        self.q_word_lengths = tf.placeholder(tf.int64, [None, params['max_sentence_len']])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        if params['encode_name'] == 'CNN':
            self.question_encoder = CNNEncoder(params['question_config'], 'question_cnn')
            # relation_encoder = CNNEncoder(params['relation_config'], 'relation_cnn')
            relation_encoder = AdditionEncoder(params['relation_config'], 'relation_add')
            if 'char_dim' in params['question_config']:
                question = self.question_encoder.encode(self.q_char_ids)
            else:
                question = self.question_encoder.encode(self.q_word_ids)
            pos_relation = relation_encoder.encode(self.pos_relation_ids, None, False)
            neg_relation = relation_encoder.encode(self.neg_relation_ids, None, True)

        elif params['encode_name'] == 'ADD':
            self.question_encoder = AdditionEncoder(params['question_config'], 'question_add')
            relation_encoder = AdditionEncoder(params['relation_config'], 'relation_add')
            question = self.question_encoder.encode(self.q_word_ids, self.q_sentence_lengths)
            pos_relation = relation_encoder.encode(self.pos_relation_ids, None, False)
            neg_relation = relation_encoder.encode(self.neg_relation_ids, None, True)
        elif params['encode_name'] == 'RNN':
            self.question_encoder = RNNEncoder(params['question_config'], 'question_rnn')
            # relation_encoder = RNNEncoder(params['relation_config'], 'relation_rnn')
            relation_encoder = AdditionEncoder(params['relation_config'], 'relation_add')
            question = self.question_encoder.encode(self.q_word_ids, self.q_sentence_lengths, self.q_char_ids, self.q_word_lengths, False)
            pos_relation = relation_encoder.encode(self.pos_relation_ids, None, False)
            neg_relation = relation_encoder.encode(self.neg_relation_ids, None, True)
        else:
            raise ValueError('encoder_name should be one of [CNN, ADD, RNN]')

        self.question_drop = tf.nn.dropout(question, self.dropout_keep_prob)
        self.pos_relation_drop = tf.nn.dropout(pos_relation, self.dropout_keep_prob)
        neg_relation_drop = tf.nn.dropout(neg_relation, self.dropout_keep_prob)
        self.pos_sim = self.sim(self.question_drop, self.pos_relation_drop)
        neg_sim = self.sim(self.question_drop, neg_relation_drop)
        self.loss = tf.reduce_sum(tf.maximum(0., neg_sim + params['margin'] - self.pos_sim))
        tvars = tf.trainable_variables()
        max_grad_norm = 5
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(params['lr'])
        self.train_op = optimizer.apply_gradients(zip(self.grads, tvars))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session = tf.Session(config=config)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        if params['load_path']:
            self.saver.restore(self.session, params['load_path'])
        else:
            self.session.run(tf.initialize_all_variables())

        self.params = params

    @staticmethod
    def sim(u, v):
        dot = tf.reduce_sum(tf.mul(u, v), 1)
        sqrt_u = tf.sqrt(tf.reduce_sum(u ** 2, 1))
        sqrt_v = tf.sqrt(tf.reduce_sum(v ** 2, 1))
        epsilon = 1e-5
        cosine = dot / (sqrt_u * sqrt_v)
        # cosine = dot / (tf.maximum(sqrt_u * sqrt_v, epsilon))
        # cosine = tf.maximum(dot / (tf.maximum(sqrt_u * sqrt_v, epsilon)), epsilon)
        return cosine

    def fit(self,
            question_word_ids,
            question_sentence_lengths,
            question_char_ids,
            question_word_lengths,
            pos_relation_ids,
            neg_relation_ids,
            dropout_keep_prob):
        feed_dict = dict()

        if 'word_dim' in self.params['question_config']:
            feed_dict[self.q_word_ids] = question_word_ids
            feed_dict[self.q_sentence_lengths] = question_sentence_lengths

        if 'char_dim' in self.params['question_config']:
            feed_dict[self.q_char_ids] = question_char_ids
            feed_dict[self.q_word_lengths] = question_word_lengths

        feed_dict[self.dropout_keep_prob] = dropout_keep_prob
        feed_dict[self.pos_relation_ids] = pos_relation_ids
        feed_dict[self.neg_relation_ids] = neg_relation_ids
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss

    def predict(self,
                question_word_ids,
                question_sentence_lengths,
                question_char_ids,
                question_char_lengths,
                relation_ids,
                include_repr=False):
        feed_dict = dict()
        if 'word_dim' in self.params['question_config']:
            feed_dict[self.q_word_ids] = question_word_ids
            feed_dict[self.q_sentence_lengths] = question_sentence_lengths

        if 'char_dim' in self.params['question_config']:
            feed_dict[self.q_char_ids] = question_char_ids
            feed_dict[self.q_word_lengths] = question_char_lengths

        feed_dict[self.dropout_keep_prob] = 1
        feed_dict[self.pos_relation_ids] = relation_ids

        if include_repr:
            return self.session.run([self.pos_sim, self.question_drop, self.pos_relation_drop], feed_dict)
        else:
            return self.session.run(self.pos_sim, feed_dict)

    def get_question_repr(self,
                          question_word_ids,
                          question_sentence_lengths,
                          question_char_ids,
                          question_char_lengths):
        feed_dict = dict()
        if 'word_dim' in self.params['question_config']:
            feed_dict[self.q_word_ids] = question_word_ids
            feed_dict[self.q_sentence_lengths] = question_sentence_lengths

        if 'char_dim' in self.params['question_config']:
            feed_dict[self.q_char_ids] = question_char_ids
            feed_dict[self.q_word_lengths] = question_char_lengths
        feed_dict[self.dropout_keep_prob] = 1
        return self.session.run(self.question_drop, feed_dict)

    def get_relation_repr(self,
                          relation_ids):
        feed_dict = dict()
        feed_dict[self.pos_relation_ids] = relation_ids
        feed_dict[self.dropout_keep_prob] = 1
        return self.session.run(self.pos_relation_drop, feed_dict)

    def save(self, save_path):
        return self.saver.save(self.session, save_path)

