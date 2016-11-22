import tensorflow as tf

class RNNEncoder:
    def __init__(self):
        pass


class CNNEncoder:
    def __init__(self, params, scope):
        self.scope = scope
        self.word_filter_sizes = params['word_filter_sizes']
        self.word_num_filters = params['word_num_filters']
        self.char_filter_sizes = params.get('char_filter_size', None)
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
        inputs_expanded = tf.expand_dims(inputs, -1)
        pooled_outputs = []
        with tf.variable_scope(scope):
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, inputs_expanded, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        inputs_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv"
                    )
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID'
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

    def encode(self, input_ids):
        inputs = tf.nn.embedding_lookup(self.embeddings, input_ids)
        shape = inputs.get_shape()
        assert len(shape) == 3 or self.char_num_filters
        sentence_len = shape[1]
        with tf.variable_scope(self.scope):
            # character-level CNN
            if self.char_num_filters:
                word_len = shape[-2]
                char_dim = shape[-1]
                char_inputs = tf.reshape(inputs, [-1, word_len, char_dim])
                word_inputs = self.cnn_layer(char_inputs, self.char_filter_sizes, self.char_num_filters, "char_cnn_layer")
                word_inputs = tf.reshape(word_inputs, [-1, sentence_len, word_inputs.get_shape()[-1]])
            else:
                word_inputs = inputs
        # character-level CNN
        outputs = self.cnn_layer(word_inputs, self.word_filter_sizes, self.word_num_filters, "word_cnn_layer")  # [batch_size, output_dim]
        return outputs


class AdditionEncoder:
    def __init__(self, params, scope):
        assert 'word_dim' in params
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        with tf.variable_scope(scope):
            self.embeddings = tf.get_variable('word_embedding', [params['num_word'], params['word_dim']], initializer=initializer)


    def encode(self, input_ids, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            return tf.reduce_sum(tf.nn.embedding_lookup(self.embeddings, input_ids), axis=1)

class RelationMatcher:
    def __init__(self, margin, params, load_path, encoder_name="ADD"):
        self.pos_relation_ids = tf.placeholder(tf.int32, [None, 3])
        self.neg_relation_ids = tf.placeholdertf.int32, [None, 3]
        self.q_word_ids = tf.placeholder(tf.int32, [None, params['max_sentence_len']], name='q_word_ids')
        self.q_sentence_lengths = tf.placeholder(tf.int32, [None], name="q_sentence_lengths")
        self.q_char_ids = tf.placeholder(tf.int32, [None, params['max_sentence_len'], params['max_word_len']], name='q_char_ids')
        self.q_word_lengths = tf.placeholder(tf.int32, [None, params['max_sentence_len']])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        if encoder_name == 'CNN':
            question_encoder = CNNEncoder(params['question_config'], 'question_cnn')
            relation_encoder = CNNEncoder(params['relation_config'], 'relation_cnn')
            if 'char_dim' in params['question_config']:
                question = question_encoder.encode(self.q_char_ids)
            else:
                question = question_encoder.encode(self.q_word_ids)
            pos_relation = relation_encoder.encode(self.pos_relation_ids)
            neg_relation = relation_encoder.encode(self.neg_relation_ids)

        elif encoder_name == 'ADD':
            question_encoder = AdditionEncoder(params['question_config'], 'question_add')
            relation_encoder = AdditionEncoder(params['relation_config'], 'relation_add')
            question = question_encoder.encode(self.q_word_ids)
            pos_relation = relation_encoder.encode(self.pos_relation_ids)
            neg_relation = relation_encoder.encode(self.neg_relation_ids)
        else:
            raise ValueError('encoder_name should be one of [CNN, ADD, RNN]')

        question_drop = tf.nn.dropout(question, self.dropout_keep_prob)
        pos_relation_drop = tf.nn.dropout(pos_relation, self.dropout_keep_prob)
        neg_relation_drop = tf.nn.dropout(neg_relation, self.dropout_keep_prob)
        self.pos_sim = self.sim(question_drop, pos_relation_drop)
        neg_sim = self.sim(question_drop, neg_relation_drop)
        self.loss = tf.reduce_sum(tf.maximum(0., neg_sim + margin - self.pos_sim))
        tvars = tf.trainable_variables()
        max_grad_norm = 5
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(params['lr'])
        self.train_op = optimizer.apply_gradients(zip(self.grads, tvars))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session = tf.InteractiveSession(config=config)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        if load_path:
            self.saver.restore(self.session, load_path)
        else:
            self.session.run(tf.initialize_all_variables())

        self.params = params

    @staticmethod
    def sim( u, v):
        dot = tf.reduce_sum(tf.mul(u, v), 1)
        sqrt_u = tf.sqrt(tf.reduce_sum(u ** 2, 1))
        sqrt_v = tf.sqrt(tf.reduce_sum(v ** 2, 1))
        epsilon = 1e-5
        cosine = dot / (tf.maximum(sqrt_u * sqrt_v, epsilon))
        # cosine = tf.maximum(dot / (tf.maximum(sqrt_u * sqrt_v, epsilon)), epsilon)
        return cosine

    def fit(self, question_word_ids, question_lengths, question_char_ids, question_word_lengths, relation_ids, dropout_keep_prob):
        feed_dict = dict()

        if 'word_dim' in self.params:
            feed_dict[self.q_word_ids] = question_word_ids
            feed_dict[self.q_sentence_lengths] = question_lengths

        if 'char_dim' in self.params:
            feed_dict[self.q_char_ids] = question_char_ids
            feed_dict[self.q_word_lengths] = question_word_lengths




