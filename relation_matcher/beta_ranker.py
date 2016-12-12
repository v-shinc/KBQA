import tensorflow as tf
import encoder

def fully_connected(input, layer_sizes):
    initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
    input_dim = tf.get_shape(input)[1]
    layer_sizes = [input_dim] + layer_sizes
    layers = [input]
    for i in range(len(layer_sizes)):
        with tf.variable_scope('fully-connected-%s' % i):
            w = tf.get_variable('w', [layer_sizes[i-1], layer_sizes[i]], initializer=initializer)
            b = tf.get_variable('b', [layer_sizes[i]])
            layers.append(tf.add(tf.matmul(layers[-1], w), b))
    return layers[-1]


class BetaRanker:
    def __init__(self, params):
        self.pos_relation_ids = tf.placeholder(tf.int32, [None, 3])
        self.neg_relation_ids = tf.placeholder(tf.int32, [None, 3])
        self.q_word_ids = tf.placeholder(tf.int32, [None, params['max_sentence_len']], name='q_word_ids')
        self.q_sentence_lengths = tf.placeholder(tf.int64, [None], name="q_sentence_lengths")
        self.q_char_ids = tf.placeholder(tf.int32, [None, params['max_sentence_len'], params['max_word_len']], name='q_char_ids')
        self.q_word_lengths = tf.placeholder(tf.int64, [None, params['max_sentence_len']])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.device('/gpu:%s' % params.get('gpu', 1)):
            if params['encode_name'] == 'CNN':
                question_encoder = encoder.CNNEncoder(params['question_config'], 'question_cnn')
                # relation_encoder = CNNEncoder(params['relation_config'], 'relation_cnn')
                relation_encoder = encoder.AdditionEncoder(params['relation_config'], 'relation_add')
                if 'char_dim' in params['question_config']:
                    question = question_encoder.encode(self.q_char_ids)
                else:
                    question = question_encoder.encode(self.q_word_ids)
                pos_relation = relation_encoder.encode(self.pos_relation_ids, None, False)
                neg_relation = relation_encoder.encode(self.neg_relation_ids, None, True)

            elif params['encode_name'] == 'ADD':
                question_encoder = encoder.AdditionEncoder(params['question_config'], 'question_add')
                relation_encoder = encoder.AdditionEncoder(params['relation_config'], 'relation_add')
                question = question_encoder.encode(self.q_word_ids, self.q_sentence_lengths)
                pos_relation = relation_encoder.encode(self.pos_relation_ids, None, False)
                neg_relation = relation_encoder.encode(self.neg_relation_ids, None, True)


            elif params['encode_name'] == 'RNN':
                question_encoder = encoder.RNNEncoder(params['question_config'], 'question_rnn')
                relation_encoder = encoder.RNNEncoder(params['relation_config'], 'relation_rnn')
                # relation_encoder = AdditionEncoder(params['relation_config'], 'relation_add')
                question = question_encoder.encode(self.q_word_ids, self.q_sentence_lengths, self.q_char_ids, self.q_word_lengths, False)
                pos_relation = relation_encoder.encode(self.pos_relation_ids, None, None, None, False)
                neg_relation = relation_encoder.encode(self.neg_relation_ids, None, None, None, True)
                # pos_relation = relation_encoder.encode(self.pos_relation_ids, None, False)
                # neg_relation = relation_encoder.encode(self.neg_relation_ids, None, True)
            else:
                raise ValueError('encoder_name should be one of [CNN, ADD, RNN]')

            # Concat features
            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
            dim = tf.get_shape(question)[1]
            bi_m = tf.get_variable('bi_m', [dim, dim])
            self.pos_bi_sim = tf.batch_matmul(tf.matmul(question, bi_m), pos_relation, adj_y=True)
            slef.neg_bi_sim = tf.batch_matmul(tf.matmul(question, bi_m), neg_relation, adj_y=True)
            pos_features = tf.concat(1, [question, pos_relation, self.pos_bi_sim])
            neg_features = tf.concat(1, [question, neg_relation, self.neg_bi_sim])

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

    def get_all_variables(self):
        variable_names = [v.name for v in tf.all_variables()]
        variable_values = self.session.run(tf.all_variables())
        variable = dict()
        for i in xrange(len(variable_names)):
            variable[variable_names[i]] = variable_values[i].tolist()
        return variable