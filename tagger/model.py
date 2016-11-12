import tensorflow as tf
import crf

class DeepCRF(object):
    def __init__(self,
                 max_seq_len,
                 max_word_len,
                 char_dim,
                 char_rnn_dim,
                 char_bidirect,
                 word_dim,
                 rnn_dim,
                 word_bidirect,
                 cap_dim,
                 pos_dim,
                 load_path,
                 num_word,
                 num_char,
                 num_cap,
                 num_pos,
                 num_tag):
        self.word_ids = tf.placeholder(tf.int32, [None, max_seq_len], name="word_ids")
        self.seq_lengths = tf.placeholder(tf.int64, [None], name="seq_lengths") # number of valid words
        self.char_for_ids = tf.placeholder(tf.int32, [None, max_seq_len, max_word_len], name="char_for_ids")
        self.char_rev_ids = tf.placeholder(tf.int32, [None, max_seq_len, max_word_len], name="char_rev_ids")
        self.word_lengths = tf.placeholder(tf.int32, [None, max_seq_len], name="char_pos_ids")
        self.tag_ids = tf.placeholder(tf.int32, [None, max_seq_len], name='tag_ids')
        self.cap_ids = tf.placeholder(tf.int32, [None, max_seq_len], name='cap_ids')
        self.pos_ids = tf.placeholder(tf.int32, [None, max_seq_len], name='pos_ids')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.cap_dim = cap_dim
        self.pos_dim = pos_dim
        self.char_bidirect = char_bidirect
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        inputs = []
        input_dim = 0
        with tf.device("/gpu:1"):
            if word_dim:
                word_embedding = tf.get_variable('word_embedding', [num_word, word_dim], initializer=initializer)
                word_embedded = tf.nn.embedding_lookup(word_embedding, self.word_ids, name="word_layer")
                inputs.append(word_embedded)
                print word_embedded.get_shape()
                input_dim += word_dim
            if char_dim:
                char_embedding = tf.get_variable('char_embedding', [num_char, char_dim], initializer=initializer)
                word_lengths = tf.reshape(self.word_lengths, [-1])
                with tf.variable_scope('char_forward_rnn'):

                    char_for_embedded = tf.reshape(
                        tf.nn.embedding_lookup(char_embedding, self.char_for_ids),
                        [-1, max_word_len, char_dim]
                    )
                    char_for_state = self.rnn(char_for_embedded, char_rnn_dim, word_lengths)
                    char_for_out = tf.reshape(char_for_state, [-1, max_seq_len, char_rnn_dim])

                    inputs.append(char_for_out)
                    input_dim += char_rnn_dim
                if char_bidirect:
                    with tf.variable_scope('char_backward_rnn'):
                        char_rev_embedded = tf.reshape(
                            tf.nn.embedding_lookup(char_embedding, self.char_rev_ids),
                            [-1, max_word_len, char_dim]
                        )
                        char_rev_state = self.rnn(char_rev_embedded, char_rnn_dim, word_lengths)
                        char_rev_out = tf.reshape(char_rev_state, [-1, max_seq_len, char_rnn_dim])

                        inputs.append(char_rev_out)
                        input_dim += char_rnn_dim
            if cap_dim:
                cap_embedding = tf.get_variable('cap_embedding', [num_cap, cap_dim], initializer=initializer)
                cap_embedded = tf.nn.embedding_lookup(cap_embedding, self.cap_ids, name="cap_layer")
                inputs.append(cap_embedded)
                input_dim += cap_dim
            if pos_dim:
                pos_embedding = tf.get_variable('pos_embedding', [num_pos, pos_dim], initializer=initializer)
                pos_embedded = tf.nn.embedding_lookup(pos_embedding, self.pos_ids, name='pos_layer')
                inputs.append(pos_embedded)
                input_dim += pos_dim

            inputs = tf.concat(2, inputs)

            inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)

            with tf.variable_scope('forward_rnn'):
                word_for_output = self.rnn(inputs, rnn_dim, None)

            if word_bidirect:
                inputs_rev = tf.reverse_sequence(inputs, self.seq_lengths, seq_dim=1, batch_dim=None)
                with tf.variable_scope('backward_rnn'):
                    word_rev_output = self.rnn(inputs_rev, rnn_dim, None)
                word_rev_output = tf.reverse_sequence(word_rev_output, self.seq_lengths, seq_dim=1, batch_dim=None)
                final_output = tf.concat(2, [word_for_output, word_rev_output])
                final_output = self.hidden_layer(final_output, 2*rnn_dim, rnn_dim, "tanh_layer", initializer, activation=tf.tanh)

            else:
                final_output = word_for_output

            self.tag_scores = self.hidden_layer(final_output, rnn_dim, num_tag, 'final_layer', initializer, activation=None) # [batch_size, seq_dim, num_tags]

            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time
            self.transitions = tf.get_variable("transitions", [num_tag, num_tag])
            log_likelihood, _ = crf.crf_log_likelihood(self.tag_scores, self.tag_ids, self.seq_lengths, self.transitions)
            self.loss = tf.reduce_mean(-log_likelihood)
            tvars = tf.trainable_variables()
            max_grad_norm = 5
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

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

    def hidden_layer(self, inputs, input_dim, out_dim, name, initializer, bias=True, activation=None):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [input_dim, out_dim], initializer=initializer)
            b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0))

            shape = inputs.get_shape().as_list()
            if len(shape) > 2:
                inputs = tf.reshape(inputs, [-1, input_dim])
            outputs = tf.matmul(inputs, w, name="hidden_linear")
            if bias:
                outputs = tf.add(outputs, b)
            if activation:
                outputs = activation(outputs)
            if len(shape) > 2:
                shape[0] = -1
                shape[-1] = out_dim
                outputs = tf.reshape(outputs, shape)
            return outputs

    def rnn(self, embedded_seq, rnn_dim, pos_ids):
        # embedded_seq: [batch_size, seq_dim, embed_dim]
        # pos_ids: [batch_size]
        # return: [batch_size, rnn_dim]

        # seq_len = embedded_seq.get_shape()[1]
        # embedded_list = [tf.squeeze(inp, [1]) for inp in tf.split(1, seq_len, embedded_seq)]
        # cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
        # outputs, states = tf.nn.rnn(cell, embedded_list, dtype=tf.float32)
        # outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])  # [batch_size, seq_dim, embed_dim]
        # if pos_ids!=None: # dynamic_rnn is alternative
        #     batch_size = embedded_seq.get_shape()[0]
        #     flattened_inputs = tf.reshape(embedded_seq, [-1, rnn_dim])
        #     flattened_indices = tf.range(batch_size) * seq_len + pos_ids
        #     return tf.gather(flattened_inputs, flattened_indices)
        #
        # else:
        #     return outputs
        cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
        print "[rnn embedded_seq]", embedded_seq.get_shape()
        outputs, state = tf.nn.dynamic_rnn(cell, inputs=embedded_seq, sequence_length=pos_ids, dtype=embedded_seq.dtype)
        if pos_ids != None:
            return state
        else:
            return outputs  # [batch_size, max_time, cell.output_size].

    def fit(self, tag_ids, seq_lengths, word_ids, char_for_ids, char_rev_ids, word_lengths, cap_ids, pos_ids, dropout_keep_prob):
        feed_dict = dict()
        feed_dict[self.seq_lengths] = seq_lengths
        feed_dict[self.tag_ids] = tag_ids
        feed_dict[self.dropout_keep_prob] = dropout_keep_prob
        if self.word_dim:
            feed_dict[self.word_ids] = word_ids
        if self.char_dim:
            feed_dict[self.char_for_ids] = char_for_ids
            feed_dict[self.word_lengths] = word_lengths
            if self.char_bidirect:
                feed_dict[self.char_rev_ids] = char_rev_ids
        if self.cap_dim:
            feed_dict[self.cap_ids] = cap_ids
        if self.pos_dim:
            feed_dict[self.pos_ids] = pos_ids
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss

    def predict(self, seq_lengths, word_ids, char_for_ids, char_rev_ids, word_lengths, cap_ids, pos_ids):
        feed_dict = {}
        feed_dict[self.seq_lengths] = seq_lengths
        feed_dict[self.dropout_keep_prob] = 1
        if self.word_dim:
            feed_dict[self.word_ids] = word_ids
        if self.char_dim:
            feed_dict[self.char_for_ids] = char_for_ids
            feed_dict[self.word_lengths] = word_lengths
            if self.char_bidirect:
                feed_dict[self.char_rev_ids] = char_rev_ids
        if self.cap_dim:
            feed_dict[self.cap_ids] = cap_ids
        if self.pos_dim:
            feed_dict[self.pos_ids] = pos_ids
        tag_scores, transitions = self.session.run([self.tag_scores, self.transitions], feed_dict)
        batch_viterbi_sequence = []
        for tag_score_, seq_length_ in zip(tag_scores, seq_lengths):
            # Remove padding from scores and tag sequence.
            tag_score_ = tag_score_[:seq_length_]

            # Compute the highest scoring sequence.
            viterbi_sequence, _ = crf.viterbi_decode(tag_score_, transitions)
            batch_viterbi_sequence.append(viterbi_sequence)
        return batch_viterbi_sequence

    def save(self, save_path):
        return self.saver.save(self.session, save_path)
    # def evaluate(self, seq_lengths, word_ids, char_for_ids, char_rev_ids, char_pos_ids, cap_ids, tag_ids):
    #     viterbi_sequences = self.predict(seq_lengths, word_ids, char_for_ids, char_rev_ids, char_pos_ids, cap_ids)
    #     correct_labels = 0
    #     total_labels = 0
    #     for tag_ids_, viterbi_sequence_ in zip(tag_ids, viterbi_sequences):
    #         # Evaluate word-level accuracy.
    #         correct_labels += np.sum(np.equal(viterbi_sequence_, tag_ids_))
    #         total_labels += sequence_length_

