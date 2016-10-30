import tensorflow as tf
import numpy as np

class DeepCRF(object):
    def __init__(self, seq_dim, word_len, num_word, num_char, num_cap, num_tags,
                 char_dim, char_rnn_dim, char_bidirect,
                 word_dim, word_rnn_dim, word_bidirect, cap_dim, load_path=None):
        self.word_ids = tf.placeholder(tf.int32, [None, seq_dim], name="word_ids")
        self.seq_lengths = tf.placeholder(tf.int32, [None], name="seq_lengths")
        self.char_for_ids = tf.placeholder(tf.int32, [None, seq_dim, word_len], name="char_for_ids")
        self.char_rev_ids = tf.placeholder(tf.int32, [None, seq_dim, word_len], name="char_rev_ids")
        self.char_pos_ids = tf.placeholder(tf.int32, [None, seq_dim], name="char_pos_ids")
        self.tag_ids = tf.placeholder(tf.int32, [None, seq_dim])
        self.cap_ids = tf.placeholder(tf.int32, [None, seq_dim])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.cap_dim = cap_dim
        self.char_bidirect = char_bidirect
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        inputs = []
        input_dim = 0
        with tf.divice("/gpu:1"):
            if word_dim:
                word_embedding = tf.get_variable('word_embedding', [num_word, word_dim], initializer=initializer)
                word_embedded = tf.nn.embedding_lookup(word_embedding, self.word_ids, name="word_layer")
                inputs.append(word_embedded)
                input_dim += word_dim
            if char_dim:
                char_embedding = tf.get_variable('char_embedding', [num_char, char_dim], initializer=initializer)
                with tf.variable_scope('char_forward_rnn'):
                    char_for_embedded = tf.reshape(
                        tf.nn.embedding_lookup(char_embedding, self.char_for_ids),
                        [-1, word_len, char_dim]
                    )
                    char_for_out = tf.reshape(self.recurrentNN(char_for_embedded, char_rnn_dim, self.char_pos_ids), [-1, seq_dim, char_rnn_dim])
                    inputs.append(char_for_out)
                    input_dim += char_rnn_dim
                if char_bidirect:
                    with tf.variable_scope('char_backward_rnn'):
                        char_rev_embedded = tf.reshape(
                            tf.nn.embedding_lookup(char_embedding, self.char_rev_ids),
                            [-1, word_len, char_dim]
                        )
                        char_rev_out = tf.reshape(self.recurrentNN(char_rev_embedded, char_rnn_dim, self.char_pos_ids), [-1, seq_dim, char_rnn_dim])
                        inputs.append(char_rev_out)
                        input_dim += char_rnn_dim
            if cap_dim:
                cap_embedding = tf.get_variable('cap_embedding', [num_cap, cap_dim], initializer=initializer)
                cap_embedded = tf.nn.embedding_lookup(cap_embedding, cap_dim, name="cap_layer")
                inputs.append(cap_embedded)
                input_dim += cap_dim

            inputs = tf.concat(2, inputs)

            inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)

            with tf.variable_scope('forward_rnn'):
                word_for_output = self.recurrentNN(inputs, word_rnn_dim, None)
            if word_bidirect:
                inputs_rev = tf.reverse_sequence(inputs, self.seq_lengths, input_dim, batch_dim=None)
                with tf.variable_scope('backward_rnn'):
                    word_rev_output = self.recurrentNN(inputs_rev, word_rnn_dim, None)
                word_rev_output = tf.reverse_sequence(word_rev_output, self.seq_lengths, word_rnn_dim, batch_dim=None)
                final_output = tf.concat(2, [word_for_output, word_rev_output])
                final_output = self.hiddenLayer(final_output, 2*word_rnn_dim, word_rnn_dim, "tanh_layer", initializer, activation=tf.tanh)
            else:
                final_output = word_for_output

            tags_scores = self.hiddenLayer(final_output, word_rnn_dim, num_tags, 'final_layer', initializer, activation=None) # [batch_size, seq_dim, num_tags]

            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time
            log_likelihood, self.transitions, = tf.contrib.crf_log_likelihood(tags_scores, self.tag_ids, self.seq_lengths)
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


    def hiddenLayer(self, inputs, input_dim, out_dim, name, initializer, bias=True, activation=None):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [input_dim, out_dim], initializer)
            b = tf.get_variable('b', [1], tf.constant_initializer(0))
            linear_output = tf.matmul(inputs, w)
            if bias:
                linear_output = tf.add(linear_output, b)
            if not activation:
                return linear_output
            else:
                return activation(linear_output)

    def recurrentNN(self, embedded_seq, rnn_dim, pos_ids):
        # embedded_seq: [batch_size, seq_dim, embed_dim]
        # pos_ids: [batch_size]
        # return: [batch_size, rnn_dim]
        seq_len = embedded_seq.get_shape()[1]
        embedded_list = [tf.squeeze(inp, [1]) for inp in tf.split(1, seq_len, embedded_seq)]
        cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
        outputs, states = tf.nn.rnn(cell, embedded_list, dtype=tf.float32)
        outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])  # [batch_size, seq_dim, embed_dim]
        if pos_ids:
            batch_size = embedded_seq.get_shape()[0]
            flattened_inputs = tf.reshape(embedded_seq, [-1, rnn_dim])
            flattened_indices = tf.range(batch_size) * seq_dim + pos_ids
            return tf.gather(flattened_inputs, flattened_indices)

        else:
            return outputs

    def fit(self, seq_lengths, tag_ids, word_ids, char_for_ids, char_rev_ids, char_pos_ids, cap_ids, dropout_keep_prob):
        feed_dict = {}
        feed_dict[self.seq_lengths] = seq_lengths
        feed_dict[self.tag_ids] = tag_ids
        feed_dict[self.dropout_keep_prob] = dropout_keep_prob
        if self.word_dim:
            feed_dict[self.word_ids] = word_ids
        if self.char_dim:
            feed_dict[self.char_for_ids] = char_for_ids
            feed_dict[self.char_pos_ids] = char_pos_ids
        if self.char_bidirect:
            feed_dict[self.char_rev_ids] = char_rev_ids
        if self.cap_dim:
            feed_dict[self.cap_ids] = cap_ids

        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss


    def predict(self, seq_lengths, word_ids, char_for_ids, char_rev_ids, char_pos_ids, cap_ids):
        feed_dict = {}
        feed_dict[self.seq_lengths] = seq_lengths
        feed_dict[self.dropout_keep_prob] = 1
        if self.word_dim:
            feed_dict[self.word_ids] = word_ids
        if self.char_dim:
            feed_dict[self.char_for_ids] = char_for_ids
            feed_dict[self.char_pos_ids] = char_pos_ids
        if self.char_bidirect:
            feed_dict[self.char_rev_ids] = char_rev_ids
        if self.cap_dim:
            feed_dict[self.cap_ids] = cap_ids
        tag_scores, transitions = self.session.run([self.tag_ids, self.transitions], feed_dict)
        batch_viterbi_sequence = []
        for tag_score_, seq_length_ in zip(tag_scores, seq_lengths):
            # Remove padding from scores and tag sequence.
            tag_score_ = tag_score_[:seq_length_]

            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tag_score_, transitions)
            batch_viterbi_sequence.append(viterbi_sequence)
        return batch_viterbi_sequence

    # def evaluate(self, seq_lengths, word_ids, char_for_ids, char_rev_ids, char_pos_ids, cap_ids, tag_ids):
    #     viterbi_sequences = self.predict(seq_lengths, word_ids, char_for_ids, char_rev_ids, char_pos_ids, cap_ids)
    #     correct_labels = 0
    #     total_labels = 0
    #     for tag_ids_, viterbi_sequence_ in zip(tag_ids, viterbi_sequences):
    #         # Evaluate word-level accuracy.
    #         correct_labels += np.sum(np.equal(viterbi_sequence_, tag_ids_))
    #         total_labels += sequence_length_
import os
import json
def train(fn_train, dirname, load):
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dirname))
    print("Writing to {}".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    dev_res_path = os.path.join(out_dir, 'dev.res')
    log_path = os.path.join(out_dir, 'train.log')
    config_path = os.path.join(out_dir, dirname + '_config.json')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config = locals()
    with open(config_path, 'w') as fout:
        print >> fout, json.dumps(config)

    if load:
        load_path = save_path
    else:
        load_path = None

    fout_log = open(log_path, 'a')