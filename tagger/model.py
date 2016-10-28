import tensorflow as tf

class DeepCRF(object):
    def __init__(self, seq_len, word_len, num_word, num_char, word_dim, char_dim, num_cap, cap_dim):
        self.word_ids = tf.placeholder(tf.int32, [None, seq_len], name="word_ids")
        self.char_for_ids = tf.placeholder(tf.int32, [None, seq_len, word_len], name="char_for_ids")
        self.char_rev_ids = tf.placeholder(tf.int32, [None, seq_len, word_len], name="char_rev_ids")
        self.char_pos_ids = tf.placeholder(tf.int32, [None, seq_len], name="char_pos_ids")
        self.tag_ids = tf.placeholder(tf.int32, [None, seq_len])
        self.cap_ids = tf.placeholder(tf.int32, [None, seq_len])


        with tf.device("/gpu:1"):
            with tf.variable_scope('embedding'):
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
                word_embedding = tf.get_variable('word_embedding', [num_word, word_dim], initializer=initializer)
                char_embedding = tf.get_variable('char_embedding', [num_char, char_dim], initializer=initializer)
                cap_embedding = tf.get_variable('cap_embedding', [num_cap, cap_dim], initializer=initializer)


    def rnn(self, embedded_seq, rnn_dim):
        # embedded_seq: [batch_size, seq_len, embed_dim]
        seq_len = embedded_seq.get_shape()[1]
        embedded_list = [tf.squeeze(inp, [1]) for inp in tf.split(1, seq_len, embedded_seq)]
        cell = tf.nn.rnn_cell.GRUCell(rnn_dim)
        outputs, states = tf.nn.rnn(cell, embedded_list, dtype=tf.float32)

        return outputs[-1]  #