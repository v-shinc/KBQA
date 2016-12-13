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