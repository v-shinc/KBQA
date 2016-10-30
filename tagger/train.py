import os
import json
import sys
from collections import OrderedDict
from model import DeepCRF
from data_helper import DataSet
import tensorflow as tf

tf.app.flags.DEFINE_string("fn_train", "", "Train set location.")
tf.app.flags.DEFINE_string("fn_dev", "", "Name of directory to save model.")
tf.app.flags.DEFINE_string("fn_word", "", "Word list location.")
tf.app.flags.DEFINE_string("fn_char", "", "Character list location.")
tf.app.flags.DEFINE_string("dir_name", "", "Dev set location.")
tf.app.flags.DEFINE_int("max_seq_len", 20, "Max sentence length.")
tf.app.flags.DEFINE_int("max_word_len", 10, "Max number of character in a word.")
tf.app.flags.DEFINE_string("tag_scheme", "iobes", "Tagging scheme (IOB or IOBES).")
tf.app.flags.DEFINE_int("char_dim", 25, "Character embedding dimension.")
tf.app.flags.DEFINE_int("char_rnn_dim", 25, "Char RNN hidden layer size.")
tf.app.flags.DEFINE_boolean("char_bidirect", False, "Use a bidirectional RNN for chars.")
tf.app.flags.DEFINE_int("word_dim", 50, "Token embedding dimension.")
tf.app.flags.DEFINE_int("word_rnn_dim", 50, "Token LSTM hidden layer size.")
tf.app.flags.DEFINE_boolean("word_bidirect", False, "Use a bidirectional RNN for words.")
tf.app.flags.DEFINE_int("cap_dim", 0, "Capitalization feature dimension (0 to disable).")
tf.app.flags.DEFINE_float("dropout_keep_rate", 0.5, "Droupout keep rate on the input (1 = no dropout).")
tf.app.flags.DEFINE_boolean("reload", False, "Reload the last saved model.")
tf.app.flags.DEFINE_int("num_epoch", 50, "Number of training epoch.")
tf.app.flags.DEFINE_int("batch_size", 50, "Batch size to use during training.")
FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':

    # Check parameters validity
    assert FLAGS.fn_train and os.path.isfile(FLAGS.fn_train)
    assert FLAGS.char_dim > 0 or FLAGS.word_dim > 0
    assert 0. <= FLAGS.dropout < 1.0
    assert FLAGS.tag_scheme in ['iob', 'iobes']
    assert FLAGS.dir_name
    assert FLAGS.num_epoch > 0
    if FLAGS.fn_dev:
        assert os.path.isfile(FLAGS.fn_dev)

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.dir_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    dev_res_path = os.path.join(out_dir, 'dev.res')
    log_path = os.path.join(out_dir, 'train.log')
    config_path = os.path.join(out_dir, FLAGS.dir_name + '_config.json')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Parse parameter
    parameters = OrderedDict()
    parameters['max_seq_len'] = FLAGS.max_seq_len
    parameters['max_word_len'] = FLAGS.max_word_len
    parameters['char_dim'] = FLAGS.char_dim
    parameters['char_rnn_dim'] = FLAGS.char_lstm_dim
    parameters['char_bidirect'] = FLAGS.char_bidirect == 1
    parameters['word_dim'] = FLAGS.word_dim
    parameters['word_rnn_dim'] = FLAGS.word_lstm_dim
    parameters['word_bidirect'] = FLAGS.word_bidirect == 1
    parameters['cap_dim'] = FLAGS.cap_dim
    if FLAGS.reload == 1:
        parameters['load_path'] = save_path
    else:
        parameters['load_path'] = None

    dataset = DataSet(FLAGS.fn_word, FLAGS.fn_char, parameters)
    parameters['num_word'] = dataset.num_word
    parameters['num_char'] = dataset.num_char
    parameters['num_cap'] = dataset.num_cap
    parameters['num_tag'] = dataset.num_tag

    model = DeepCRF(**parameters)

    with open(config_path, 'w') as fout:
        print >> fout, json.dumps(parameters)

    lno = 0
    total_loss = 0.
    for epoch_index in xrange(FLAGS.num_epoch):
        iterator = dataset.batch_iterator(FLAGS.fn_train, FLAGS.batch_size)
        for data in iterator:
            if lno % 1000 == 0:
                sys.stdout.write("Process to %d\r" % lno)
                sys.stdout.flush()
            lno += FLAGS.batch_size
            loss = model.fit(
                data['tag_ids'],
                data['sentence_lengths'],
                data['word_ids'],
                data['char_for_ids'],
                data['char_rev_ids'],
                data['word_lengths'],
                data['cap_ids'],
                data['dropout_keep_prob']
            )
            total_loss += loss



