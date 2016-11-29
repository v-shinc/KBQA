import os
import json
import sys
from time import time
from collections import OrderedDict
from model import DeepCRF
from data_helper import DataSet
from evaluate import evaluate
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("fn_train", "", "Train set location.")
flags.DEFINE_string("fn_dev", "", "Dev set location.")
flags.DEFINE_string("fn_word", "", "Word list location.")
flags.DEFINE_string("fn_char", "", "Character list location.")
flags.DEFINE_string("fn_pos", "", "Part of speech tag list location.")
flags.DEFINE_string("dir_name", "", "Name of directory to save model.")
flags.DEFINE_integer("max_sentence_len", 36, "Max sentence length.")
flags.DEFINE_integer("max_word_len", 22, "Max number of character in a word.")
flags.DEFINE_string("tag_scheme", "iobes", "Tagging scheme (IOB or IOBES).")
flags.DEFINE_integer("char_dim", 25, "Character embedding dimension.")
flags.DEFINE_integer("char_rnn_dim", 25, "Char RNN hidden layer size.")
flags.DEFINE_boolean("char_bidirect", True, "Use a bidirectional RNN for chars.")
flags.DEFINE_integer("word_dim", 50, "Token embedding dimension.")
flags.DEFINE_integer("word_rnn_dim", 50, "Token LSTM hidden layer size.")
flags.DEFINE_boolean("word_bidirect", True, "Use a bidirectional RNN for words.")
flags.DEFINE_integer("cap_dim", 0, "Capitalization feature dimension (0 to disable).")
flags.DEFINE_integer("pos_dim", 0, "Part of speech feature dimension (0 to disable).")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Droupout keep rate on the input (1 = no dropout).")
flags.DEFINE_boolean("reload", False, "Reload the last saved model.")
flags.DEFINE_integer("num_epoch", 50, "Number of training epoch.")
flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
FLAGS = flags.FLAGS


if __name__ == '__main__':
    # Check parameters validity
    assert FLAGS.fn_train and os.path.isfile(FLAGS.fn_train)
    assert FLAGS.char_dim >= 0 or FLAGS.word_dim >= 0
    assert 0. <= FLAGS.dropout_keep_prob < 1.0
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
    config_path = os.path.join(out_dir, 'config.json')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Parse parameter
    parameters = OrderedDict()
    parameters['max_sentence_len'] = FLAGS.max_sentence_len
    parameters['max_word_len'] = FLAGS.max_word_len
    parameters['char_dim'] = FLAGS.char_dim
    parameters['char_rnn_dim'] = FLAGS.char_rnn_dim
    parameters['char_bidirect'] = FLAGS.char_bidirect == 1
    parameters['word_dim'] = FLAGS.word_dim
    parameters['word_rnn_dim'] = FLAGS.word_rnn_dim
    parameters['word_bidirect'] = FLAGS.word_bidirect == 1
    parameters['cap_dim'] = FLAGS.cap_dim
    parameters['pos_dim'] = FLAGS.pos_dim
    parameters['dropout_keep_prob'] = FLAGS.dropout_keep_prob
    if FLAGS.reload == 1:
        parameters['load_path'] = save_path
    else:
        parameters['load_path'] = None
    parameters['tag_scheme'] = FLAGS.tag_scheme
    parameters['fn_word'] = os.path.abspath(os.path.join(os.path.curdir, FLAGS.fn_word))
    parameters['fn_char'] = os.path.abspath(os.path.join(os.path.curdir, FLAGS.fn_char))
    if FLAGS.pos_dim:
        parameters['fn_pos'] = os.path.abspath(os.path.join(os.path.curdir, FLAGS.fn_pos))
    dataset = DataSet(parameters)
    parameters['num_word'] = dataset.num_word
    parameters['num_char'] = dataset.num_char
    parameters['num_cap'] = dataset.num_cap
    parameters['num_tag'] = dataset.num_tag
    parameters['num_pos'] = dataset.num_pos

    model = DeepCRF(
        FLAGS.max_sentence_len,
        FLAGS.max_word_len,
        FLAGS.char_dim,
        FLAGS.char_rnn_dim,
        FLAGS.char_bidirect == 1,
        FLAGS.word_dim,
        FLAGS.word_rnn_dim,
        FLAGS.word_bidirect == 1,
        FLAGS.cap_dim,
        FLAGS.pos_dim,
        save_path if FLAGS.reload else None,
        dataset.num_word,
        dataset.num_char,
        dataset.num_cap,
        dataset.num_pos,
        dataset.num_tag)
    fout_log = open(log_path, 'a')

    with open(config_path, 'w') as fout:
        print >> fout, json.dumps(parameters)

    best_avg_f1 = 0.
    if FLAGS.fn_dev:
        accuracy, avg_precision, avg_recall, avg_f1, new_f1, eval_info = evaluate(dataset, model, FLAGS.fn_dev, dev_res_path)
        print >> fout_log, eval_info
        best_avg_f1 = avg_f1
    for epoch_index in xrange(FLAGS.num_epoch):
        tic = time()
        lno = 0
        total_loss = 0.
        for data in dataset.batch_iterator(FLAGS.fn_train, FLAGS.batch_size):
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
                data['pos_ids'],
                FLAGS.dropout_keep_prob
            )
            total_loss += loss
        info = '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
        print info
        print >> fout_log, info

        old_path = model.save("%s-%s" % (save_path, epoch_index))
        if FLAGS.fn_dev:
            accuracy, avg_precision, avg_recall, avg_f1, new_f1, eval_info = evaluate(dataset, model, FLAGS.fn_dev, dev_res_path)
            print >> fout_log, eval_info
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                os.rename(old_path, save_path)
                os.rename('%s.meta' % old_path, '%s.meta' % save_path)
                print "best mode", old_path


