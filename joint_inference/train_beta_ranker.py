
import os
import json
import sys
import tensorflow as tf
from time import time
from data_helper import DataSet
from beta_ranker import BetaRanker
from evaluate_beta_ranker import evaluate
import config_beta_ranker
flags = tf.flags

flags.DEFINE_string("config_name", "", "Configuration name")
FLAGS = flags.FLAGS

if __name__ == '__main__':
    assert FLAGS.config_name

    config = config_beta_ranker.configuration[FLAGS.config_name]

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.config_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    dev_res_path = os.path.join(out_dir, 'dev.res')
    log_path = os.path.join(out_dir, 'train.log')
    config_path = os.path.join(out_dir, 'config.json')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if config['reload']:
        config['load_path'] = save_path
    else:
        config['load_path'] = None

    dataset = DataSet(config)
    config["pattern_config"]['num_word'] = dataset.num_word
    config["pattern_config"]['num_char'] = dataset.num_char
    config['relation_config']['num_word'] = dataset.num_relation
    config['relation_config']['num_char'] = dataset.num_char
    config['topic_config']['num_word'] = dataset.num_char  # this is char-based
    model = BetaRanker(config)

    fout_log = open(log_path, 'a')
    with open(config_path, 'w') as fout:
        print >> fout, json.dumps(config)

    best_p_at_1 = 0
    if "fn_dev" in config:
        p_at_1, average_rank, num_avg_candidates, eval_info = \
            evaluate(dataset, model, config['fn_dev'], dev_res_path)
        best_p_at_1 = p_at_1
        print >> fout_log, eval_info

    for epoch_index in xrange(config['num_epoch']):
        tic = time()
        lno = 0
        total_loss = 0.
        total_margin_loss = 0.
        total_reg_loss = 0.
        for data in dataset.train_batch_iterator(config['batch_size']):
            if lno % 1000 == 0:
                sys.stdout.write("Process to %d\r" % lno)
                sys.stdout.flush()
            lno += config['batch_size']
            
            loss, margin_loss, reg_loss = model.fit(
                data['pattern_word_ids'],
                data['sentence_lengths'],
                None,  # data['pattern_char_ids'],
                None,  # data['word_lengths'],
                data['relation_ids'],
                data['mention_char_ids'],
                data['topic_char_ids'],
                data['extras'],
                config['dropout_keep_prob']
            )
            total_loss += loss
            total_margin_loss += margin_loss
            total_reg_loss += reg_loss
        info = '# %s: loss = %s, margin_loss = %s, reg_loss = %s, it costs %ss' % (
            epoch_index, total_loss, total_margin_loss, total_reg_loss, time() - tic)
        print info
        print >> fout_log, info

        old_path = model.save("%s-%s" % (save_path, epoch_index))
        if config['fn_dev']:
            p_at_1, average_rank, num_avg_candidates, eval_info = \
                evaluate(dataset, model, config['fn_dev'], dev_res_path)
            print >> fout_log, eval_info
            if p_at_1 > best_p_at_1:
                best_p_at_1 = p_at_1
                os.rename(old_path, save_path)
                os.rename('%s.meta' % old_path, '%s.meta' % save_path)
                print "best mode", old_path
        if epoch_index % 3 == 0:
            print "Evaluation over training data"
            evaluate(dataset, model, config['fn_train'], dev_res_path)

