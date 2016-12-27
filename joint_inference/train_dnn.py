import os
import json
import sys
import tensorflow as tf
from time import time
import numpy as np
import dnn_model_config
import dnn_classifier
import evaluate_dnn
flags = tf.flags
flags.DEFINE_string("config_name", "", "Configuration name")
FLAGS = flags.FLAGS


class DataSet:
    def __init__(self, feature_keys):
        self.feature_keys = set(feature_keys)

    def batch_iterator(self, fn_train, batch_size, include_info=False):
        num = 0
        with open(fn_train) as fin:
            for _ in fin:
                num += 1
        index = 0
        num_batch = num // batch_size + int(num % batch_size > 0)
        fin = open(fn_train)
        for _ in xrange(num_batch):
            features = []
            labels = []
            question_ids = []
            answers = []
            details = []
            while len(features) < batch_size:
                if index == num:
                    fin.seek(0)
                    index = 0
                index += 1
                line = fin.readline()
                data = json.loads(line)
                feature = []
                for f in self.feature_keys:
                    feature.append(data.get(f, 0))

                labels.append(data['label'])
                question_ids.append(data['qid'])
                features.append(feature)
                if include_info:
                    answers.append(data['answer'])
                    details.append(data)
            ret = {
                "features": np.array(features, dtype=np.float32),
                "labels": np.array(labels, dtype=np.int32),
                "question_ids": question_ids
            }
            if include_info:
                ret['answers'] = answers
                ret['details'] = details
            yield ret


if __name__ == '__main__':
    assert FLAGS.config_name and FLAGS.config_name in dnn_model_config.configuration
    config = dnn_model_config.configuration[FLAGS.config_name]
    assert "features" in config
    assert "reload" in config
    assert "layer_dims" in config
    assert "lr" in config
    assert "dropout_keep_prob" in config

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

    dataset = DataSet(config['features'])

    model = dnn_classifier.DNNClassifier(config['layer_dims'], 2, config['lr'], config['load_path'])

    fout_log = open(log_path, 'a')
    with open(config_path, 'w') as fout:
        print >> fout, json.dumps(config)

    best_accuracy = 0
    if "fn_dev" in config:
        accuracy, total_precision, total_recall, total_f1, new_f1, eval_info = \
            evaluate_dnn.evaluate(dataset, model, config['fn_dev'], dev_res_path)
        best_accuracy = accuracy
        print >> fout_log, eval_info

    for epoch_index in xrange(config['num_epoch']):
        tic = time()
        lno = 0
        total_loss = 0.
        for data in dataset.batch_iterator(config['fn_train'], config['batch_size']):
            if lno % 1000 == 0:
                sys.stdout.write("Process to %d\r" % lno)
                sys.stdout.flush()
            lno += config['batch_size']
            loss = model.fit(
                data['features'],
                data['labels'],
                config['dropout_keep_prob']
            )
            total_loss += loss
        info = '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
        print info
        print >> fout_log, info

        old_path = model.save("%s-%s" % (save_path, epoch_index))
        if config['fn_dev']:
            accuracy, total_precision, total_recall, total_f1, new_f1, eval_info = \
                evaluate_dnn.evaluate(dataset, model, config['fn_dev'], dev_res_path if epoch_index % 10 == 0 else None)
            print >> fout_log, eval_info
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                os.rename(old_path, save_path)
                os.rename('%s.meta' % old_path, '%s.meta' % save_path)
                print "best mode", old_path