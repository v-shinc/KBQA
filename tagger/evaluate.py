import sys
import os
import numpy as np
from data_helper import DataSet
from model import DeepCRF
import json
def compute_f1(gold_set, predict_set):
    if not isinstance(gold_set, set):
        gold_set = set(gold_set)
    if not isinstance(predict_set, set):
        predict_set = set(predict_set)
    if len(gold_set) == 0:
        return 0, 1, 0
    if len(predict_set) == 0:
        return 1, 0, 0

    same = gold_set.intersection(predict_set)
    if len(same) == 0:
        return 0, 0, 0
    recall = len(same) * 1.0 / len(gold_set)
    precision = len(same) * 1.0 / len(predict_set)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def evaluate(dataset, model, fn_dev, fn_res):
    lno = 0
    correct_labels = 0
    total_labels = 0
    batch_size = 5
    avg_precision = 0.
    avg_recall = 0.
    avg_f1 = 0.
    num = 0
    res_file = None
    if fn_res:
        res_file = open(fn_res, 'w')
    for data in dataset.batch_iterator(fn_dev, batch_size):
        if lno % 100 == 0:
            sys.stdout.write("Process to %d\r" % lno)
            sys.stdout.flush()
        lno += batch_size
        viterbi_sequences = model.predict(
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
            data['pos_ids']
        )

        for i in xrange(len(viterbi_sequences)):

            seq_len = data['sentence_lengths'][i]
            y_ = data['tag_ids'][i][:seq_len]
            words = data['words'][i][:seq_len]
            gold_entities, gold_tag_sequence = dataset.get_named_entity_from_words(words, y_)
            correct_labels += np.sum(np.equal(viterbi_sequences[i], y_))
            total_labels += seq_len

            viterbi_sequence_ = viterbi_sequences[i]
            pred_entities, pred_tag_sequence = dataset.get_named_entity_from_words(words, viterbi_sequence_)
            precision, recall, f1 = compute_f1(gold_entities, pred_entities)
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            num += 1
            if res_file:
                tag_res = " ".join(["%s|%s|%s" % (w, p, g) for w, p, g in
                                    zip(words, gold_tag_sequence, pred_tag_sequence)])
                print >> res_file, json.dumps(
                    {'tag_res': tag_res,
                     'gold': "\t".join(gold_entities),
                     'predict': "\t".join(pred_entities),
                     'entity': data['entities'][i],
                     'pos': dataset.pos_ids_to_words(data['pos_ids'][i][:seq_len])
                     }, encoding='utf8', )

    accuracy = 100 * correct_labels / float(total_labels)
    avg_f1 /= num
    avg_precision /= num
    avg_recall /= num
    if (avg_recall + avg_precision) == 0:
        new_f1 = 0
    else:
        new_f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)
    res_info = ""
    res_info += "Word-level accuracy: %s\n" % accuracy
    res_info += "Average precision: %s\n" % avg_precision
    res_info += "Average recall: %s\n" % avg_recall
    res_info += "Average f1 score over all sentence: %s\n" % avg_f1
    res_info += "F1 of average recall and precision: %s\n" % new_f1
    print res_info
    return accuracy, avg_precision, avg_recall, avg_f1, new_f1, res_info


def evaluate_top2(dataset, model, fn_dev, fn_res):
    lno = 0
    correct_labels = 0
    total_labels = 0
    batch_size = 5
    avg_precision = 0.
    avg_recall = 0.
    avg_f1 = 0.
    num = 0
    res_file = None
    if fn_res:
        res_file = open(fn_res, 'w')
    for data in dataset.batch_iterator(fn_dev, batch_size):
        if lno % 100 == 0:
            sys.stdout.write("Process to %d\r" % lno)
            sys.stdout.flush()
        lno += batch_size
        batch_viterbi_sequences, batch_scores = model.predict_topk(
            data['sentence_lengths'],
            data['word_ids'],
            data['char_for_ids'],
            data['char_rev_ids'],
            data['word_lengths'],
            data['cap_ids'],
            data['pos_ids']
        )

        for i in xrange(len(batch_viterbi_sequences)):

            seq_len = data['sentence_lengths'][i]
            y_ = data['tag_ids'][i][:seq_len]
            words = data['words'][i][:seq_len]
            gold_entities, gold_tag_sequence = dataset.get_named_entity_from_words(words, y_)
            correct_labels += np.sum(np.equal(batch_viterbi_sequences[i][0], y_))
            total_labels += seq_len

            all_pred_entities = set()
            all_pred_tag_sequence = []
            for k in range(2):
                if k == 1 and batch_scores[i][1] * 1.0 / batch_scores[i][0] < 0.95:
                    break
                viterbi_sequence_ = batch_viterbi_sequences[i][k]
                pred_entities, pred_tag_sequence = dataset.get_named_entity_from_words(words, viterbi_sequence_)
                all_pred_entities.update(pred_entities)
                all_pred_tag_sequence.append(pred_tag_sequence)
            precision, recall, f1 = compute_f1(gold_entities, all_pred_entities)
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            num += 1
            if res_file:
                tag_res = " ".join(["%s|%s|%s" % (w, p, g) for w, p, g in
                          zip(words, gold_tag_sequence, all_pred_tag_sequence[0])])
                print >> res_file, json.dumps(
                    {'tag_res': tag_res,
                     'gold': "\t".join(gold_entities),
                     'predict': "\t".join(all_pred_entities),
                     'entity': data['entities'][i],
                     'pos': dataset.pos_ids_to_words(data['pos_ids'][i][:seq_len])
                     }, encoding='utf8',)


    accuracy = 100 * correct_labels / float(total_labels)
    avg_f1 /= num
    avg_precision /= num
    avg_recall /= num
    if (avg_recall + avg_precision) == 0:
        new_f1 = 0
    else:
        new_f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)
    res_info = ""
    res_info += "Word-level accuracy: %s\n" % accuracy
    res_info += "Average precision: %s\n" % avg_precision
    res_info += "Average recall: %s\n" % avg_recall
    res_info += "Average f1 score over all sentence: %s\n" % avg_f1
    res_info += "F1 of average recall and precision: %s\n" % new_f1
    print res_info
    return accuracy, avg_precision, avg_recall, avg_f1,new_f1, res_info

if __name__ == '__main__':
    dir_path = sys.argv[1]
    fn_dev = sys.argv[2]
    if len(sys.argv) == 3:
        res_name = 'test.res'
    else:
        res_name = sys.argv[3]
    dir_path = os.path.abspath(dir_path)
    checkpoint_dir = os.path.join(dir_path, "checkpoints")
    save_path = os.path.join(checkpoint_dir, "model")

    config_path = os.path.join(dir_path, 'config.json')
    parameters = json.load(open(config_path))
    dataset = DataSet(parameters)
    model = DeepCRF(
        parameters['max_sentence_len'],
        parameters['max_word_len'],
        parameters['char_dim'],
        parameters['char_rnn_dim'],
        parameters['char_bidirect'] == 1,
        parameters['word_dim'],
        parameters['word_rnn_dim'],
        parameters['word_bidirect'] == 1,
        parameters['cap_dim'],
        parameters['pos_dim'],
        save_path,
        parameters['num_word'],
        parameters['num_char'],
        parameters['num_cap'],
        parameters['num_pos'],
        parameters['num_tag']
    )
    fn_res = os.path.join(dir_path, res_name)
    evaluate(dataset, model, fn_dev, fn_res)