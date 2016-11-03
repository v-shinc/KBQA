import sys
import numpy as np

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
    batch_size = 1
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
        )
        for i in xrange(len(viterbi_sequences)):
            seq_len = data['sentence_lengths'][i]
            y_ = data['tag_ids'][i][:seq_len]
            viterbi_sequence_ = viterbi_sequences[i]
            # Evaluate word-level accuracy
            correct_labels += np.sum(np.equal(viterbi_sequence_, y_))
            total_labels += seq_len
            word_ids = data['word_ids'][i][:seq_len]
            pred_entities, sentence, pred_tag_sequence = dataset.get_named_entity(word_ids, viterbi_sequence_)
            gold_entities, _, gold_tag_sequence = dataset.get_named_entity(word_ids, y_)
            print gold_entities, pred_entities
            precision, recall, f1 = compute_f1(gold_entities, pred_entities)
            print precision, recall, f1
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            num += 1
            if res_file:
                print >> res_file, (" ".join(["%s|%s|%s" % (w, p, g)for w, p, g in
                                             zip(sentence, gold_tag_sequence, pred_tag_sequence)])).encode('utf8')
                print >> res_file, ("gold entities: %s" % " ".join(gold_entities)).encode('utf8')
                print >> res_file, ("predicted entities: %s" % " ".join(pred_entities)).encode('utf8')

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
    if res_file:
        print >> res_file
    print res_info
    return accuracy, avg_precision, avg_recall, avg_f1,new_f1