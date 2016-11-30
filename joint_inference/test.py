import sys
import json
from pipeline import BetaAnswer


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


def basic_exp(fn_test, fn_res):
    # only use entity linking module and relation match module
    ba = BetaAnswer()
    webquestion = json.load(open(fn_test))
    gold, pred = [], []
    num_topic_solved = 0
    avg_num_topic = 0
    with open(fn_res, 'w') as fout:
        for data in webquestion:
            print '**' * 20
            question = data['utterance']
            question, graphs = ba.parse(question)
            topic_set = set()
            for g in graphs:
                topic_set.add(g['topic'])
            if data['mid1'] in topic_set:
                num_topic_solved += 1
            avg_num_topic += len(topic_set)
            gold.append(set(data['mids'].values()))
            if len(graphs) == 0:
                pred.append(set())
            else:
                rank_list = sorted(graphs, key=lambda x: x['relation_score'], reverse=True)
                # relation_match_score = dict()
                # for r in rank_list:
                #     relation_match_score[(r['pattern'], r['relation'])] = r['relation_score']
                # relation_match_score = sorted(relation_match_score.items(), key=lambda x: x[1], reverse=True)
                # print 'correct topic:', data['mid1']
                # for r in relation_match_score:
                #     print r
                answers = set()
                best_score = rank_list[0]['relation_score']
                for entry in rank_list:
                    if best_score == entry['relation_score']:
                        answers.add(entry['answer'])
                    else:
                        break
                pred.append(answers)

                print >> fout, "{} {}".format(rank_list[0]['relation'], rank_list[0]['relation_score'])
            print >> fout, "correct relation {}".format([p[0] for p in data['paths']])
            print >> fout, "{} {}, {}".format(question, gold[-1], pred[-1]).encode('utf8')
        print "{} question find correct topic".format(num_topic_solved)
        print "average number of candidate topics = {}".format(avg_num_topic * 1.0 / num_topic_solved)

    # compute metric
    avg_precision = 0.
    avg_recall = 0.
    avg_f1 = 0.
    num = 0
    for g, p in zip(gold, pred):
        precision, recall, f1 = compute_f1(g, p)
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
        num += 1

    avg_f1 /= num
    avg_precision /= num
    avg_recall /= num
    if (avg_recall + avg_precision) == 0:
        new_f1 = 0
    else:
        new_f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision)
    res_info = ""
    res_info += "Average precision: %s\n" % avg_precision
    res_info += "Average recall: %s\n" % avg_recall
    res_info += "Average f1 score over all sentence: %s\n" % avg_f1
    res_info += "F1 of average recall and precision: %s\n" % new_f1
    print res_info


if __name__ == '__main__':
    basic_exp("../data/wq.test.complete.v2", "tmp")

