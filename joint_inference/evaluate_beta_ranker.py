import sys
import os
import json
from data_helper import DataSet
from beta_ranker import BetaRanker

def evaluate(dataset, model, fn_dev, fn_res):
    lno = 0
    average_candidate_count = 0
    count = 0
    res_file = None
    average_f1 = 0.
    average_rank = 0. # best query pattern rank
    if fn_res:
        res_file = open(fn_res, 'w')
    for data in dataset.test_iterator(fn_dev):
        if lno % 100 == 0:
            sys.stdout.write("Process to %d\r" % lno)
            sys.stdout.flush()
        lno += 1

        scores = model.predict(
            data['pattern_word_ids'],
            data['sentence_lengths'],
            None,  # TODO: support pattern char-based feature
            None,
            data['relation_ids'],
            data['mention_char_ids'],
            data['topic_char_ids'],
            data['extras']
        )

        scores = scores.tolist()
        rank = sorted(zip(scores, data['paths']), key=lambda x: x[0], reverse=True)
        print >> res_file, data['question']
        for i in xrange(len(scores)):
            print >> res_file, rank[i][1], rank[i][0]
        best_index = -100000
        best_predicted_score = -1
        for i in xrange(len(scores)):
            if scores[i] > best_predicted_score:
                best_index = i
                best_predicted_score = scores[i]

        average_f1 += data['f1'][best_index]
        count += 1
        gold_index = -100000
        gold_score = -1
        for i in xrange(len(data['f1'])):
            if data['f1'][i] > gold_score:
                gold_score = data['f1'][i]
                gold_index = i
        rank_index = 1
        for s in scores:
            if s > scores[gold_index]:
                rank_index += 1
        average_rank += rank_index

        average_candidate_count += len(scores)
        if res_file:
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            print >> res_file, data['question']
            for j in indices:
                print >> res_file, scores[j], data['f1'][j], data['pattern_words'][j], data['mentions'][j], data['paths'][j], ' '.join(map(str, data['extras'][j]))
            print >> res_file
    average_f1 *= 1.0 / count
    average_candidate_count *= 1.0 / count
    average_rank = average_rank * 1.0 / count
    res_info = "F1 : {}\n".format(average_f1)
    res_info += "Number of test case: {} \nAverage rank: {}\nAverage number of candidates: {}"\
        .format(count, average_rank, average_candidate_count)
    if res_file:
        print >> res_file, res_info
    print res_info
    return average_f1, average_rank, average_candidate_count, res_info

if __name__ == '__main__':
    dir_path = sys.argv[1]  # model dir path
    fn_dev = sys.argv[2]    # test file path
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
    model = BetaRanker(
        parameters
    )
    fn_res = os.path.join(dir_path, res_name)
    evaluate(dataset, model, fn_dev, fn_res)
