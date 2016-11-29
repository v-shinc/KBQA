import sys
import os
import json
from data_helper import DataSet
from model import RelationMatcherModel

def evaluate(dataset, model, fn_dev, fn_res):
    lno = 0
    num_p_at_1 = 0
    average_rank_index = 0
    average_candidate_count = 0
    count = 0
    res_file = None
    if fn_res:
        res_file = open(fn_res, 'w')
    for data in dataset.test_iterator(fn_dev):
        if lno % 100 == 0:
            sys.stdout.write("Process to %d\r" % lno)
            sys.stdout.flush()
        lno += 1
        scores = model.predict(
            data['word_ids'],
            data['sentence_lengths'],
            data['char_ids'],
            data['word_lengths'],
            data['relation_ids'],
        )
        count += 1
        pos_score = 0
        num_pos = data['num_pos']
        for i in xrange(data['num_pos']):
            if scores[i] > pos_score:
                pos_score = scores[i]

        rank_index = 1
        for i in xrange(data['num_pos'], len(scores)):
            if scores[i] > pos_score:
                rank_index += 1
        if rank_index == 1:
            num_p_at_1 += 1
        average_rank_index += rank_index
        average_candidate_count += len(scores)
        if res_file:
            pos_relation_score = []
            rank_list = sorted(zip(data['relations'], scores), key=lambda x: x[1], reverse=True)
            for i in range(num_pos):
                score = scores[i]
                rank_index = 1
                for s in scores:
                    if s > score:
                        rank_index += 1
                pos_relation_score.append([data['relations'][i], rank_index])
            # rank_list_str = ''.join(["%s: %s" % (r, s) for r, s in rank_list])
            print >> res_file, '{0}\t{1}\t{2}'.format(data['words'], pos_relation_score, rank_list).encode('utf8')

    p_at_1 = num_p_at_1 * 1.0 / count
    average_candidate_count *= 1.0 / count
    average_rank = average_rank_index * 1.0 / count
    res_info = "Number of test case: {} \nP@1: {}\nAverage rank: {}\nAverage number of candidates: {}"\
        .format(count, p_at_1, average_rank, average_candidate_count)
    if res_file:
        print >> res_file, res_info
    print res_info
    return p_at_1, average_rank, average_candidate_count, res_info

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
    model = RelationMatcherModel(
        parameters
    )
    fn_res = os.path.join(dir_path, res_name)
    evaluate(dataset, model, fn_dev, fn_res)
