import sys
import json
from pipeline import Pipeline
from utils.string_utils import naive_split


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
    # only use entity linking module and relation matching module
    pipeline = Pipeline()
    webquestion = json.load(open(fn_test))
    gold, pred = [], []
    num_topic_solved = 0
    avg_num_topic = 0
    with open(fn_res, 'w') as fout:
        for data in webquestion:
            print '**' * 20
            question = data['utterance']


            question, features = pipeline.add_topic_feature(question)
            question, features = pipeline.add_path_feature(question, features)

            topic_set = set()
            for g in features:
                topic_set.add(g['topic'])
            if data['mid1'] in topic_set:
                num_topic_solved += 1
            avg_num_topic += len(topic_set)
            gold.append(set(data['mids'].values()))
            if len(features) == 0:
                pred.append(set())
            else:
                rank_list = sorted(features, key=lambda x: x['relation_score'], reverse=True)
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


def gen_data_for_relation_matcher(fn_webquestion_list, fn_simplequstion_list, fn_out):
    pipeline = Pipeline()
    # symbols = {"_", "_'s", "_'"}

    def map_word(x):
        if x.startswith('_') or x.endswith('_'):
            return x.replace('_', "<$>")
        else:
            return x

    with open(fn_out, 'w') as fout:
        # process simple question
        for fn in fn_simplequstion_list:
            with open(fn) as fin:
                for line in fin:
                    _, positive_relation, _, pattern, question = line.decode('utf8').strip().split('\t')
                    # question, candidate_relations = pipeline.gen_candidate_relations(question)
                    pattern = ' '.join(naive_split(' '.join([map_word(w) for w in pattern.split()])))
                    # negative_relations = candidate_relations - {positive_relation}
                    print >> fout, json.dumps({
                        "question": pattern,
                        "pos_relation": [positive_relation],
                        # "neg_relation": list(negative_relations)},
                        "neg_relation": []},
                        ensure_ascii=False).encode('utf8')
        # process webquestion
        for fn in fn_webquestion_list:
            webquestion = json.load(open(fn), encoding="utf8")
            for data in webquestion:
                positive_relations = set()
                for path in data['paths']:
                    if path[1] not in {"forward_pass_non_cvt", "forward_pass_cvt", "forward_direct"}:
                        raise ValueError('path type error')
                    if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                        positive_relations.add(path[0].split()[-2])
                if len(positive_relations) == 0:
                    continue
                pattern = ' '.join(naive_split(' '.join([map_word(w) for w in data['sentence'].split()])))

                question, candidate_relations = pipeline.gen_candidate_relations(data['utterance'])
                negative_relations = candidate_relations - positive_relations
                print >> fout, json.dumps({
                    "question": pattern,
                    "pos_relation": list(positive_relations),
                    "neg_relation": list(negative_relations)},
                    ensure_ascii=False).encode('utf8')


# def gen_query_graph(fn_wq_list, fn_simple_list, fn_out):
#     pipeline = Pipeline()
#     complete_qids = set()
#     qids = set()
#     with open(fn_out, 'w') as fout:
#         # process webquestion
#         for fn in fn_wq_list:
#             webq = json.load(open(fn), encoding="utf8")
#             for data in webq:
#
#                 positive_relations = set()
#                 for path in data['paths']:
#                     if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
#                         positive_relations.add(path[0].split()[-2])
#
#                 if len(positive_relations) == 0:
#                     continue
#                 qids.add(data['id'])
#                 question, candidate_graphs = pipeline.gen_candidate_query_graph(data['utterance'])
#                 gold_answers = set(data['mids'].values())
#                 for g in candidate_graphs:
#                     if g['topic'] == data['mid1'] and g['answer'] in gold_answers and g['relation'] in positive_relations:
#                         g['label'] = 1
#                     else:
#                         g['label'] = 0
#                     g['question'] = question
#                     g['qid'] = data['id']
#                     if g['label'] == 1:
#                         complete_qids.add(data['id'])
#                     print >> fout, json.dumps(g, ensure_ascii=False).encode('utf8')
#     print "total valid question", len(qids)
#     print "complete question", len(complete_qids)

def gen_query_graph(fn_wq_list, fn_simple_list, fn_out):
    pipeline = Pipeline()
    complete_qids = set()
    qids = set()
    with open(fn_out, 'w') as fout:
        qid = 0

        # Process WEBQUESTION
        for fn in fn_wq_list:
            webq = json.load(open(fn), encoding="utf8")
            for data in webq:
                qid += 1
                positive_relations = set()
                for path in data['paths']:
                    if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                        positive_relations.add(path[0].split()[-2])
                if len(positive_relations) == 0:
                    continue
                qids.add(qid)
                question, query_graphs = pipeline.gen_candidate_query_graph(data['utterance'])
                for j in xrange(len(query_graphs)):
                    query_graphs[j]['qid'] = qid
                gold_answers = set(data['mids'].values())
                query_patterns = pipeline.extract_query_pattern_and_f1(query_graphs, gold_answers)

                # Just for statistic
                for j in xrange(len(query_graphs)):
                    if query_graphs[j]['topic'] == data['mid1'] and query_graphs[j]['answer'] in gold_answers and \
                                    query_graphs[j]['relation'] in positive_relations:
                        query_graphs[j]['label'] = 1
                    else:
                        query_graphs[j]['label'] = 0
                    query_graphs[j]['question'] = question
                    # query_graphs[j]['qid'] = data['id']
                    if query_graphs[j]['label'] == 1:
                        complete_qids.add(qid)
                        # print >> fout, json.dumps(g, ensure_ascii=False).encode('utf8')

                # Write query pattern to file
                for j in xrange(len(query_patterns)):
                    print >> fout, json.dumps(query_patterns[j], ensure_ascii=False).encode('utf8')
    print "total valid question", len(qids)
    print "complete question", len(complete_qids)

def gen_svm_ranker_data(fn_query_pattern, fn_svm):
    pipeline = Pipeline()
    keys = ['mention_score', 'entity_score', 'relation_score',
            'constraint_entity_word',
            'constraint_entity_word']
    with open(fn_svm, 'w') as fout, open(fn_query_pattern) as fin:
        for line in fin:
            query_pattern = json.loads(line)
            print >> fout, pipeline.to_svm_ranker_input(query_pattern, keys)

def evaluate_svm_test(fn_query_pattern, fn_svm_res):

    with open(fn_query_pattern) as fin:
        query_patterns = [json.loads(line) for line in fin]
    with open(fn_svm_res) as fin:
        scores = [float(line) for line in fin]

    qid_hash_to_query_score = dict()
    for i in xrange(len(scores)):
        # query_patterns[i]['score'] = scores[i]
        qid_hash_to_query_score[(query_patterns[i]['qid'], query_patterns[i]['hash'])] = scores[i]

    # How to decide answers???


def debug(question):
    pipeline = Pipeline()
    question, graph = pipeline.gen_candidate_query_graph(question)
    print graph

if __name__ == '__main__':
    # basic_exp("../data/wq.test.complete.v2", "tmp")

    # Freebase is combination of fb.triple.mini and FB2M
    # gen_data_for_relation_matcher(
    #     ["../data/wq.train.complete.v2", "../data/wq.dev.complete.v2"],
    #     ["../data/simple.train.dev.el.v2"],
    #     "../data/merge_data/relation.train"
    # )

    # Freebase is combination of fb.triple.mini and FB2M
    # gen_data_for_relation_matcher(
    #     ["../data/wq.test.complete.v2"],
    #     [],
    #     "../data/merge_data/relation.wq.test"
    # )
    # gen_data_for_relation_matcher(
    #     [],
    #     ["../data/simple.test.el.v2"],
    #     "../data/merge_data/relation.simple.test"
    # )

    # Freebase is fb.triple.mini
    # gen_data_for_relation_matcher(
    #     ["../data/wq.train.complete.v2", "../data/wq.dev.complete.v2"],
    #     ["../data/simple.train.dev.el.v2"],
    #     "../data/my_fb/relation.train"
    # )
    # Freebase is fb.triple.mini
    # gen_data_for_relation_matcher(
    #     ["../data/wq.test.complete.v2"],
    #     [],
    #     "../data/my_fb/wq.relation.test"
    # )
    # gen_data_for_relation_matcher(
    #     [],
    #     ["../data/simple.test.el.v2"],
    #     "../data/my_fb/simple.relation.test"
    # )

    # Generate overall features for answer selection
    gen_query_graph(
        ['../data/wq.train.complete.v2', '../data/wq.dev.complete.v2'],
        [],
        '../data/wq.answer.selection.train.top3'
    )
    gen_query_graph(
        ['../data/wq.test.complete.v2'],
        [],
        '../data/wq.answer.selection.test.top3'
    )

    # gen_svm_ranker_data('../data/wq.answer.selection.train.top3', '../data/wq.train.top3.svm')
    # debug(sys.argv[1])

