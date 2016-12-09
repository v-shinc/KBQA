__author__ = 'chensn'
import sys
sys.path.insert(0, '..')
# sys.path.append('../kb_manager/gen_py')
import json
import heapq
import itertools
import copy
import globals
from tagger.predict import EntityLinker, EntityMentionTagger
from relation_matcher.predict import RelationMatcher
from kb_manager.db_manager import DBManager
globals.read_configuration('../config.cfg')


class FeatureExtractor(object):

    def __init__(self):
        self.relation_matcher = RelationMatcher(globals.config.get('BetaAnswer', 'relation-matcher'))
        mention_tagger = EntityMentionTagger(globals.config.get('BetaAnswer', 'entity-mention-tagger'))
        self.entity_linker = EntityLinker(mention_tagger)
        self.db_manger = DBManager()

    def add_topic_feature(self, qid, question):

        # generate entity feature
        question, features = self.entity_linker.get_candidate_topic_entities(question)
        print '[FeatureExtractor.add_topic_feature]', question
        for i in xrange(len(features)):
            features[i]['qid'] = qid
        return question, features

    def add_relation_feature(self, question, features, topk):
        extended_features = []
        pat_rel_set = set()
        relations = []
        patterns = []
        for i in xrange(len(features)):
            subject = features[i]['topic']
            core_paths = self.db_manger.get_core_paths_without_object(subject)
            for path in core_paths:
                feat = copy.deepcopy(features[i])

                feat['path'] = path
                feat['relation'] = path[-1]
                feat['pattern'] = question.replace(feat['mention'], '<$>')
                if (feat['pattern'], feat['relation']) not in pat_rel_set:
                    pat_rel_set.add((feat['pattern'], feat['relation']))
                    patterns.append(feat['pattern'])
                    relations.append(feat['relation'])
                extended_features.append(feat)
        if len(extended_features) == 0:
            return []
        features = None
        # generate pattern-relation match score, distributed representations of pattern and relation
        scores, pattern_reprs, relation_reprs = self.relation_matcher.get_batch_match_score(patterns, relations)
        # pattern_reprs = dict(zip(patterns, pattern_reprs))
        # relation_reprs = dict(zip(relations, relation_reprs))
        relation_match_score = dict()
        if topk > 0:
            # use pattern-relation score to filter
            pq = []
            for i, s in enumerate(scores):
                s = float(s)
                # if s < 0:
                #     continue
                if len(pq) < topk:
                    pq.append([s, i])
                elif pq[0][0] < s:
                    heapq.heapreplace(pq, [s, i])
            for s, i in pq:
                relation_match_score[(patterns[i], relations[i])] = s
        else:
            print '[Pipeline.add_relation_match_feature] generate pattern-relation score'
            for p, r, s in itertools.izip(patterns, relations, scores):
                relation_match_score[(p, r)] = float(s)
        ret_features = []
        for i in xrange(len(extended_features)):
            if (extended_features[i]['pattern'], extended_features[i]['relation']) in relation_match_score:
                extended_features[i]['relation_score'] = relation_match_score[(extended_features[i]['pattern'], extended_features[i]['relation'])]
                ret_features.append(extended_features[i])
            # features[i]['pattern_repr'] = pattern_reprs[features[i]['pattern']]
            # features[i]['relation_repr'] = relation_reprs[features[i]['relation']]

        return ret_features

    def get_name(self, entry):
        if entry.startswith('m.'):
            return self.db_manger.get_name(entry)[0]

        elif entry.endswith('^^gYear'):
            return entry[1:-8]
        elif entry.endswith('^^date'):
            return entry[1:5]
        else:
            print entry, "has no name"
            return None

    def add_constraints(self, question, features):
        qwords = set(question.split())
        # TODO: use aqqu to find constraint entities
        candidates_topics = set()
        for i in xrange(len(features)):
            candidates_topics.add(features[i]['topic'])

        for i in xrange(len(features)):
            # Add constraint feature for CVT
            if len(features[i]['path']) == 5:
                cvt = features[i]['path'][2]
                cons_paths = self.db_manger.get_one_hop_path(cvt)
                features[i]['constraint_entity_in_q'] = 0
                features[i]['constraint_entity_word'] = 0
                features[i]['constraints'] = []
                # features[i]['constraint_entity_word_detail'] = ""
                num_name_cross = 0
                # if len(cons_paths) > 10:
                #     continue

                for _, rel, obj in cons_paths:
                    if rel == features[i]['relation'] or obj == features[i]['topic']:  # avoid constraint node being answer
                        continue
                    # The constraint entity occurs in the question
                    if obj in candidates_topics:
                        features[i]['constraint_entity_in_q'] += 1
                        features[i]['constraints'].append({
                            "source_node_index": 0,
                            "node_predicate": rel,
                            "argument": obj
                        })
                    # Some words of the constraint entity's name appear in the question
                    # percentage of the words in the name of the constraint entity appear in the question
                    name = self.get_name(obj)
                    if not name:
                        print obj, "has no name!"
                    else:
                        cons_words = set(name.lower().split())
                        intersect_per = len(cons_words.intersection(qwords)) * 1.0 / len(cons_words)
                        features[i]['constraint_entity_word'] += intersect_per

                        if intersect_per > 0:
                            num_name_cross += 1
                            # features[i]['constraint_entity_word_detail'] += '\t'+name

                if num_name_cross > 0:
                    features[i]['constraint_entity_word'] *= 1.0 / num_name_cross

        return features

    @staticmethod
    def add_rank(features, gold_topic, gold_relations):
        # if e_pred == e_gold and r_pred == r_gold => rank = 3
        # else if e_pred == e_gold or r_pred == r_gold => rank = 2
        # else rank = 1
        if not isinstance(gold_relations, set):
            gold_relations = set(gold_relations)

        for i in xrange(len(features)):
            if features[i]['relation'] in gold_relations and features[i]['topic'] == gold_topic:
                features[i]['rank'] = 3
            elif features[i]['relation'] in gold_relations or features[i]['topic'] == gold_topic:
                features[i]['rank'] = 2
            else:
                features[i]['rank'] = 1
        return features

    @staticmethod
    def to_svm_ranker_input(features, keys):
        ranker_input = []
        for i in xrange(len(features)):
            f = ' '.join(["{}:{}".format(j + 1, features[i].get(k, 0)) for j, k in enumerate(keys)])
            ranker_input.append("{} qid:{} {}".format(features[i]['rank'], features[i]['qid'], f))
        return ranker_input

    def extract_query_feature(self, question, qid, gold_topic, gold_relations):
        question, features = self.add_topic_feature(qid, question)
        features = self.add_relation_feature(question, features, topk=10)
        features = self.add_constraints(question, features)
        features = self.add_rank(features, gold_topic, gold_relations)
        return question, features

def gen_query_graph(fn_wq_list, fn_simple_list, fn_out):
    extractor = FeatureExtractor()
    complete_qids = set()
    qids = set()
    with open(fn_out, 'w') as fout:
        # process webquestion
        for fn in fn_wq_list:
            webq = json.load(open(fn), encoding="utf8")
            for data in webq:

                positive_relations = set()
                for path in data['paths']:
                    if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                        positive_relations.add(path[0].split()[-2])

                if len(positive_relations) == 0:
                    continue
                qids.add(data['id'])
                question, features = \
                    extractor.extract_query_feature(data['utterance'], data['id'], data['mid1'], positive_relations)

                for g in features:
                    g['question'] = question
                    if g['rank'] == 3:
                        complete_qids.add(data['id'])

                svm_inputs = extractor.to_svm_ranker_input(features, ['mention_score', 'entity_score', 'relation_score', 'constraint_entity_in_q', 'constraint_entity_word'])
                for line in svm_inputs:
                    print >> fout, line

    print "total valid question", len(qids)
    print "complete question", len(complete_qids)

if __name__ == '__main__':
    globals.read_configuration('../config.cfg')
    gen_query_graph(
        ['../data/wq.train.complete.v2', '../data/wq.dev.complete.v2'],
        [],
        '../data/wq.train.top3.svm'
    )