__author__ = 'chensn'
import sys
sys.path.insert(0, '..')
sys.path.append('../kb_manager/gen_py')
import heapq
import itertools
import copy
import globals
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tagger.predict import EntityLinker, EntityMentionTagger
from relation_matcher.predict import RelationMatcher
from kb_manager.db_manager import DBManager
# from kb_manager.mm_freebase import MemoryFreebaseHelper
# from kb_manager.freebase_client import FreebaseClient
globals.read_configuration('../config.cfg')


class Pipeline(object):

    def __init__(self):
        self.relation_matcher = RelationMatcher(globals.config.get('BetaAnswer', 'relation-matcher'))
        mention_tagger = EntityMentionTagger(globals.config.get('BetaAnswer', 'entity-mention-tagger'))
        self.entity_linker = EntityLinker(mention_tagger)

        # self.freebase = FreebaseClient()
        self.db_manger = DBManager()
        self.lemma = dict()
        self.lemmatiser = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def _get_lemma(self, word):
        if word not in self.lemma:
            self.lemma[word] = self.lemmatiser.lemmatize(word)
        return self.lemma[word]


    def add_topic_feature(self, question):
        """
        :param question:
        :return:
            queries: a list of query
                      a feature have keys
                      ["topic", "topic_name", "mention", "mention score", "entity_score"]
        """
        # generate entity feature
        question, queries = self.entity_linker.get_candidate_topic_entities(question)
        ret_queries = []
        for i in xrange(len(queries)):
            name = self.get_name(queries[i]['topic'])
            if name == None:
                continue
                # raise ValueError("Topic name is None")
            queries[i]['topic_name'] = name
            ret_queries.append(queries[i])
        print '[Pipeline.add_topic_feature]', question
        return question, ret_queries

    def add_path_feature(self, question, queries, topk=-1, include_relation_score=True):
        """
        :param question:
        :return:
            queries: a list of query
                      the increased feature keys are
                      ["path", "relation", "pattern", "relation_score", "answer"]
        """
        # print '[Pipeline.add_relation_match_feature]', question
        new_queries = []
        pattern_relation_set = set()

        for f in queries:
            mid = f['topic']
            paths = self.db_manger.get_subgraph(mid)
            for path in paths:
                relation = path[-1][1]
                answer = path[-1][2]
                if not self.get_name(answer):
                    continue
                feat = copy.deepcopy(f)
                feat['path'] = path
                feat['pattern'] = question.replace(f['mention'], '<$>')
                feat['relation'] = relation
                relation_lemmas = [self.lemmatiser.lemmatize(w) for w in relation.split('.')[-1].split('_')]
                pattern_lemmas = [self.lemmatiser.lemmatize(w) for w in feat['pattern'].split()]
                feat['pattern_lemma'] = ' '.join(pattern_lemmas)
                feat['rel_pat_overlap'] = 1. if (set(relation_lemmas) - self.stopwords).intersection(set(pattern_lemmas)) else 0.
                pattern_relation_set.add((feat['pattern'], feat['relation']))
                feat['answer'] = answer
                new_queries.append(feat)

        if len(new_queries) == 0:
            return []
        queries = None
        # generate pattern-relation match score, distributed representations of pattern and relation
        if include_relation_score:
            patterns = []
            relations = []
            for p, r in pattern_relation_set:
                patterns.append(p)
                relations.append(r)

            scores, pattern_reprs, relation_reprs = self.relation_matcher.get_batch_match_score(patterns, relations)
            # pattern_reprs = dict(zip(patterns, pattern_reprs))
            # relation_reprs = dict(zip(relations, relation_reprs))
            relation_match_score = dict()
            if topk > 0:
                # use pattern-relation score to filter
                pq = []
                for i, s in enumerate(scores):
                    s = float(s)

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
            ret_queries = []
            for i in xrange(len(new_queries)):
                if (new_queries[i]['pattern'], new_queries[i]['relation']) in relation_match_score:
                    new_queries[i]['relation_score'] = relation_match_score[(new_queries[i]['pattern'], new_queries[i]['relation'])]
                    ret_queries.append(new_queries[i])
                # features[i]['pattern_repr'] = pattern_reprs[features[i]['pattern']]
                # features[i]['relation_repr'] = relation_reprs[features[i]['relation']]
            return ret_queries
        else:
            return new_queries

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

    def hash_query(self, path, constraints):
        """path contains mediator, constraints is list of triple"""
        sequence = []

        if len(path) == 2:
            # ignore mediator and answer
            sequence.append(path[0][0])  # subject
            sequence.append(path[0][1])  # first relation
            sequence.append(path[1][1])  # second relation
            consts = set()
            # ignore mediator
            for c in constraints:
                consts.add((c[1], c[2]))
            for c in consts:
                sequence.extend(c)
        else:
            # ignore answer
            sequence.append(path[0][0])  # subject
            sequence.append(path[0][1])  # first relation

        return hash(" ".join(sequence))


    def add_constraints(self, question, queries):
        qwords = set(question.split())
        candidates_topics = set()
        for i in xrange(len(queries)):
            candidates_topics.add(queries[i]['topic'])

        for i in xrange(len(queries)):
            # Add constraint feature for CVT
            if len(queries[i]['path']) == 2:
                cvt = queries[i]['path'][0][-1]
                cons_paths = self.db_manger.get_one_hop_path(cvt)
                queries[i]['constraint_entity_in_q'] = 0
                queries[i]['constraint_entity_word'] = 0
                queries[i]['constraints'] = []
                # queries[i]['constraint_entity_word_detail'] = ""
                num_name_cross = 0
                # if len(cons_paths) > 10:
                #     continue

                for _, rel, obj in cons_paths:
                    if obj == queries[i]['answer'] or obj == queries[i]['topic']:
                        continue
                    # The constraint entity occurs in the question
                    if obj in candidates_topics:
                        queries[i]['constraint_entity_in_q'] += 1
                        queries[i]['constraints'].append([cvt, rel, obj])
                    # Some words of the constraint entity's name appear in the question
                    # percentage of the words in the name of the constraint entity appear in the question
                    name = self.get_name(obj)
                    if not name:
                        print "constraint", obj, "has no name!"
                    else:
                        cons_words = set(name.lower().split())
                        intersect_per = len(cons_words.intersection(qwords)) * 1.0 / len(cons_words)
                        queries[i]['constraint_entity_word'] += intersect_per

                        if intersect_per > 0:
                            num_name_cross += 1
                            # queries[i]['constraint_entity_word_detail'] += '\t'+name

                if num_name_cross > 0:
                    queries[i]['constraint_entity_word'] *= 1.0 / num_name_cross

            # Add constraint feature for answer
            # name = self.get_name(queries[i]['answer'])
            # if not name:
            #     print queries[i]['answer'], "has no name!!"
            #     queries[i]['answer_word'] = 0
            # else:
            #     answer_words = set(name.lower().split())
            #     queries[i]['answer_word'] = len(answer_words.intersection(qwords)) * 1.0 / len(answer_words)

        return queries

    def create_query_graph_given_topic(self, question, mention):
        question, queries = self.entity_linker.get_candidate_topic_entities_given_mention(question, mention)
        queries = self.add_path_feature(question, queries)
        return question, queries

    def gen_candidate_relations(self, question):
        question, candidates = self.entity_linker.get_candidate_topic_entities(question)
        candidate_relations = set()

        for f in candidates:
            mid = f['topic']
            candidate_relations.update([r[-1] for r in self.db_manger.get_multiple_hop_relations(mid)])
        return question, candidate_relations

    def gen_candidate_query_graph(self, question):
        question, queries = self.add_topic_feature(question)
        queries = self.add_path_feature(question, queries, topk=-1)
        queries = self.add_constraints(question, queries)

        for i in xrange(len(queries)):
            queries[i]['hash'] = self.hash_query(queries[i]['path'], queries[i].get('constraints', []))

        return question, queries

    def extract_query_pattern_and_f1(self, queries, gold_answers):
        if not isinstance(gold_answers, set):
            gold_answers = set(gold_answers)
        query_hash_to_answers = dict()
        query_hash_to_pattern = dict()
        for i in xrange(len(queries)):
            code = queries[i]['hash']

            if code not in query_hash_to_pattern:
                query_hash_to_answers[code] = set()
                query_hash_to_pattern[code] = queries[i]  # select one representative
            query_hash_to_answers[code].add(queries[i]['answer'])

        for hash_code in query_hash_to_answers.keys():
            _, _, f1 = compute_f1(gold_answers, query_hash_to_answers[hash_code])
            query_hash_to_pattern[hash_code]['f1'] = f1
            query_hash_to_pattern[hash_code]['pattern_answer'] = list(query_hash_to_answers[hash_code])
            query_hash_to_pattern[hash_code]['num_answer'] = len(query_hash_to_answers[hash_code])
        return query_hash_to_pattern.values()

    @staticmethod
    def to_svm_ranker_input(query_pattern, keys):
        """query_pattern must contain 'qid' """
        f = ' '.join(["{}:{}".format(j + 1, query_pattern.get(k, 0)) for j, k in enumerate(keys)])
        return "{} qid:{} {}".format(query_pattern['f1'], query_pattern['qid'], f)


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

if __name__ == '__main__':
    pass














