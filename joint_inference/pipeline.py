__author__ = 'chensn'
import sys
sys.path.insert(0, '..')
# sys.path.append('../kb_manager/gen_py')
import heapq
import itertools
import copy
import globals
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tagger.predict import EntityLinker, EntityMentionTagger
from entity_linker.aqqu_entity_linker import AqquEntityLinker
from relation_matcher.predict import RelationMatcher
from kb_manager.db_manager import DBManager
from nltk.stem.porter import PorterStemmer
# from kb_manager.mm_freebase import MemoryFreebaseHelper
# from kb_manager.freebase_client import FreebaseClient



class Pipeline(object):

    def __init__(self, use_aqqu=False, use_relation_matcher=False):
        self.use_relation_matcher = use_relation_matcher
        if use_relation_matcher:
            self.relation_matcher = RelationMatcher(globals.config.get('BetaAnswer', 'relation-matcher'))
        if use_aqqu:
            print "Start initializing Aqqu entity linker..."
            self.entity_linker = AqquEntityLinker()
            print "Finish"
        else:
            print "Start initializing deep CRF entity linker..."
            mention_tagger = EntityMentionTagger(globals.config.get('BetaAnswer', 'entity-mention-tagger'))
            self.entity_linker = EntityLinker(mention_tagger)

        # self.freebase = FreebaseClient()
        self.db_manger = DBManager()
        self.lemma = dict()
        self.lemmatiser = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.qwords = {'what', "where", "when", "who", "which", "how"}
        self.qwords_rel_cooccur = dict()
        with open(globals.config.get("FREEBASE", "question-word-relation-cooccur")) as fin:
            for line in fin:
                qw_rel, cooccur = line.strip().split('\t')
                self.qwords_rel_cooccur[qw_rel] = float(cooccur)

    def _get_lemma(self, word):
        if word not in self.lemma:
            self.lemma[word] = self.lemmatiser.lemmatize(word)
        return self.lemma[word]

    def get_qword(self, question):
        question = question.split(' ')
        length = len(question)
        for i in xrange(length):
            w = question[i]
            if w in self.qwords:
                # return w
                if w == 'which':
                    return question[i+1] if question[i+1] != '<$>' else question[i+2]
                else:
                    return w
        return "none"

    def add_topic_feature(self, question, queries):
        """
        :param question:
        :return:
            queries: a list of query
                      a feature have keys
                      ["topic", "topic_name", "mention", "mention score", "entity_score"]
        """

        ret_queries = []
        for i in xrange(len(queries)):
            name = self.get_name(queries[i]['topic'])
            if name == None:
                continue
                # raise ValueError("Topic name is None")
            queries[i]['topic_name'] = name
            queries[i]['topic_notable_type'] = self.db_manger.get_notable_type(queries[i]['topic'])
            # queries[i]['topic_type'] = self.db_manger.get_type(queries[i]['topic'])
            ret_queries.append(queries[i])

        # print '[Pipeline.add_topic_feature]', question
        return ret_queries

    def _get_qword_relation_cooccur(self, question, relation):
        rel = ".".join(relation.split('.')[-3:-1])
        for w in question.split():
            if w in self.qwords:
                return self.qwords_rel_cooccur.get(w+" "+rel, 0.)
        return 0.

    def add_path_feature(self, question, queries, topk=-1, debug=False):
        """
        :param question:
        :return:
            queries: a list of query
                      the increased feature keys are
                      ["path", "relation", "pattern", "relation_score", "answer"]
        """
        print '[Pipeline.add_relation_match_feature]', question
        new_queries = []
        pattern_relation_set = set()

        for f in queries:
            mid = f['topic']
            paths = self.db_manger.get_subgraph(mid)
            # if debug:
            #     "DEBUG"
            #     print "topic", mid
            for path in paths:
                relation = path[-1][1]
                answer = path[-1][2]
                # if debug:
                #     print relation, answer
                if not self.get_name(answer):
                    # print "{} has no name, {}!!!".format(answer, relation)
                    continue
                feat = copy.deepcopy(f)
                feat['path'] = path
                feat['pattern'] = question.replace(f['mention'], '<$>')
                feat['relation'] = relation
                # relation_lemmas = [self.lemmatiser.lemmatize(w) for w in relation.split('.')[-1].split('_')]
                # pattern_lemmas = [self.lemmatiser.lemmatize(w) for w in feat['pattern'].split()]
                # feat['pattern_lemma'] = ' '.join(pattern_lemmas)
                # feat['relation_lemma'] = ' '.join(relation_lemmas)
                # feat['rel_pat_overlap'] = 1. if (set(relation_lemmas) - self.stopwords).intersection(
                #     set(pattern_lemmas)) else 0.
                relation_stem = [self.stemmer.stem(w) for w in relation.split('.')[-1].split('_')]
                pattern_stem = [self.stemmer.stem(w) for w in feat['pattern'].split()]
                feat['relation_stem'] = ' '.join(relation_stem)
                feat['pattern_stem'] = ' '.join(pattern_stem)
                feat['rel_pat_overlap'] = 1. if (set(relation_stem) - self.stopwords).intersection(
                    set(pattern_stem)) else 0.
                pattern_relation_set.add((feat['pattern'], feat['relation']))
                feat['answer'] = answer
                feat["qw_rel_occur"] = self._get_qword_relation_cooccur(question, relation)
                new_queries.append(feat)

        if len(new_queries) == 0:
            return []
        queries = None
        # generate pattern-relation match score, distributed representations of pattern and relation
        if self.use_relation_matcher:
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
                # print '[Pipeline.add_relation_match_feature] generate pattern-relation score'
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
            # print 'return new queries'
            for i in xrange(len(new_queries)):
                if 'pattern' not in new_queries[i]:
                    pass
                    # print 'xxx', new_queries[i]
            return new_queries

    def get_name(self, entry):
        if entry.startswith('m.'):
            return self.db_manger.get_name(entry)[0]

        elif entry.endswith('^^gYear'):
            return entry[1:-8]
        elif entry.endswith('^^date'):
            return entry[1:5]
        elif entry.isdigit() and len(entry) == 4:  # isYear
            return entry
        else:
            # print entry, "has no name"
            return None

    def hash_query(self, path, constraints, mention, answer_constraints):
        """path contains mediator, constraints is list of triple"""
        sequence = []

        if len(path) == 2:
            # ignore mediator and answer
            sequence.append(path[0][0])  # subject
            sequence.append(path[0][1])  # first relation
            sequence.append(path[1][1])  # second relation
            sequence.append(mention)

            consts = set()
            # ignore mediator
            for c in constraints:  # order makes no sense
                consts.add((c[1], c[2]))
            for c in consts:
                sequence.extend(c)
        else:
            # ignore answer
            sequence.append(path[0][0])  # subject
            sequence.append(path[0][1])  # first relation
            sequence.append(mention)

        # add answer constraints
        consts = set()  # order makes no sense
        for c in answer_constraints:
            consts.add((c[1], c[2]))
        for c in consts:
            sequence.extend(c)
        return hash(" ".join(sequence))

    def add_answer_constraints(self, question, queries):
        question = question.split()
        male = {'dad', 'father', 'brother', "grandfather", "grandson", "son", "husband"}
        female = {'mom', 'mother', 'sister', 'grandmother', 'granddaughter', 'daughter', 'wife'}
        # initialize answer constraints
        for i in xrange(len(queries)):
            queries[i]['answer_constraints'] = []
            queries[i]['gender_consistency'] = 0.

        for w in question:
            # gender consistency
            if w in male or w in female:
                for i in xrange(len(queries)):
                    gender_constraints = DBManager.get_property(queries[i]['answer'], "gender")
                    if len(gender_constraints) > 0:
                        queries[i]['answer_constraints'].extend(gender_constraints)
                        queries[i]['gender_consistency'] = float(
                            ("m.05zppz" == gender_constraints[0][-1] and w in male) or (
                                'm.02zsn' == gender_constraints[0][-1] and w in female))

        question_lemma = set([self._get_lemma(w) for w in question])

        for i in xrange(len(queries)):
            queries[i]['type_consistency'] = 0.
            answer = queries[i]['answer']
            types = self.get_type(answer)
            # The percentage of the words in the constraint entity that appear in the question
            best_type_index = -1
            best_type_overlap = 0.
            for j, t in enumerate(types):
                type_name = DBManager.get_name(t)[0].lower() if t.startswith('m.') else t
                num_overlap = 0
                for w_ in type_name.split():
                    w_ = self._get_lemma(w_)
                    if w_ in question_lemma:
                        num_overlap += 1
                if best_type_overlap < (num_overlap * 1. / len(type_name.split())):
                    best_type_overlap = (num_overlap * 1. / len(type_name.split()))
                    best_type_index = j

            if best_type_overlap > 0.:
                queries[i]['type_consistency'] = best_type_overlap
                queries[i]['answer_constraints'].append([answer, 'common.topic.notable_types', types[best_type_index]])
                # print "$" * 40, question, self.get_name(types[best_type_index])

        return queries

    def add_cvt_constraints(self, question, queries):
        pass

    def add_constraints(self, question, queries, debug=False):
        word_in_question = set(question.split())
        candidates_topics = set()
        topic_to_mention = dict()
        for i in xrange(len(queries)):
            candidates_topics.add(queries[i]['topic'])
            topic_to_mention[queries[i]['topic']] = queries[i]['mention']

        for i in xrange(len(queries)):
            queries[i]['constraint_entity_in_q'] = 0
            queries[i]['constraint_entity_word'] = 0
            queries[i]['constraints'] = []
            # Add constraint feature for CVT
            if len(queries[i]['path']) == 2:
                cvt = queries[i]['path'][0][-1]
                cons_paths = self.db_manger.get_one_hop_path(cvt)

                num_name_cross = 0

                for _, rel, obj in cons_paths:
                    if obj == queries[i]['answer'] or obj == queries[i]['topic']:
                        continue
                    # The constraint entity occurs in the question
                    # TODO: constraint entity can't overlap with topic entity
                    if obj in candidates_topics and topic_to_mention[obj] != topic_to_mention[queries[i]['topic']]:
                        queries[i]['constraint_entity_in_q'] += 1
                        queries[i]['constraints'].append([cvt, rel, obj])
                    # Some words of the constraint entity's name appear in the question
                    # percentage of the words in the name of the constraint entity appear in the question

                    # TODO:
                    name = self.get_name(obj)

                    if not name:
                        pass
                        # print "constraint", obj, "has no name!"
                    else:

                        cons_words = set(name.lower().split())
                        intersect_per = len(cons_words.intersection(word_in_question)) * 1.0 / len(cons_words)
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

        queries = self.add_answer_constraints(question, queries)
        return queries

    def get_type(self, entry):
        if entry.startswith('m.'):
            types = self.db_manger.get_notable_type(entry)
            return types if types else ['none']
        if entry.endswith('^^gYear'):
            return ['year']
        if entry.endswith('^^date'):
            return ['date']
        else:
            return ['illegal']

    def add_answer_feature(self, question, queries):
        """queries: this is extracted query pattern """
        qword = self.get_qword(question)
        for i in xrange(len(queries)):
            type_dist = dict()  # type distribution
            for answer in queries[i]['pattern_answer']:
                for t in self.get_type(answer):
                    type_dist[t] = type_dist.get(t, 0) + 1
            sum_ = sum(type_dist.itervalues())
            for t in type_dist.iterkeys():
                type_dist[t] *= 1. / sum_
            queries[i]['answer_types'] = type_dist
            queries[i]['qword'] = qword
        return queries

    def create_query_graph_given_topic(self, question, mention):
        question, queries = self.entity_linker.get_candidate_topic_entities_given_mention(question, mention)
        queries = self.add_path_feature(question, queries)
        return question, queries

    def gen_candidate_relations(self, question, debug=False):
        question, candidates = self.entity_linker.get_candidate_topic_entities(question)
        candidate_relations = set()
        # if debug:
        #     print question
        #     for c in candidates:
        #         print c
        for f in candidates:
            mid = f['topic']
            rels = [r[-1] for r in self.db_manger.get_multiple_hop_relations(mid)]
            # if debug:
            #     print mid
            #     for r in rels:
            #         print r
            candidate_relations.update(rels)
        return question, candidate_relations

    def gen_candidate_query_graph(self, question, debug=False):
        # generate entity feature
        question, queries = self.entity_linker.get_candidate_topic_entities(question)
        queries = self.add_topic_feature(question, queries)
        # if debug:
        #     print queries
        #     for q in queries:
        #         print q
        queries = self.add_path_feature(question, queries, topk=-1, debug=debug)
        queries = self.add_constraints(question, queries, debug)

        for i in xrange(len(queries)):
            queries[i]['hash'] = self.hash_query(queries[i]['path'], queries[i].get('constraints', []), queries[i]['mention'], queries[i]['answer_constraints'])

        # if debug:
        #     for q in queries:
        #         print q
        return question, queries

    def gen_candidate_query_graph_for_prediction(self, question):
        # print '[Pipeline.gen_candidate_query_graph_for_prediction]'
        # print ' entity_linker.get_candidate_topic_entities({})'.format(question)
        question, entity_link_res = self.entity_linker.get_candidate_topic_entities(question)
        # print ' add_topic_feature'
        queries = self.add_topic_feature(question, entity_link_res)
        # print ' add_path_feature'
        queries = self.add_path_feature(question, queries, topk=-1)
        # print ' add_constraints'
        queries = self.add_constraints(question, queries)

        for i in xrange(len(queries)):
            queries[i]['hash'] = self.hash_query(queries[i]['path'], queries[i].get('constraints', []), queries[i]['mention'], queries[i]['answer_constraints'])

        # print ' extract query pattern'
        query_hash_to_pattern = dict()
        query_hash_to_answers = dict()
        for i in xrange(len(queries)):
            code = queries[i]['hash']

            if code not in query_hash_to_pattern:
                query_hash_to_answers[code] = set()
                query_hash_to_pattern[code] = queries[i]  # select one representative
            query_hash_to_answers[code].add(queries[i]['answer'])

        for hash_code in query_hash_to_answers.keys():
            query_hash_to_pattern[hash_code]['pattern_answer'] = list(query_hash_to_answers[hash_code])
            query_hash_to_pattern[hash_code]['num_answer'] = len(query_hash_to_answers[hash_code])
        return question, entity_link_res, query_hash_to_pattern.values()

    def extract_query_pattern_and_f1(self, queries, gold_answers, debug=False):
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
    globals.read_configuration('../config.cfg')
    import sys
    pipeline = Pipeline(False)
    # question, queries = pipeline.add_topic_feature(sys.argv[1])
    # print queries
    # queries = pipeline.add_path_feature(question, queries, debug=False)
    # pipeline.gen_candidate_relations(sys.argv[1], True)
    # queries = pipeline.add_constraints(question, queries)
    # for q in queries:
    #     print q
    question, queries = pipeline.gen_candidate_query_graph(sys.argv[1], debug=True)
    query_pattern = pipeline.extract_query_pattern_and_f1(queries, {'m.0f2zfl'})

    # for q in query_pattern:
    #     print q














