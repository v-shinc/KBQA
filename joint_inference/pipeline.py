__author__ = 'chensn'
import sys
sys.path.insert(0, '..')
sys.path.append('../kb_manager/gen_py')
import heapq
import itertools
import copy
import globals
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

    def add_topic_feature(self, question):
        """
        :param question:
        :return:
            features: a list of feature
                      a feature have keys
                      ["topic", "mention", "mention score", "entity_score", "path",
                      "pattern", "answer", "relation_score", "pattern_repr", "relation_repr"]
        """
        # generate entity feature
        question, candidates = self.entity_linker.get_candidate_topic_entities(question)
        print '[Pipeline.add_topic_feature]', question
        features = []
        for mid, f in candidates.iteritems():
            f['topic'] = mid
            features.append(f)

        return question, features

    def add_path_feature(self, question, features, topk=-1):
        print '[Pipeline.add_relation_match_feature]', question
        new_features = []
        pattern_relation_set = set()

        for f in features:
            mid = f['topic']
            paths = self.db_manger.get_subgraph(mid)
            for path in paths:
                # if len(path) == 2:
                #     mediator = path[0][2]
                #     name = self.get_name(mediator)
                #     if name:
                #         relation = path[0][1]
                #         answer = path[0][2]
                #         print mediator, "mediator node has name", name
                #         print f['mention'], relation, mediator, path[1][1], self.get_name(path[1][2]), path[1][2]
                #
                #     else:
                #         relation = path[1][1]
                #         answer = path[1][2]
                # else:
                #     if not self.get_name(path[0][2]):
                #         print path[-1][2], "has no name!!!"
                #         continue
                #     else:
                #         relation = path[0][1]
                #         answer = path[0][2]

                relation = path[-1][1]
                answer = path[-1][2]
                if not self.get_name(answer):
                    continue
                feat = copy.deepcopy(f)
                feat['path'] = path
                feat['pattern'] = question.replace(f['mention'], '<$>')
                feat['relation'] = relation
                pattern_relation_set.add((feat['pattern'], feat['relation']))
                feat['answer'] = answer
                new_features.append(feat)

        if len(new_features) == 0:
            return []
        features = None
        # generate pattern-relation match score, distributed representations of pattern and relation

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
        for i in xrange(len(new_features)):
            if (new_features[i]['pattern'], new_features[i]['relation']) in relation_match_score:
                new_features[i]['relation_score'] = relation_match_score[(new_features[i]['pattern'], new_features[i]['relation'])]
                ret_features.append(new_features[i])
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
        candidates_topics = set()
        for i in xrange(len(features)):
            candidates_topics.add(features[i]['topic'])

        for i in xrange(len(features)):
            # Add constraint feature for CVT
            if len(features[i]['path']) == 2:
                cvt = features[i]['path'][0][-1]
                cons_paths = self.db_manger.get_one_hop_path(cvt)
                features[i]['constraint_entity_in_q'] = 0
                features[i]['constraint_entity_word'] = 0
                features[i]['constraint_entity_word_detail'] = ""
                num_name_cross = 0
                if len(cons_paths) > 10:
                    continue

                for _, _, obj in cons_paths:
                    if obj == features[i]['answer'] or obj == features[i]['topic']:
                        continue
                    # The constraint entity occurs in the question
                    if obj in candidates_topics:
                        features[i]['constraint_entity_in_q'] += 1
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
                            features[i]['constraint_entity_word_detail'] += '\t'+name

                if num_name_cross > 0:
                    features[i]['constraint_entity_word'] *= 1.0 / num_name_cross

            # Add constraint feature for answer
            name = self.get_name(features[i]['answer'])
            if not name:
                print features[i]['answer'], "has no name!!"
                features[i]['answer_word'] = 0
            else:
                answer_words = set(name.lower().split())
                features[i]['answer_word'] = len(answer_words.intersection(qwords)) * 1.0 / len(answer_words)

        return features

    def gen_candidate_relations(self, question):
        question, candidates = self.entity_linker.get_candidate_topic_entities(question)
        candidate_relations = set()

        for mid, f in candidates.iteritems():
            candidate_relations.update([r[-1] for r in self.db_manger.get_relations(mid)])
        return question, candidate_relations

    def gen_candidate_query_graph(self, question):

        question, features = self.add_topic_feature(question)
        features = self.add_path_feature(question, features, topk=-1)
        # features = self.add_constraints(question, features)
        return question, features

if __name__ == '__main__':
    pass














