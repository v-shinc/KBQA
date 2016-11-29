__author__ = 'chensn'
import sys
sys.path.insert(0, '..')
sys.path.append('../kb_manager/gen_py')
import itertools
import copy
import globals
from tagger.predict import EntityLinker, EntityMentionTagger
from relation_matcher.predict import RelationMatcher
# from kb_manager.mm_freebase import MemoryFreebaseHelper
from kb_manager.freebase_client import FreebaseClient
globals.read_configuration('../config.cfg')


class BetaAnswer(object):

    def __init__(self):
        self.relation_matcher = RelationMatcher(globals.config.get('BetaAnswer', 'relation-matcher'))
        mention_tagger = EntityMentionTagger(globals.config.get('BetaAnswer', 'entity-mention-tagger'))
        self.entity_linker = EntityLinker(mention_tagger)

        self.freebase = FreebaseClient()

    def parse(self, question):
        # generate entity feature
        question, candidates = self.entity_linker.get_candidate_topic_entities(question)
        print '[BetaAnswer.parse]', question
        pattern_relation_set = set()
        features = []
        for mid, f in candidates.iteritems():
            f['topic'] = mid
            paths = self.freebase.get_subgraph(mid)
            for path in paths:
                feat = copy.deepcopy(f)
                feat['path'] = path
                feat['pattern'] = question.replace(f['mention'], '<$>')
                feat['relation'] = path[-1][1]
                pattern_relation_set.add((feat['pattern'], feat['relation']))
                feat['answer'] = path[-1][2]
                features.append(feat)
        if len(features) == 0:
            return question, []

        # generate pattern-relation match score, distributed representations of pattern and relation
        patterns = []
        relations = []
        for p, r in pattern_relation_set:
            patterns.append(p)
            relations.append(r)

        scores, pattern_reprs, relation_reprs = self.relation_matcher.get_batch_match_score(patterns, relations)
        relation_match_score = dict()
        # pattern_reprs = dict(zip(patterns, pattern_reprs))
        # relation_reprs = dict(zip(relations, relation_reprs))
        for p, r, s in itertools.izip(patterns, relations, scores):
            relation_match_score[(p, r)] = s

        for i in xrange(len(features)):
            features[i]['relation_score'] = relation_match_score[(features[i]['pattern'], features[i]['relation'])]
            # features[i]['pattern_repr'] = pattern_reprs[features[i]['pattern']]
            # features[i]['relation_repr'] = relation_reprs[features[i]['relation']]

        return question, features

    def answer(self, question):
        question, candidates = self.entity_linker.get_candidate_topic_entities(question)


if __name__ == '__main__':
    pass














