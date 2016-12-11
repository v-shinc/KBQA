import numpy as np
import json
from utils.string_utils import naive_split, replace_mention_holder

# generate question group, label is relation
def gen_cluster(fn_wq_list, fn_simple_list, fn_out):
    clusters = dict()
    for fn in fn_wq_list:
        wq = json.load(open(fn))
        for data in wq:
            for path in data['paths']:
                if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                    rel = path[0].split()[-2]
                    if rel not in clusters:
                        clusters[rel] = set()
                    clusters[rel].add(' '.join(naive_split(replace_mention_holder(data['sentence']))))
    for fn in fn_simple_list:
        with open(fn) as fin:
            for line in fin:
                _, rel, _, pattern, _ = line.decode('utf8').strip().split('\t')
                if rel not in clusters:
                    clusters[rel] = set()
                clusters[rel].add(' '.join(naive_split(replace_mention_holder(pattern))))

    with open(fn_out, 'w') as fout:
        for rel, patterns in clusters.iteritems():
            print >> fout, '{}\t{}'.format(rel, ' '.join(patterns)).encode('utf8')

# Test code
# use word embedding from relation matcher model
class PatternCluster:
    def __init__(self, fn_cluster, fn_emb):
        self.clusters = dict()
        with open(fn_cluster) as fin:
            for line in fin:
                relation, line = line.strip().decode('utf8').split('\t')
                self.clusters[relation] = line.split()
        self.embeddings = dict()
        with open(fn_emb) as fin:
            for line in fin:
                line = line.strip().decode('utf8').split('\t')
                self.embeddings[line[0]] = np.array(map(float, line[1].split()))


    def semantic_score(self, q1, q2):
        repr1 = np.sum([self.embeddings[w] for w in q1.split() if w in self.embddings], 0)
        repr2 = np.sum([self.embeddings[w] for w in q2.split() if w in self.embeddings], 0)
        dot = np.dot(repr1, repr2)
        norm1 = np.sqrt(np.sum(repr1 ** 2))
        norm2 = np.sqrt(np.sum(rep2 ** 2))
        return dot / norm1 / norm2

    def get_match_score(self, pattern, relation):
        if relation not in self.clusters:
            return 0
        score = 0.
        for q in self.clusters[relation]:
            score += self.semantic_score(pattern, q)
        score /= len(self.clusters[relation])
        return score

    def get_topk_relation(self, pattern, topk):
        rank = []
        for r in self.clusters.keys():
            rank.append([r, self.get_match_score(pattern, r)])
        rank = sorted(rank, key=lambda x: x[1])
        if topk == -1:
            return rank
        else:
            return rank[:topk]


def evaluate(fn_cluster_train, fn_embedding, fn_wq):
    cluster = PatternCluster(fn_cluster_train, fn_embedding)
    num_p_at_1 = 0
    num_question = 0
    with open(fn_wq) as fin:
        wq = json.load(fin)
        for data in wq:
            pattern = ' '.join(naive_split(replace_mention_holder(data['sentence']))) # palceholder should be <$>
            best_relation = cluster.get_topk_relation(pattern, 1)[0]
            positive_relations = set()
            for path in data['paths']:
                if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                    positive_relations.add(path[0].split()[-2])
            if len(positive_relations) == 0:
                continue

            if best_relation in positive_relations:
                num_p_at_1 += 1
            num_question += 1
            print "p@1: {}".format(num_p_at_1 * 1.0 / num_question)


if __name__ == '__main__':
    fn_cluster = '../pattern_cluster/pattern.cluster.train'
    fn_wq_test = '../data/wq.test.complete.v2'
    fn_embedding = 'xxx'
    gen_cluster(['../data/wq.train.complete.v2', '../data/wq.dev.complete.v2'], ['../data/simple.train.dev.el.v2'], fn_cluster)
    evaluate(fn_cluster, fn_embedding, fn_wq_test)


