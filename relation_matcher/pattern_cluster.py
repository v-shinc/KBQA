import numpy as np
import json
import sys
import heapq
sys.path.insert(0, '..')
from utils.string_utils import naive_split, replace_mention_holder


# Generate question group, label is relation
def gen_pattern_cluster(fn_wq_list, fn_simple_list, fn_out):
    clusters = dict()
    for fn in fn_wq_list:
        wq = json.load(open(fn))
        for data in wq:
            for path in data['paths']:
                if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                    core_path = path[0].split()
                    if len(core_path) == 3:
                        rel = core_path[1]
                    else:
                        rel = core_path[1] + ' ' + core_path[3]

                    if rel not in clusters:
                        clusters[rel] = set()
                    # clusters[rel].add(' '.join(naive_split(replace_mention_holder(data['sentence']))))
                    clusters[rel].add(replace_mention_holder(data['sentence']))
    for fn in fn_simple_list:
        with open(fn) as fin:
            for line in fin:
                _, rel, _, pattern, _ = line.decode('utf8').strip().split('\t')
                if rel not in clusters:
                    clusters[rel] = set()
                # clusters[rel].add(' '.join(naive_split(replace_mention_holder(pattern))))
                clusters[rel].add(replace_mention_holder(pattern))

    # with open(fn_out, 'w') as fout:
    #     for rel, patterns in clusters.iteritems():
    #         print >> fout, ('%s\t%s' % (rel, '\t'.join(patterns))).encode('utf8')
    output = []
    for rel, pattern in clusters.items():
        output.append({'relation': rel, 'pattern': list(pattern)})
    with open(fn_out, 'w') as fout:
        json.dump(output, fout, indent=4)

def gen_question_cluster(fn_wq_list, fn_simple_list, fn_out):
    clusters = dict()
    for fn in fn_wq_list:
        wq = json.load(open(fn))
        for data in wq:
            for path in data['paths']:
                if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                    core_path = path[0].split()
                    if len(core_path) == 3:
                        rel = core_path[1]
                    else:
                        rel = core_path[1] + ' ' + core_path[3]

                    if rel not in clusters:
                        clusters[rel] = set()
                    # clusters[rel].add(' '.join(naive_split(replace_mention_holder(data['sentence']))))
                    clusters[rel].add(data['utterance'].rstrip('?'))
    for fn in fn_simple_list:
        with open(fn) as fin:
            for line in fin:
                _, rel, _, _, question = line.decode('utf8').strip().split('\t')
                if rel not in clusters:
                    clusters[rel] = set()
                # clusters[rel].add(' '.join(naive_split(replace_mention_holder(pattern))))
                clusters[rel].add(question)

    # with open(fn_out, 'w') as fout:
    #     for rel, patterns in clusters.iteritems():
    #         print >> fout, ('%s\t%s' % (rel, '\t'.join(patterns))).encode('utf8')
    output = []
    for rel, pattern in clusters.items():
        output.append({'relation': rel, 'pattern': list(pattern)})
    with open(fn_out, 'w') as fout:
        json.dump(output, fout, indent=4)

# Test code
# Use word embedding from relation matcher model
class PatternCluster:
    def __init__(self, fn_cluster, fn_emb):
        clusters = dict()
        with open(fn_cluster) as fin:
            all_data = json.load(fin)
            for data in all_data:
                # line = line.strip().decode('utf8').split('\t')
                clusters[data['relation']] = data['pattern']
        self.embeddings = dict()
        with open(fn_emb) as fin:
            for line in fin:
                word, line = line.strip().decode('utf8').split('\t')
                self.embeddings[word] = np.array(map(float, line.split()))

        self.relation_to_matrix = dict()
        for rel, patterns in clusters.iteritems():
            self.relation_to_matrix[rel] = []
            for p in patterns:
                repr = np.sum([self.embeddings[w] for w in p.split() if w in self.embeddings], 0)
                repr = repr / np.sqrt(np.sum(repr ** 2))
                self.relation_to_matrix[rel].append(repr)

            self.relation_to_matrix[rel] = np.array(self.relation_to_matrix[rel])

    def get_top_k(self, data, k):
        pq = []
        for r, s in data:
            if len(pq) < k:
                heapq.heappush(pq, [r, s])
            elif pq[0][1] < s:
                heapq.heapreplace(pq, [r, s])
        ret = []
        while len(pq):
            ret.append(heapq.heappop(pq))
        return ret

    def get_topk_relation(self, pattern, topk):
        repr = np.sum([self.embeddings[w] for w in pattern.split() if w in self.embeddings], 0)
        repr = repr / np.sqrt(np.sum(repr ** 2))

        rank = []
        for r in self.relation_to_matrix.keys():
            rank.append([r, float(np.max(np.dot(self.relation_to_matrix[r], repr)))])
        if topk == -1:
            rank = sorted(rank, key=lambda x: x[1], reverse=True)
            return rank
        else:
            rank = self.get_top_k(rank, topk)
            return rank[:topk]


def evaluate(fn_cluster_train, fn_embedding, fn_wq):
    cluster = PatternCluster(fn_cluster_train, fn_embedding)
    num_p_at_k = [0. for _ in range(3)]
    num_question = 0
    average_rank_index = 0.
    average_candidate_count = 0.
    with open(fn_wq) as fin:
        wq = json.load(fin)
        for data in wq:
            # pattern = ' '.join(naive_split(replace_mention_holder(data['sentence']))) # palceholder should be <$>
            pattern = replace_mention_holder(data['sentence'])
            positive_relations = set()
            for path in data['paths']:
                if path[1] == "forward_pass_cvt" or path[1] == "forward_direct":
                    core_path = path[0].split()
                    if len(core_path) == 3:
                        rel = core_path[1]
                    else:
                        rel = core_path[1] + ' ' + core_path[3]
                    positive_relations.add(rel)
            if len(positive_relations) == 0:
                continue
            rank = cluster.get_topk_relation(pattern, -1)
            rel_to_score = dict(rank)
            pos_score = -1.
            for r in positive_relations:
                if r not in rel_to_score:
                    # print "{} doesn't exists in rel_to_score".format(r)
                    continue
                if rel_to_score[r] > pos_score:
                    pos_score = rel_to_score[r]
            if pos_score == 0:
                print "relations of question '{}' never appear: {}".format(pattern, positive_relations)

            rank_index = 1
            for r, s in rank:
                if s > pos_score:
                    rank_index += 1

            for j in range(3):
                if rank_index <= j + 1:  # j start from 0
                    num_p_at_k[j] += 1

            average_rank_index += rank_index
            average_candidate_count += len(rank)

            num_question += 1
        for i in range(3):
            print "p@{}: {}".format(i+1, num_p_at_k[i] * 1.0 / num_question)
        print 'average of rank', average_rank_index * 1.0 / num_question
        print "Number of question", num_question


if __name__ == '__main__':
    fn_pattern_cluster = '../data/pattern_cluster/pattern.cluster.train.json'
    fn_question_cluster = '../data/pattern_cluster/question.cluster.train.json'
    fn_wq_test = '../data/wq.test.complete.v2'
    fn_embedding = 'runs/word-add-200-exact-d0.8/word.embeddings'
    # gen_pattern_cluster(['../data/wq.train.complete.v2', '../data/wq.dev.complete.v2'],
    #                     ['../data/simple.train.dev.el.v2'],
    #                     fn_pattern_cluster)
    #
    # gen_question_cluster(['../data/wq.train.complete.v2', '../data/wq.dev.complete.v2'],
    #                      ['../data/simple.train.dev.el.v2'],
    #                      '../data/pattern_cluster/question.cluster.train.json')

    evaluate(fn_question_cluster, fn_embedding, fn_wq_test)
