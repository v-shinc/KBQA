
import json
import random
import numpy as np

def load_word_list(fn_word):
    word2idx = dict()
    idx = 0
    with open(fn_word) as fin:
        for line in fin:
            w = line.decode('utf8').strip()
            if w in word2idx:
                continue
            word2idx[w] = idx
            idx += 1
    return word2idx


class WordConverter(object):
    def __init__(self, fn_word):
        self.word2idx = load_word_list(fn_word)
        print len(self.word2idx)
        self.idx2word = dict()
        for w, i in self.word2idx.iteritems():
            self.idx2word[i] = w

    def words2indices(self, words):
        if not isinstance(words, list):
            words = words.split()
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def indices2words(self, indices):
        if isinstance(indices, str):
            indices = indices.split()
        return [self.idx2word[i] for i in indices if i in self.idx2word]

    @property
    def num_word(self):
        return len(self.idx2word)

def load_description(fn_in, converter):
    ent2desc = {}
    with open(fn_in) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            ent2desc[ll[0]] = []
            for s in ll[1:]:
                indices = converter.words2indices(s)
                if len(indices) < 3:
                    continue
                ent2desc[ll[0]].append(indices)
    return ent2desc

import leveldb
class MNTrainData(object):
    def __init__(self, fn_train, word_converter, sentence_len, num_sentence):
        self.wc = word_converter

        self.file = open(fn_train)
        self.cur_line = 0
        self.num_line = 0
        for _ in self.file:
            self.num_line += 1
        self.file.seek(0)
        self.PAD = self.wc.num_word
        self.sentence_len = sentence_len
        self.num_sentence = num_sentence
        self.db = leveldb.LevelDB('../db/description.db')
        self.entities = []

    def get_entities_from_db(self):
        for key, _ in self.db:
            self.entities.append(key)

    def get_description(self, mid):
        try:
            description =  self.db.Get(mid)
            ret = []
            for s in description.decode('utf8').split('\t'):
                s = self.wc.words2indices(s)
                if len(s) < 3:
                    continue
                ret.append(s)
            return ret
        except KeyError:
            return None


    def add_pad(self, indices_list, max_len):
        for i, l in enumerate(indices_list):
            indices_list[i] = indices_list[i][:max_len]
            indices_list[i] += [self.PAD] * (max_len - len(l))


    def next_batch(self, batch_size):
        questions = []
        descriptions = []
        labels = []
        while len(questions) < batch_size:
            if self.cur_line == self.num_line:
                self.file.seek(0)
                self.cur_line = 0
            line = self.file.readline()
            self.cur_line += 1
            data = json.loads(line, encoding='utf8')
            if data['q'][:1] == '?':
                ques = data['q'][:-1]
            else:
                ques = data['q']

            ques = self.wc.words2indices(ques)
            if len(ques) == 0:
                continue

            desc = self.get_description(data['pos'])
            if not desc:
                # print data['pos'], 'has no description'
                continue

            questions.append(ques)
            descriptions.append(ques)
            labels.append(1)

            num_neg = 0
            for neg in data['neg']:
                desc = self.get_description(neg)
                if not desc:
                    continue
                questions.append(ques)
                descriptions.append(self.ent2desc[neg])
                labels.append(0)
                num_neg += 1
            if num_neg == 0:
                negs = random.sample(self.entities, 20)
                for neg in negs:
                    if neg not in self.ent2desc:
                        continue
                    questions.append(ques)
                    descriptions.append(self.ent2desc[neg])
                    labels.append(0)


            # neg = random.sample(data['neg'], 1)
            # while neg not in self.ent2desc:
            #     neg = random.sample(data['neg'], 1)
            #
            # questions.append(ques)
            # descriptions.append(self.ent2desc[neg])
            # labels.append(0)

        self.add_pad(questions, self.sentence_len)
        for i in xrange(len(descriptions)):
            descriptions[i] = descriptions[i][:self.num_sentence]
            while len(descriptions[i]) < self.num_sentence:
                descriptions[i].append([])

            self.add_pad(descriptions[i], self.sentence_len)


        return [np.array(e) for e in [questions, descriptions, labels]]



class MNTestData(object):
    def __init__(self, fn_test, ent2desc, word_converter, sentence_len, num_sentence):
        self.wc = word_converter
        self.ent2desc = ent2desc
        self.fn_test = fn_test
        self.PAD = len(ent2desc)
        self.sentence_len = sentence_len
        self.num_sentence = num_sentence
        self.entities = self.ent2desc.keys()

    def add_pad(self, indices_list, max_len):
        for i, l in enumerate(indices_list):
            indices_list[i] = indices_list[i][:max_len]
            indices_list[i] += [self.PAD] * (max_len - len(l))

    def __call__(self, verbose=False):

        no_keyword = 0
        lack_pos = 0
        lack_neg = 0
        if verbose:
            file = open('evaluate.res', 'w')
        with open(self.fn_test) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                if data['q'][:1] == '?':
                    ques = data['q'][:-1]
                else:
                    ques = data['q']
                ques = self.wc.words2indices(ques)

                if len(ques) == 0:
                    if verbose:
                        no_keyword += 1
                        print >> file, '%s: keyword is empty' % data['q']
                    continue

                if data['pos'] not in self.ent2desc:
                    if verbose:
                        lack_pos += 1
                        print >> file, '%s: %s has no description' % (data['q'], data['pos'])
                    continue

                descriptions = []
                descriptions.append(self.ent2desc[data['pos']])

                not_found = []
                for e in data['neg']:
                    if e in self.ent2desc:
                        descriptions.append(self.ent2desc[e])
                    elif verbose:
                        not_found.append(e)

                if len(not_found) > 0:
                    lack_neg += 1
                    print >> file, '%s: [%s] have no descriptions' % (data['q'], ','.join(not_found))
                if len(data['neg']) == 1:
                    negs = random.sample(self.entities, 20)
                    for neg in negs:
                        if neg not in self.ent2desc:
                            continue
                        descriptions.append(self.ent2desc[neg])
                questions = [ques] * len(descriptions)
                self.add_pad(questions, self.sentence_len)
                for i in xrange(len(descriptions)):
                    descriptions[i] = descriptions[i][:self.num_sentence]
                    while len(descriptions[i]) < self.num_sentence:
                        descriptions[i].append([])

                    self.add_pad(descriptions[i], self.sentence_len)


                yield np.array(questions), np.array(descriptions)
        if verbose:
            print >> file, '#question is empty after being transformed to indices', no_keyword
            print >> file, '#positive entities lack descriptions', lack_pos
            print >> file, '#negative entities lack descriptions', lack_neg
            file.close()

    def evaluate(self, model, verbose=False):
        average_rank = 0
        p_at_1 = 0
        num = 0
        average_candidate_num = 0
        for q, d in self.__call__(verbose):
            scores = model.predict(q, d)
            rank = 1
            for i in range(1, len(scores)):
                if scores[i] > scores[0]:
                    rank += 1
            average_rank += rank * 1.
            average_candidate_num += len(scores)
            if rank == 1:
                p_at_1 += 1
            num += 1
        average_rank = average_rank / num
        p_at_1 = p_at_1 * 1. / num
        average_candidate_num = average_candidate_num * 1.0 / num
        print "num =", num, "p@1 =", p_at_1, 'average_rank =', average_rank, 'average candidate num', average_candidate_num
        return p_at_1