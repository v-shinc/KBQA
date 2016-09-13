import json
import random
from numpy import random
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


def load_description(filename, converter):
    ent2des = dict()
    with open(filename) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if len(ll) != 2:
                continue
            ent, desc = ll
            ent2des[ent] = converter.words2indices(desc.split())
    return ent2des


class TrainData(object):
    def __init__(self, fn_train, ent2desc, word_converter, max_ques_len, max_desc_len):
        self.wc = word_converter
        self.ent2desc = ent2desc
        self.file = open(fn_train)
        self.cur_line = 0
        self.num_line = 0
        for _ in self.file:
            self.num_line += 1
        self.file.seek(0)
        self.PAD = self.wc.num_word
        self.max_ques_len = max_ques_len
        self.max_desc_len = max_desc_len

    def add_pad(self, indices_list, max_len):
        for i, l in enumerate(indices_list):
            indices_list[i] = indices_list[i][:max_len]
            indices_list[i] += [self.PAD] * (max_len - len(l))

    def next_batch(self, batch_size):
        questions = []
        descriptions = []
        neg_descriptions = []
        while len(questions) < batch_size:
            if self.cur_line == self.num_line:
                self.file.seek(0)
                self.cur_line = 0
            line = self.file.readline()
            self.cur_line += 1
            data = json.loads(line, encoding='utf8')

            ques = self.wc.words2indices(data['Q'])
            if len(ques):
                continue

            if data['TOPIC'] not in self.ent2desc:
                continue

            neg = random.sample(data['NEG'], 1)
            while neg not in self.ent2desc:
                neg = random.sample(data['NEG'], 1)


            questions.append(ques)
            descriptions.append(self.ent2desc[data['TOPIC']])
            neg_descriptions.append(self.ent2desc[neg])

        self.add_pad(questions, self.max_ques_len)
        self.add_pad(descriptions, self.max_desc_len)
        self.add_pad(neg_descriptions, self.max_desc_len)
        return [np.array(e) for e in [questions, descriptions, neg_descriptions]]


class TestData(object):
    def __init__(self, fn_test, ent2desc, word_converter, ques_len, desc_len):
        self.wc = word_converter
        self.ent2desc = ent2desc
        self.fn_test = fn_test
        self.PAD = len(ent2desc)
        self.ques_len = ques_len
        self.desc_len = desc_len

    def add_pad(self, indices_list, max_len):
        for i, l in enumerate(indices_list):
            indices_list[i] = indices_list[i][:max_len]
            indices_list[i] += [self.PAD] * (max_len - len(l))

    def __call__(self):
        with open(self.fn_test) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                ques = self.wc.words2indices(data['Q'])

                if len(ques) == 0:
                    continue

                if data['TOPIC'] not in self.ent2desc:
                    continue

                descriptions = []
                descriptions.append(self.ent2desc(data['TOPIC']))
                for e in data['NEG']:
                    descriptions.append(self.ent2desc[e])
                self.add_pad(descriptions, self.desc_len)
                questions = self.add_pad([ques] * len(descriptions), self.ques_len)
                yield np.array(questions), np.array(descriptions)

    def evaluate(self, model):
        average_rank = 0
        num = 0
        for q, d in self():
            scores = model.predict(q, d)

            rank = 1

            for i in range(1, len(scores)):
                if scores[i] > scores[0]:
                    rank += 1
            average_rank += rank * 1. / len(scores)
            num += 1
        average_rank = average_rank / num
        print "p@1 =", average_rank
        return average_rank



