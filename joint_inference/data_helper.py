import numpy as np
import json
import math
import random

def load_mapping_and_count(fn_word):
    index = 0
    word_to_id = dict()
    id_to_word = dict()
    word_to_count = dict()
    with open(fn_word) as fin:
        for line in fin:
            w, cnt = line.decode('utf8').strip().split('\t')
            word_to_id[w] = index
            id_to_word[index] = w
            word_to_count[w] = cnt
            index += 1
    return word_to_id, id_to_word, word_to_count

def load_mapping(fn):
    index = 0
    xx_to_id = dict()
    id_to_xx = dict()
    with open(fn) as fin:
        for line in fin:
            w = line.decode('utf8').strip()
            xx_to_id[w] = index
            id_to_xx[index] = w
            index += 1
    return xx_to_id, id_to_xx


class DataSet:
    def __init__(self, params):
        self.word_based = False
        if 'word_dim' in params['pattern_config']:
            self.word_based = True
            self.word_to_id, self.id_to_word = load_mapping(params['fn_word'])
            if '<$>' not in self.word_to_id:
                index = len(self.word_to_id)
                self.word_to_id['<$>'] = index
                self.id_to_word[index] = '<$>'
            index = len(self.word_to_id)
            self.word_to_id['<UNK>'] = index
            self.id_to_word['index'] = '<UNK>'
            self.word_padding = len(self.word_to_id)

        self.char_based = False
        self.max_pattern_len = params['max_pattern_len']

        # if 'char_dim' in params['question_config']:
        self.char_based = True
        self.char_to_id, self.id_to_char = load_mapping(params['fn_char'])
        index = len(self.char_to_id)
        self.char_to_id[' '] = index
        self.id_to_char[index] = ' '
        self.char_padding = len(self.char_to_id)

        if 'type_config' in params:
            self.type_to_id, self.id_to_type = load_mapping(params['fn_type'])
            self.type_padding = len(self.type_to_id)
            self.max_question_len = params['max_question_len']
            self.max_type_len = params['max_type_len']

        self.max_word_len = params['max_word_len']  # for pattern char-based rnn or cnn
        self.max_name_len = params['max_name_len']

        self.sub_relation_to_id, self.id_to_sub_relation = load_mapping(params['fn_sub_relation'])
        self.sub_relation_padding = len(self.sub_relation_to_id)  # added

        relation_to_id, _ = load_mapping(params['fn_relation'])
        self.relations = [r for r in relation_to_id.keys()]
        self.params = params
        self.qid_to_rank = dict()
        self.queries = dict()
        with open(params['fn_train']) as fin:
            for line in fin:
                data = json.loads(line)
                qid = data['qid']
                if qid not in self.qid_to_rank:
                    self.qid_to_rank[qid] = []
                parsed = self._parse(data)
                if parsed:
                    self.qid_to_rank[qid].append(data['hash'])
                    self.queries[(qid, data['hash'])] = parsed
        self.num = len(self.qid_to_rank)
        for qid in self.qid_to_rank:
            self.qid_to_rank[qid] = sorted(self.qid_to_rank[qid], key=lambda h: self.queries[(qid, h)]['f1'],
                                           reverse=True)  # sort rank according to f1 score
        self.qids = self.qid_to_rank.keys()

    @property
    def num_word(self):
        return len(self.word_to_id) + 1 if self.word_based else 0

    @property
    def num_char(self):
        return len(self.char_to_id) + 1 if self.char_based else 0

    @property
    def num_relation(self):
        return len(self.sub_relation_to_id) + 1  # added

    @property
    def num_type(self):
        return len(self.type_to_id) + 1

    def _parse(self, data):
        parsed = dict()
        # mention VS topic name
        if "topic_config" in self.params:
            parsed['mention_char_ids'] = [self.char_to_id[c] for c in data['mention'].lower() if c in self.char_to_id]
            if 'topic_name' not in data:
                if data['f1'] > 0:
                    print "topic name not in", data['question']
                return None
            parsed['topic_char_ids'] = [self.char_to_id[c] for c in data['topic_name'].lower() if c in self.char_to_id]

        # pattern VS relation
        if self.word_based:
            parsed['pattern_word_ids'] = [self.word_to_id.get(w if w in self.word_to_id else '<UNK>') for w in data['pattern'].split()]

        if self.char_based:
            parsed['pattern_char_ids'] = [[self.char_to_id[c] for c in w if self.char_to_id]
                                for w in data['pattern'].split()]
            # pattern_char_ids, word_lengths = self.pad_chars(parsed['pattern_char_ids'], self.max_word_len)
            # parsed['pattern_word_lengths'] = word_lengths

        # relation_ids = [self.sub_relation_to_id[r] for r in data['relation'].split('.')[-1:] if
        #                 r in self.sub_relation_to_id]

        # question VS type
        if 'type_config' in self.params:
            parsed['question_word_ids'] = [self.word_to_id.get(w if w in self.word_to_id else '<UNK>') for w in data['question'].split()]
            parsed['type_ids'] = [self.type_to_id.get(t) for t in data['topic_type']]
            if len(parsed['type_ids']) == 0:
                parsed['type_ids'] = [self.type_padding]
        # relation is cast-actor or place_of_birth
        if len(data['path']) == 2:
            relation = [data['path'][0][1].split('.')[-1], data['path'][1][1].split('.')[-1]]
        else:
            relation = [data['path'][0][1].split('.')[-1]]

        relation_ids = [self.sub_relation_to_id[r] for r in relation]

        if len(relation_ids) == 0:
            raise ValueError("len(relation_ids) == 0")

        parsed['relation_ids'] = relation_ids

        # Extra feature
        extra_features = [float(data.get(k, 0)) for k in self.params['extra_keys']]
        parsed['extra'] = extra_features
        parsed['f1'] = data['f1']
        return parsed

    def test_iterator(self, fn_test):
        qid = -1
        ret = {
            "pattern_word_ids": [],
            "sentence_lengths": [],
            "relation_ids": [],
            "relation_lengths": [],
            "mention_char_ids": [],
            "topic_char_ids": [],
            "mention_lengths": [],
            "topic_lengths": [],
            "question_word_ids": [],
            "question_lengths": [],
            "type_ids": [],
            "type_lengths": [],
            "extras": [],
            "f1": [],
            "pattern_words": [],
            "paths": [],
            "mentions": [],
            "question": ""
        }
        with open(fn_test) as fin:
            for line in fin:
                data = json.loads(line)
                if data['qid'] != qid:
                    if len(ret['pattern_word_ids']) > 0:
                        for k, v in ret.items():
                            ret[k] = np.array(v)
                        yield ret
                    for k in ret.keys():
                        if k == 'question':
                            ret[k] = ""
                        else:
                            ret[k] = []
                    qid = data['qid']
                    ret['question'] = data['question']
                parsed = self._parse(data)
                if parsed == None:
                    continue
                ret['sentence_lengths'].append(len(parsed['pattern_word_ids']))
                ret['pattern_word_ids'].append(self.pad_words(parsed['pattern_word_ids'], self.max_pattern_len, self.word_padding))
                ret['relation_lengths'].append(len(parsed['relation_ids']))
                ret['relation_ids'].append(self.pad_words(parsed['relation_ids'], 2, self.sub_relation_padding))
                if "type_config" in self.params:
                    ret['question_lengths'].append(len(parsed['question_word_ids']))
                    ret['question_word_ids'].append(self.pad_words(parsed['question_word_ids'], self.max_question_len, self.word_padding))
                    ret['type_lengths'].append(len(parsed['type_ids']))
                    ret['type_ids'].append(self.pad_words(parsed['type_ids'], self.max_type_len, self.type_padding))
                if "topic_config" in self.params:
                    ret['mention_char_ids'].append(self.pad_words(parsed['mention_char_ids'], self.max_name_len, self.char_padding))
                    ret['topic_char_ids'].append(self.pad_words(parsed['topic_char_ids'], self.max_name_len, self.char_padding))
                    ret['mention_lengths'].append(len(parsed['mention_char_ids']))
                    ret['topic_lengths'].append(len(parsed['topic_char_ids']))
                ret['extras'].append(parsed['extra'])
                ret['f1'].append(parsed['f1'])
                ret['pattern_words'].append(data['pattern'])
                ret['paths'].append(data['path'])
                ret['mentions'].append(data['mention'])
        if len(ret['pattern_word_ids']) > 0:
            for k, v in ret.items():
                ret[k] = np.array(v)
            yield ret

    def train_batch_iterator(self, batch_size):
        index = 0

        num_batch = self.num // batch_size + int(self.num % batch_size > 0)

        all_pattern_word_ids = [[], []]
        all_sentence_lengths = [[], []]
        all_relation_ids = [[], []]
        all_relation_lengths = [[], []]
        all_mention_char_ids = [[], []]
        all_topic_char_ids = [[], []]
        all_mention_lengths = [[], []]
        all_topic_lengths = [[], []]
        all_question_word_ids = [[], []]
        all_question_lengths = [[], []]
        all_type_ids = [[], []]
        all_type_lengths = [[], []]
        all_extras = [[], []]

        def wrap(pair):
            for i in [0, 1]:
                all_sentence_lengths[i].append(len(pair[i]['pattern_word_ids']))
                all_pattern_word_ids[i].append(
                    self.pad_words(pair[i]['pattern_word_ids'], self.max_pattern_len, self.word_padding))
                all_relation_lengths[i].append(len(pair[i]['relation_ids']))
                all_relation_ids[i].append(self.pad_words(pair[i]['relation_ids'], 2, self.sub_relation_padding))
                if "topic_config" in self.params:
                    all_mention_char_ids[i].append(
                        self.pad_words(pair[i]['mention_char_ids'], self.max_name_len, self.char_padding))
                    all_topic_char_ids[i].append(
                        self.pad_words(pair[i]['topic_char_ids'], self.max_name_len, self.char_padding))
                    all_mention_lengths[i].append(len(pair[i]['mention_char_ids']))
                    all_topic_lengths[i].append(len(pair[i]['topic_char_ids']))
                if "type_config" in self.params:
                    all_question_lengths[i].append(len(pair[i]['question_word_ids']))
                    all_question_word_ids[i].append(self.pad_words(pair[i]['question_word_ids'], self.max_question_len, self.word_padding))
                    all_type_lengths[i].append(len(pair[i]['type_ids']))
                    all_type_ids[i].append(self.pad_words(pair[i]['type_ids'], self.max_type_len, self.type_padding))
                all_extras[i].append(pair[i]['extra'])

        for _ in xrange(num_batch):
            all_pattern_word_ids = [[], []]
            all_sentence_lengths = [[], []]
            all_relation_ids = [[], []]
            all_relation_lengths = [[], []]
            all_mention_char_ids = [[], []]
            all_topic_char_ids = [[], []]
            all_mention_lengths = [[], []]
            all_topic_lengths = [[], []]
            all_question_word_ids = [[], []]
            all_question_lengths = [[], []]
            all_type_ids = [[], []]
            all_type_lengths = [[], []]
            all_extras = [[], []]
            num_diff_question = 0
            while num_diff_question < batch_size:
                if index == self.num:
                    index = 0
                    random.shuffle(self.qids)

                qid = self.qids[index]
                index += 1

                pairs = []
                if len(self.qid_to_rank[qid]) < 2:
                    print 'len(qid_to_rank[{}])'.format(qid), "< 2"

                for i in xrange(len(self.qid_to_rank[qid])):
                    for j in xrange(i+1, len(self.qid_to_rank[qid])):
                        h1 = self.qid_to_rank[qid][0]
                        h2 = self.qid_to_rank[qid][j]
                        if self.queries[(qid, h1)]['f1'] > self.queries[(qid, h2)]['f1']:
                            pairs.append((h1, h2))
                            # wrap([self.queries[(qid, h1)], self.queries[(qid, h2)]])
                if len(pairs) == 0:
                    continue
                k = 5
                if len(pairs) > k:
                    pairs = random.sample(pairs, k)
                # wrap([self.queries[(qid, one[0][0])], self.queries[(qid, one[0][1])]])
                for h1, h2 in pairs:
                    wrap([self.queries[(qid, h1)], self.queries[(qid, h2)]])
                num_diff_question += 1
            ret = {
                "pattern_word_ids": all_pattern_word_ids,
                "sentence_lengths": all_sentence_lengths,
                "relation_ids": all_relation_ids,
                "relation_lengths": all_relation_lengths,
                "mention_char_ids": all_mention_char_ids,
                "topic_char_ids": all_topic_char_ids,
                "mention_lengths": all_mention_lengths,
                "topic_lengths": all_topic_lengths,
                "question_word_ids": all_question_word_ids,
                "question_lengths": all_question_lengths,
                "type_ids": all_type_ids,
                "type_lengths": all_type_lengths,
                "extras": all_extras,
            }
            for k, v in ret.items():
                ret[k] = np.array(v)
            yield ret

    def recover(self, data):

        pattern_words = [[' '.join([self.id_to_word[w] for w in s if w in self.id_to_word]) for s in case] for case in data['pattern_word_ids']]
        print "pattern"
        print pattern_words
        print "sentence_lengths"
        print data['sentence_lengths']
        relation_words = [['.'.join([self.id_to_sub_relation[sr] for sr in r]) for r in case]for case in data['relation_ids']]
        print "relations"
        print relation_words

    def pad_chars(self, char_ids, max_word_len):
        char_ids = char_ids[:self.max_pattern_len]
        if len(char_ids) < self.max_pattern_len:
            char_ids += [[self.char_padding] * 1] * (self.max_pattern_len - len(char_ids))

        word_lengths = []
        for i, w in enumerate(char_ids):
            w = w[:max_word_len]
            padding = [self.char_padding] * (max_word_len - len(w))
            word_lengths.append(len(w))
            char_ids[i] = w + padding
        return char_ids, word_lengths

    def pad_words(self, seq, max_sequence_len, padding):
        seq = seq[:max_sequence_len]
        return seq + [padding] * (max_sequence_len - len(seq))