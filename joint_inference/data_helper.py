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
            self.unknown_id = len(self.word_to_id)
            self.word_padding = len(self.word_to_id) + 1

        self.char_based = False
        self.max_sentence_len = params['max_sentence_len']
        # if 'char_dim' in params['question_config']:
        self.char_based = True
        self.char_to_id, self.id_to_char = load_mapping(params['fn_char'])
        index = len(self.char_to_id)
        self.char_to_id[' '] = index
        self.id_to_char[index] = ' '
        self.char_padding = len(self.char_to_id)

        self.max_word_len = params['max_word_len'] # for pattern char-based rnn or cnn
        self.max_name_len = params['max_name_len']

        self.sub_relation_to_id, self.id_to_sub_relation = load_mapping(params['fn_sub_relation'])
        relation_to_id, _ = load_mapping(params['fn_relation'])
        self.relations = [r for r in relation_to_id.keys() if len(r.split('.')) == 3]
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

    @property
    def num_word(self):
        return len(self.word_to_id) + 2 if self.word_based else 0

    @property
    def num_char(self):
        return len(self.char_to_id) + 1 if self.char_based else 0

    @property
    def num_relation(self):
        return len(self.sub_relation_to_id)

    def _parse(self, data):
        parsed = dict()
        # mention VS topic name
        parsed['mention_char_ids'] = [self.char_to_id[c] for c in data['mention'].lower() if c in self.char_to_id]
        if 'topic_name' not in data:
            if data['f1'] > 0:
                print "topic name not in", data['question']
            return None
        parsed['topic_char_ids'] = [self.char_to_id[c] for c in data['topic_name'].lower() if c in self.char_to_id]

        # pattern VS relation
        if self.word_based:
            parsed['pattern_word_ids'] = [self.word_to_id.get(w, self.unknown_id) for w in data['pattern'].split()]

        if self.char_based:
            parsed['pattern_char_ids'] = [[self.char_to_id[c] for c in w if self.char_to_id]
                                for w in data['pattern'].split()]
            # pattern_char_ids, word_lengths = self.pad_chars(parsed['pattern_char_ids'], self.max_word_len)
            # parsed['pattern_word_lengths'] = word_lengths

        relation_ids = [self.sub_relation_to_id[r] for r in data['relation'].split('.')[-3:] if
                        r in self.sub_relation_to_id]

        if len(relation_ids) != 3:
            return None
        parsed['relation_ids'] = relation_ids

        # Extra feature
        extra_features = [float(data.get(k, 0)) for k in self.params['extra_keys']]

        # extra_features.append(float(data['constraint_entity_in_q']))
        # extra_features.append(float(data['constraint_entity_word']))
        # extra_features.append(float(data['entity_score']))
        # extra_features.append(float(data['rel_pat_overlap']))
        # extra_features.append(float(data['num_answer']))
        parsed['extra'] = extra_features
        parsed['f1'] = data['f1']
        return parsed

    def test_iterator(self, fn_test):
        qid = -1
        ret = {
            "pattern_word_ids": [],
            "sentence_lengths": [],
            "relation_ids": [],
            "mention_char_ids": [],
            "topic_char_ids": [],
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
                ret['pattern_word_ids'].append(self.pad_words(parsed['pattern_word_ids'], self.max_sentence_len, self.word_padding))
                ret['relation_ids'].append(parsed['relation_ids'])
                ret['mention_char_ids'].append(self.pad_words(parsed['mention_char_ids'], self.max_name_len, self.char_padding))
                ret['topic_char_ids'].append(self.pad_words(parsed['topic_char_ids'], self.max_name_len, self.char_padding))
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
        qids = self.qid_to_rank.keys()

        num_batch = self.num // batch_size + int(self.num % batch_size > 0)
        for _ in xrange(num_batch):
            all_pattern_word_ids = []
            all_sentence_lengths = []
            all_relation_ids = []
            all_mention_char_ids = []
            all_topic_char_ids = []
            all_extras = []

            def wrap(qs):
                all_sentence_lengths.append([len(qs[i]['pattern_word_ids']) for i in [0, 1]])
                all_pattern_word_ids.append([self.pad_words(qs[i]['pattern_word_ids'], self.max_sentence_len, self.word_padding) for i in [0, 1]])
                all_relation_ids.append([qs[i]['relation_ids'] for i in [0, 1]])
                all_mention_char_ids.append([self.pad_words(qs[i]['mention_char_ids'], self.max_name_len, self.char_padding) for i in [0, 1]])
                all_topic_char_ids.append([self.pad_words(qs[i]['topic_char_ids'], self.max_name_len, self.char_padding) for i in [0, 1]])
                all_extras.append([qs[i]['extra'] for i in [0, 1]])

            while len(all_relation_ids) < batch_size:
                if index == self.num:
                    random.shuffle(qids)
                    index = 0
                qid = qids[index]
                index += 1

                pairs = []
                if len(self.qid_to_rank[qid]) < 2:
                    print 'len(qid_to_rank[{}])'.format(qid), "< 2"
                # while j < len(self.qid_to_rank[qid]) - 1:
                #     h1 = self.qid_to_rank[qid][j]
                #     h2 = self.qid_to_rank[qid][j+1]
                #     if self.queries[(qid, h2)]['f1'] == 0.:
                #         break
                #     if self.queries[(qid, h1)]['f1'] > self.queries[(qid, h2)]['f1']:
                #         pairs.append((h1, h2))
                #     j += 1
                # num_pos = min(3, j)
                # if num_pos == 0:
                #     continue
                # pairs.extend(zip(self.qid_to_rank[qid][:num_pos], random.sample(self.qid_to_rank[qid][j:], num_pos)))
                for i in xrange(len(self.qid_to_rank[qid])):
                    for j in xrange(len(self.qid_to_rank[qid])):
                        h1 = self.qid_to_rank[qid][i]
                        h2 = self.qid_to_rank[qid][j]
                        if self.queries[(qid, h1)]['f1'] > self.queries[(qid, h2)]['f1']:
                            # pairs.append((h1, h2))
                            wrap([self.queries[(qid, h1)], self.queries[(qid, h2)]])

                # for h1, h2 in random.sample(pairs, min(len(pairs), 20)):
                #     wrap([self.queries[(qid, h1)], self.queries[(qid, h2)]])
            ret = {
                "pattern_word_ids": all_pattern_word_ids,
                "sentence_lengths": all_sentence_lengths,
                "relation_ids": all_relation_ids,
                "mention_char_ids": all_mention_char_ids,
                "topic_char_ids": all_topic_char_ids,
                "extras": all_extras,
            }

            for k, v in ret.items():
                ret[k] = np.array(v)
            yield ret

    def pad_chars(self, char_ids, max_word_len):
        char_ids = char_ids[:self.max_sentence_len]
        if len(char_ids) < self.max_sentence_len:
            char_ids += [[self.char_padding] * 1] * (self.max_sentence_len - len(char_ids))

        word_lengths = []
        for i, w in enumerate(char_ids):
            w = w[:max_word_len]
            padding = [self.char_padding] * (max_word_len - len(w))
            word_lengths.append(len(w))
            char_ids[i] = w + padding
        return char_ids, word_lengths

    def pad_words(self, seq, max_sentence_len, padding):
        seq = seq[:max_sentence_len]
        return seq + [padding] * (max_sentence_len - len(seq))