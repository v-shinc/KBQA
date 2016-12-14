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

def tanh(x):
    return 1 - 2. / (math.exp(2 * x) + 1)

class DataSet:
    def __init__(self, params):
        self.word_based = False
        if 'word_dim' in params['question_config']:
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
        if 'char_dim' in params['question_config']:
            self.char_based = True
            self.char_to_id, self.id_to_char = load_mapping(params['fn_char'])
            self.char_padding = len(self.char_to_id)
            self.max_word_len = params['max_word_len']
        self.sub_relation_to_id, self.id_to_sub_relation = load_mapping(params['fn_sub_relation'])
        relation_to_id, _ = load_mapping(params['fn_relation'])
        self.relations = [r for r in relation_to_id.keys() if len(r.split('.')) == 3]
        self.params = params


    @property
    def num_word(self):
        return len(self.word_to_id) + 2 if self.word_based else 0

    @property
    def num_char(self):
        return len(self.char_to_id) + 1 if self.char_based else 0

    @property
    def num_relation(self):
        return len(self.sub_relation_to_id)

    def create_model_input(self, patterns, relations):
        all_word_ids = []
        all_char_ids = []
        all_word_lengths = []
        all_sentence_lengths = []
        all_relations = []
        all_relation_ids = []

        for pattern, relation in zip(patterns, relations):
            pattern = pattern.split()[:self.max_sentence_len]
            if self.word_based:
                pattern_ids = [self.word_to_id.get(w, self.unknown_id) for w in pattern]
                if len(pattern_ids) == 0:
                    raise ValueError('len(pattern_ids) == 0')
                all_word_ids.append(self.pad_words(pattern_ids, self.word_padding))
            all_sentence_lengths.append(len(pattern))
            if self.char_based:
                char_ids = [[self.char_to_id[c] for c in w if self.char_to_id]
                            for w in pattern]
                char_ids, word_lengths = self.pad_chars(char_ids)
                all_char_ids.append(char_ids)
                all_word_lengths.append(word_lengths)

            all_relations.append(relation)
            all_relation_ids.append([self.sub_relation_to_id[r] for r in relation.split('.')[-3:]])
        ret = {
            "word_ids": all_word_ids,
            "sentence_lengths": all_sentence_lengths,
            "char_ids": all_char_ids,
            "word_lengths": all_word_lengths,
            "relation_ids": all_relation_ids,
            "relations": all_relations
        }
        return ret

    def test_iterator(self, fn_dev):
        with open(fn_dev) as fin:
            for line in fin:
                all_word_ids = []
                all_sentence_lengths = []
                all_char_ids = []
                all_word_lengths = []
                all_relations = []
                all_relation_ids = []

                data = json.loads(line, encoding='utf8')

                # handle positive relation
                for pos_rel in data['pos_relation']:
                    all_relations.append(pos_rel)
                    all_relation_ids.append([self.sub_relation_to_id[r] for r in pos_rel.split('.')[-3:]])
                num_pos = len(data['pos_relation'])

                # handle negative relation
                if 'neg_relation' in data and len(data['neg_relation']) > 0:
                    for neg_rel in data['neg_relation']:
                        all_relations.append(neg_rel)
                        all_relation_ids.append([self.sub_relation_to_id[r] for r in neg_rel.split('.')[-3:]])
                else:
                    for neg_rel in self.relations:
                        all_relations.append(neg_rel)
                        all_relation_ids.append([self.sub_relation_to_id[r] for r in neg_rel.split('.')[-3:]])

                # handle question
                str_words = data['question'].split()[:self.max_sentence_len]
                if self.word_based:

                    word_ids = [self.word_to_id.get(w, self.unknown_id) for w in str_words]
                    if len(word_ids) == 0:
                        raise ValueError('len(word_ids) == 0')
                    all_word_ids = [self.pad_words(word_ids, self.word_padding)] * len(all_relation_ids)
                all_sentence_lengths = [len(str_words)] * len(all_relation_ids)

                if self.char_based:
                    char_ids = [[self.char_to_id[c] for c in w if self.char_to_id]
                                for w in str_words]
                    char_ids, word_lengths = self.pad_chars(char_ids)
                    all_char_ids = [char_ids] * len(all_relation_ids)
                    all_word_lengths = [word_lengths] * len(all_relation_ids)
                ret = {
                    "num_pos": num_pos,
                    "words": ' '.join(str_words),
                    "word_ids": all_word_ids,
                    "sentence_lengths": all_sentence_lengths,
                    "char_ids": all_char_ids,
                    "word_lengths": all_word_lengths,
                    "relation_ids": all_relation_ids,
                    "relations": all_relations
                }
                yield ret

    def train_batch_iterator(self, fn_train, batch_size):
        num = 0
        with open(fn_train) as fin:
            for _ in fin:
                num += 1
        train_file = open(fn_train)
        index = 0
        num_batch = num // batch_size + int(num % batch_size > 0)
        for _ in xrange(num_batch):
            all_words = []
            all_word_ids = []
            all_char_ids = []
            all_word_lengths = []
            all_sentence_lengths = []
            all_pos_relation_ids = []
            all_neg_relation_ids = []
            while len(all_words) < batch_size:
                if index == num:
                    train_file.seek(0)
                    index = 0
                index += 1
                line = train_file.readline()
                data = json.loads(line, encoding='utf8')

                pos_rel = np.random.choice(data['pos_relation'])

                pos_relation_ids = [self.sub_relation_to_id[r] for r in pos_rel.split('.')[-3:] if r in self.sub_relation_to_id]
                if len(pos_relation_ids) != 3:
                    continue

                str_words = data['question'].split()[:self.max_sentence_len]
                if self.word_based:
                    # for w in str_words:
                    #     if w not in self.word_to_id:
                    #         print data['question']
                    #         raise ValueError("{} not in self.word_to_id".format(w))
                    word_ids = [self.word_to_id.get(w, self.unknown_id) for w in str_words]
                    if len(word_ids) == 0:
                        raise ValueError('len(word_ids) == 0')
                    all_word_ids.append(self.pad_words(word_ids, self.word_padding))
                all_sentence_lengths.append(len(str_words))
                all_words.append(str_words)
                if self.char_based:
                    char_ids = [[self.char_to_id[c] for c in w if self.char_to_id]
                                for w in str_words]
                    char_ids, word_lengths = self.pad_chars(char_ids)
                    all_char_ids.append(char_ids)
                    all_word_lengths.append(word_lengths)

                # if 'neg_relation' in data and len(data['neg_relation']) > 0:
                # while True:
                #     neg_rel = np.random.choice(self.relations)
                #     if neg_rel not in data['pos_relation']:
                #         break

                # if np.random.uniform(0, 1) < tanh(len(data['neg_relation']) * 1. / 180):
                if len(data['neg_relation']) > 0:
                    neg_rel = np.random.choice(data['neg_relation'])
                else:
                    while True:
                        neg_rel = np.random.choice(self.relations)
                        if neg_rel not in data['pos_relation']:
                            break
                neg_relation_ids = [self.sub_relation_to_id[r] for r in neg_rel.split('.')[-3:]]
                all_pos_relation_ids.append(pos_relation_ids)
                all_neg_relation_ids.append(neg_relation_ids)
            ret = {
                "words": all_words,
                "word_ids": all_word_ids,
                "sentence_lengths": all_sentence_lengths,
                "char_ids": all_char_ids,
                "word_lengths": all_word_lengths,
                "pos_relation_ids": all_pos_relation_ids,
                "neg_relation_ids": all_neg_relation_ids
            }
            for k, v in ret.items():
                ret[k] = np.array(v)
            yield ret
        train_file.close()

    def train_shuffled_batch_iterator(self, fn_train, batch_size):
        num = 0
        all_data = []
        with open(fn_train) as fin:
            for line in fin:
                all_data.append(json.loads(line))
                num += 1
        train_file = open(fn_train)
        index = 0
        num_batch = num // batch_size + int(num % batch_size > 0)
        for _ in xrange(num_batch):
            all_words = []
            all_word_ids = []
            all_char_ids = []
            all_word_lengths = []
            all_sentence_lengths = []
            all_pos_relation_ids = []
            all_neg_relation_ids = []
            while len(all_words) < batch_size:
                if index == num:
                    random.shuffle(all_data)
                    index = 0
                data = all_data[index]
                index += 1
                pos_rel = np.random.choice(data['pos_relation'])

                pos_relation_ids = [self.sub_relation_to_id[r] for r in pos_rel.split('.')[-3:] if r in self.sub_relation_to_id]
                if len(pos_relation_ids) != 3:
                    continue

                str_words = data['question'].split()[:self.max_sentence_len]
                if self.word_based:
                    # for w in str_words:
                    #     if w not in self.word_to_id:
                    #         print data['question']
                    #         raise ValueError("{} not in self.word_to_id".format(w))
                    word_ids = [self.word_to_id.get(w, self.unknown_id) for w in str_words]
                    if len(word_ids) == 0:
                        raise ValueError('len(word_ids) == 0')
                    all_word_ids.append(self.pad_words(word_ids, self.word_padding))
                all_sentence_lengths.append(len(str_words))
                all_words.append(str_words)
                if self.char_based:
                    char_ids = [[self.char_to_id[c] for c in w if self.char_to_id]
                                for w in str_words]
                    char_ids, word_lengths = self.pad_chars(char_ids)
                    all_char_ids.append(char_ids)
                    all_word_lengths.append(word_lengths)

                # if 'neg_relation' in data and len(data['neg_relation']) > 0:
                # while True:
                #     neg_rel = np.random.choice(self.relations)
                #     if neg_rel not in data['pos_relation']:
                #         break

                # if np.random.uniform(0, 1) < tanh(len(data['neg_relation']) * 1. / 180):
                if len(data['neg_relation']) > 0:
                    neg_rel = np.random.choice(data['neg_relation'])
                else:
                    while True:
                        neg_rel = np.random.choice(self.relations)
                        if neg_rel not in data['pos_relation']:
                            break
                neg_relation_ids = [self.sub_relation_to_id[r] for r in neg_rel.split('.')[-3:]]
                all_pos_relation_ids.append(pos_relation_ids)
                all_neg_relation_ids.append(neg_relation_ids)
            ret = {
                "words": all_words,
                "word_ids": all_word_ids,
                "sentence_lengths": all_sentence_lengths,
                "char_ids": all_char_ids,
                "word_lengths": all_word_lengths,
                "pos_relation_ids": all_pos_relation_ids,
                "neg_relation_ids": all_neg_relation_ids
            }
            for k, v in ret.items():
                ret[k] = np.array(v)
            yield ret
        train_file.close()

    def pad_chars(self, char_ids):
        char_ids = char_ids[:self.max_sentence_len]
        if len(char_ids) < self.max_sentence_len:
            char_ids += [[self.char_padding] * 1] * (self.max_sentence_len - len(char_ids))

        word_lengths = []
        for i, w in enumerate(char_ids):
            w = w[:self.max_word_len]
            padding = [self.char_padding] * (self.max_word_len - len(w))
            word_lengths.append(len(w))
            char_ids[i] = w + padding
        return char_ids, word_lengths

    def pad_words(self, seq, padding):
        seq = seq[:self.max_sentence_len]
        return seq + [padding] * (self.max_sentence_len - len(seq))


class RankerDataSet:
    def __init__(self, params):
        self.word_based = False
        if 'word_dim' in params['question_config']:
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
        if 'char_dim' in params['question_config']:
            self.char_based = True
            self.char_to_id, self.id_to_char = load_mapping(params['fn_char'])
            self.char_padding = len(self.char_to_id)
            self.max_word_len = params['max_word_len']
        self.sub_relation_to_id, self.id_to_sub_relation = load_mapping(params['fn_sub_relation'])
        relation_to_id, _ = load_mapping(params['fn_relation'])
        self.relations = [r for r in relation_to_id.keys() if len(r.split('.')) == 3]
        self.params = params

        self.num = 0
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
                self.num += 1
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
        parsed['topic_char_ids'] = [self.char_to_id[c] for c in data['topic_name'].lower() if c in self.char_to_id]

        # pattern VS relation
        if self.word_based:
            parsed['pattern_word_ids'] = [self.word_to_id.get(w, self.unknown_id) for w in data['pattern']]
            if len(parsed['pattern_word_ids']) == 0:
                raise ValueError("len(parsed['pattern_word_ids']) == 0")
        if self.char_based:
            parsed['pattern_char_ids'] = [[self.char_to_id[c] for c in w if self.char_to_id]
                                for w in data['pattern']]
            # pattern_char_ids, word_lengths = self.pad_chars(pattern_char_ids)
            # parsed['pattern_word_lengths'] = word_lengths

        relation_ids = [self.sub_relation_to_id[r] for r in data['relation'].split('.')[-3:] if
                        r in self.sub_relation_to_id]

        if len(relation_ids) != 3:
            return None

        parsed['relation_ids'] = relation_ids

        # Extra feature
        extra_features = list()
        extra_features.append(float(data['constraint_entity_in_q']))
        extra_features.append(float(data['constraint_entity_word']))
        extra_features.append(float(data['entity_score']))
        extra_features.append(float(data['rel_pat_overlap']))
        extra_features.append(float(data['num_answer']))
        parsed['extra'] = extra_features
        parsed['f1'] = data['f1']
        return parsed

    def train_batch_iterator(self, batch_size):
        index = 0
        qids = self.qid_to_rank.keys()
        num_batch = self.num // batch_size + int(self.num % batch_size > 0)
        for _ in xrange(num_batch):
            all_pos_pattern_word_ids = []
            all_pos_sentence_lengths = []
            all_pos_relation_ids = []
            all_pos_mention_char_ids = []
            all_pos_topic_char_ids = []
            all_pos_extra = []

            all_neg_pattern_word_ids = []
            all_neg_sentence_lengths = []
            all_neg_relation_ids = []
            all_neg_mention_char_ids = []
            all_neg_topic_char_ids = []
            all_neg_extra = []

            def wrap_pos(q):
                all_pos_sentence_lengths.append(len(q['pattern_word_ids']))
                all_pos_pattern_word_ids.append(self.pad_words(q['pattern_word_ids'], self.word_padding))
                all_pos_relation_ids.append(q['relation_ids'])
                all_pos_mention_char_ids.append(self.pad_words(q['mention_char_ids'], self.char_padding))
                all_pos_topic_char_ids.append(self.pad_words(q['topic_char_ids'], self.char_padding))
                all_pos_extra.append(q['extra'])

            def wrap_neg(q):
                all_neg_sentence_lengths.append(len(q['pattern_word_ids']))
                all_neg_pattern_word_ids.append(self.pad_words(q['pattern_word_ids'], self.word_padding))
                all_neg_relation_ids.append(q['relation_ids'])
                all_neg_mention_char_ids.append(self.pad_words(q['mention_char_ids'], self.char_padding))
                all_neg_topic_char_ids.append(self.pad_words(q['topic_char_ids'], self.char_padding))
                all_neg_extra.append(q['extra'])

            while len(all_pos_relation_ids) < batch_size:
                if index == self.num:
                    random.shuffle(qids)
                    index = 0
                qid = qids[index]
                index += 1
                j = 0
                pairs = []
                while j < len(self.qid_to_rank) - 1:
                    h1 = self.qid_to_rank[qid][j]
                    h2 = self.qid_to_rank[qid][j+1]
                    if self.queries[(qid, h2)]['f1'] == 0.:
                        break
                    if self.queries[(qid, h1)]['f1'] > self.queries[(qid, h2)]['f1']:
                        pairs.append((h1, h2))
                    j += 1
                num_pos = min(3, j)
                pairs.extend(zip(self.qid_to_rank[:num_pos], random.sample(self.qid_to_rank[qid][j:], num_pos)))
                for h1, h2 in random.sample(pairs, min(len(pairs),10)):
                    wrap_pos(self.queries[(qid, h1)])
                    wrap_neg(self.queries[(qid, h2)])
            ret = {
                "pos_pattern_word_ids": all_pos_pattern_word_ids,
                "pos_sentence_lengths": all_pos_sentence_lengths,
                "pos_relation_ids": all_pos_relation_ids,
                "pos_mention_char_ids": all_pos_mention_char_ids,
                "pos_topic_char_ids": all_pos_topic_char_ids,
                "pos_extra": all_pos_extra,

                "neg_pattern_word_ids": all_neg_pattern_word_ids,
                "neg_sentence_lengths": all_neg_sentence_lengths,
                "neg_relation_ids": all_neg_relation_ids,
                "neg_mention_char_ids": all_neg_mention_char_ids,
                "neg_topic_char_ids": all_neg_topic_char_ids,
                "neg_extra": all_pos_extra,
            }
            for k, v in ret.items():
                ret[k] = np.array(v)
            yield ret

    def pad_chars(self, char_ids):
        char_ids = char_ids[:self.max_sentence_len]
        if len(char_ids) < self.max_sentence_len:
            char_ids += [[self.char_padding] * 1] * (self.max_sentence_len - len(char_ids))

        word_lengths = []
        for i, w in enumerate(char_ids):
            w = w[:self.max_word_len]
            padding = [self.char_padding] * (self.max_word_len - len(w))
            word_lengths.append(len(w))
            char_ids[i] = w + padding
        return char_ids, word_lengths

    def pad_words(self, seq, padding):
        seq = seq[:self.max_sentence_len]
        return seq + [padding] * (self.max_sentence_len - len(seq))