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
        # self.relations = [r for r in relation_to_id.keys() if len(r.split('.')) == 3] # WHY
        self.relations = [r for r in relation_to_id.keys()]
        self.params = params
        self.use_rel_position = "use_position" in params['question_config'] and params['question_config']
        self.use_pat_position = "use_position" in params['relation_config'] and params['relation_config']
        self.pattern_dim = params['question_config']['word_dim']
        self.relation_dim = params['relation_config']['word_dim']
        self.relation_position = None
        if self.use_rel_position:
            self.relation_position = self.get_position(self.relation_dim, 3, 3)

    @property
    def num_word(self):
        return len(self.word_to_id) + 2 if self.word_based else 0

    @property
    def num_char(self):
        return len(self.char_to_id) + 1 if self.char_based else 0

    @property
    def num_relation(self):
        return len(self.sub_relation_to_id)

    def get_position(self, dim, length, max_sentence_len):
        vecs = [np.zeros(dim) for i in xrange(max_sentence_len)]
        for i in xrange(length):
            for j in xrange(dim):
                v = (i+1.) * dim / ((j+1.) * length)
                vecs[i][j] = min(v, 1. / v)
        return vecs

    def create_model_input(self, patterns, relations):
        all_word_ids = []
        all_char_ids = []
        all_word_lengths = []
        all_sentence_lengths = []
        all_relations = []
        all_relation_ids = []
        all_pattern_positions = []
        all_relation_positions = []
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
            if self.use_pat_position:
                all_pattern_positions.append(self.get_position(self.pattern_dim, len(pattern), self.max_sentence_len))

            if self.use_rel_position:
                all_relation_positions.append(self.relation_position)

            all_relations.append(relation)
            all_relation_ids.append([self.sub_relation_to_id[r] for r in relation.split('.')[-3:]])
        ret = {
            "word_ids": all_word_ids,
            "sentence_lengths": all_sentence_lengths,
            "char_ids": all_char_ids,
            "word_lengths": all_word_lengths,
            "relation_ids": all_relation_ids,
            "relations": all_relations,
            "relation_positions": all_relation_positions,
            "pattern_positions": all_pattern_positions
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
                all_pattern_positions = []
                all_relation_positions = []
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
                if self.use_pat_position:
                    all_pattern_positions = [self.get_position(self.pattern_dim, len(str_words), self.max_sentence_len)] * len(all_relation_ids)

                if self.use_rel_position:
                    all_relation_positions = [self.relation_position] * len(all_relation_ids)

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
                    "relations": all_relations,
                    "relation_positions": all_relation_positions,
                    "pattern_positions": all_pattern_positions
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
            all_relation_postition = []
            all_pattern_position = []
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
                if self.use_pat_position:
                    all_pattern_position.append(self.get_position(self.pattern_dim, len(str_words), self.max_sentence_len))

                if self.use_rel_position:
                    all_relation_postition.append(self.relation_position)

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
                "neg_relation_ids": all_neg_relation_ids,
                "relation_positions": all_relation_postition,
                "pattern_positions": all_pattern_position
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
            all_pattern_position = []
            all_relation_postition = []
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
                if self.use_pat_position:
                    all_pattern_position.append(self.get_position(self.pattern_dim, len(str_words), self.max_sentence_len))

                if self.use_rel_position:
                    all_relation_postition.append(self.relation_position)

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
                "neg_relation_ids": all_neg_relation_ids,
                "relation_positions": all_relation_postition,
                "pattern_positions": all_pattern_position
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
