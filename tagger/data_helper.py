import numpy as np

# data format:  word1[space]tag1[tab]word2[space]tag2[tab]...

def load_mapping(fn_word):
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

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

tag_scheme = {}
tag_scheme['iob'] = ['I', 'O', 'B', 'START', 'END']
tag_scheme['iobes'] = ['I', 'O', 'B', 'E', 'S', 'START', 'END']

def load_tag_mapping(tag_scheme_name):
    tag_to_id = dict(zip(tag_scheme[tag_scheme_name], range(len(tag_scheme[tag_scheme_name]))))
    id_to_tag = dict(zip(range(len(tag_scheme[tag_scheme_name])), tag_scheme[tag_scheme_name]))
    return tag_to_id, id_to_tag

class DataSet(object):
    def __init__(self, fn_word, fn_char, params):

        self.word_to_id, self.id_to_word, self.word_to_count = load_mapping(fn_word)
        if '<UNK>' not in self.word_to_id:
            unknow_idx = len(self.word_to_id)
            self.word_to_id['<UNK>'] = unknow_idx
            self.id_to_word[unknow_idx] = '<UNK>'
            self.word_to_count['<UNK>'] = 1000

        if '<START>' not in self.word_to_id:
            start_idx = len(self.word_to_id)
            self.word_to_id['<START>'] = start_idx
            self.id_to_word[start_idx] = '<START>'
            self.word_to_count['<START>'] = 1000
        if '<END>' not in self.word_to_id:
            end_idx = len(self.word_to_id)
            self.word_to_id['<END>'] = end_idx
            self.id_to_word[end_idx] = '<END>'
            self.word_to_count['<END>'] = 1000
        if 'char_dim' in params:
            self.char_to_id, self.id_to_char, self.char_to_count = load_mapping(fn_char)
        self.tag_scheme_name = params['tag_scheme']
        self.tag_to_id, self.id_to_tag = load_tag_mapping(self.tag_scheme_name)
        self.char_padding = len(self.char_to_id)
        self.word_padding = len(self.word_to_id)
        self.tag_padding = self.tag_to_id['END']
        self.max_word_len = params['max_word_len']
        self.max_sentence_len = params['max_sentence_len']
        self.params = params

    @property
    def num_word(self):
        return len(self.word_to_id)
    @property
    def num_char(self):
        return len(self.char_to_id) if self.char_to_id else 0
    @property
    def num_cap(self):
        return 4 if 'cap_dim' in self.params else 0
    @property
    def num_tag(self):
        return len(self.tag_to_id)

    def batch_iterator(self, fn_train, batch_size):
        num = 0
        with open(fn_train) as fin:
            for _ in fin:
                num += 1
        train_file = open(fn_train)
        index = 0
        num_batch = num // batch_size + int(num % batch_size > 0)
        for _ in xrange(num_batch):
            all_word_ids = []
            all_char_for_ids = []
            all_char_rev_ids = []
            all_word_lengths = []
            all_sentence_lengths = []
            all_cap_ids = []
            all_tags_ids = []
            while len(all_tags_ids) < batch_size:
                if index == num:
                    train_file.seek(0)
                    index = 0
                line = train_file.readline()
                index += 1

                line = line.decode('utf8').strip().split('\t')
                sentence = [w.split() for w in line[:self.max_sentence_len]]
                tags = [w[1] for w in sentence]
                str_words = [w[0] for w in sentence]

                if self.params['word_dim']:
                    word_ids = [self.word_to_id[w if (w in self.word_to_id and self.word_to_count[w] > 1) else '<UNK>']
                         for w in str_words]
                    if len(word_ids) == 0:
                        raise ValueError('len(word_ids) == 0')
                        # continue
                    all_word_ids.append(self.pad_word(word_ids))

                all_sentence_lengths.append(len(str_words))
                # Skip characters that are not in the training set
                if self.params['char_dim']:
                    char_ids = [[self.char_to_id[c] for c in w if c in self.char_to_id]
                             for w in str_words]

                    char_for_ids, char_rev_ids, word_lengths = self.pad_chars(char_ids)
                    all_char_for_ids.append(char_for_ids)
                    all_char_rev_ids.append(char_rev_ids)
                    all_word_lengths.append(word_lengths)

                if self.params['cap_dim']:
                    cap_ids = [cap_feature(w) for w in str_words]
                    all_cap_ids.append(self.pad_cap(cap_ids))

                tag_ids = self.pad_tag([self.tag_to_id[t] for t in tags])

                all_tags_ids.append(tag_ids)

            ret = {
                "word_ids": all_word_ids,
                "sentence_lengths": all_sentence_lengths,
                "char_for_ids": all_char_for_ids,
                "char_rev_ids": all_char_rev_ids,
                "word_lengths": all_word_lengths,
                "cap_ids": all_cap_ids,
                "tag_ids": all_tags_ids
            }
            for k, v in ret.items():
                ret[k] = np.array(v)
            yield ret
        train_file.close()


    def get_named_entity(self, sentence, tag_sequence):
        entities = []
        entity = []
        sentence = [self.id_to_word[i] for i in sentence]

        tag_sequence = [self.id_to_tag[t] for t in tag_sequence]
        if self.tag_scheme_name == "iobes":
            for w, t in zip(sentence, tag_sequence):
                if t == 'B' or t == 'O' or t == 'END' or t == 'S':
                    if len(entity) > 0:
                        entities.append(' '.join(entity))
                        entity = []
                if t == 'B' or t == 'I' or t == 'E' or t == 'S':
                    entity.append(w)
        else:
            for w, t in zip(sentence, tag_sequence):
                if t == 'B' or t == 'O' or t == 'END':
                    if len(entity) > 0:
                        entities.append(' '.join(entity))
                if t == 'B' or t == 'I':
                    entity.append(w)
        return entities, sentence, tag_sequence

    def iob_to_iobes(self):
        pass

    def iobes_to_iob(self):
        pass

    def pad_chars(self, char_ids):
        """
            Pad the characters of the words in a sentence.
            Input:
                - list of lists of ints (list of words, a word being a list of char indexes)
            Output:
                - padded list of lists of ints
                - padded list of lists of ints (where chars are reversed)
                - list of ints corresponding to the index of the last character of each word
        """

        char_ids = char_ids[:self.max_sentence_len]

        if len(char_ids) < self.max_sentence_len:
            char_ids += [[self.char_padding] * 1] * (self.max_sentence_len - len(char_ids))

        char_for = []
        char_rev = []
        word_lengths = []
        for w in char_ids:
            w = w[:self.max_word_len]
            padding = [self.char_padding] * (self.max_word_len - len(w))
            char_for.append(w + padding)
            char_rev.append(w[::-1] + padding)
            word_lengths.append(len(w))
        return char_for, char_rev, word_lengths

    def pad_cap(self, caps):
        return caps + [0] * (self.max_sentence_len - len(caps))
    def pad_word(self, words):
        """
        Pad the words of the sentence
        :param words: list of ints (list of word index)
        :return: padded sentence
        """
        words = words[:self.max_sentence_len]
        return words + [self.word_padding] * (self.max_sentence_len - len(words))

    def pad_tag(self, tags):
        tags = tags[:self.max_sentence_len]
        return tags + [self.tag_padding] * (self.max_sentence_len - len(tags))


