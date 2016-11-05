#coding=utf8
import json
from utils import split_sentence, normalize_word
import random

def replace_placeholder_one(pattern, sentence):
    pattern = pattern.split()
    sentence = sentence.split()
    j = 0
    new_pattern = []
    for i, w in enumerate(sentence):
        if j >= len(pattern):
            continue
        elif w == pattern[j]:
            j += 1
            new_pattern.append(w)
        elif pattern[j] == '_':
            new_pattern.append("<$>")
            j += 1
        elif pattern[j] == "_'s":
            new_pattern.append("<$>'s")
            j += 1
    return ' '.join(new_pattern)

def replace_placeholder(fn_in, fn_out):
    with open(fn_out, 'w') as fout:
        with open(fn_in) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                pattern, sentence = ll[-2:]
                pattern = pattern.split()
                ll[3] = replace_placeholder_one(pattern, sentence)
                line = '\t'.join(ll)
                print >> fout, line.encode('utf8')


def tag_sentence_iob(pattern, sentence):

    pattern, _ = split_sentence(pattern)
    sentence, _ = split_sentence(sentence)

    j = 0
    tagged = [['<START>', 'START']]
    for i, w in enumerate(sentence):
        if j >= len(pattern):
            tagged.append([w, 'I'])
        elif w == pattern[j]:
            j += 1
            tagged.append([w, 'O'])
        elif pattern[j] == '<$>' or pattern[j] == "<$>'s":
            tagged.append([w, 'B'])
            j += 1
        else:
            tagged.append([w, 'I'])
    tagged.append(['<END>', 'END'])
    return tagged


def iob_to_iobes(tagged):
    for i in range(len(tagged)):
        if tagged[i][-1] == 'B' and tagged[i+1][-1] != 'I':
            tagged[i][-1] = 'S'
        if tagged[i][-1] == 'I' and tagged[i+1][-1] != 'I':
            tagged[i][-1] = 'E'
    return tagged


def tag_sentence_iobes(pattern, sentence):
    return iob_to_iobes(tag_sentence_iob(pattern, sentence))

def gen_tagged_sentence(fn_list, fn_out, scheme):
    if scheme == "iob":
        fn = tag_sentence_iob
    else:
        fn = tag_sentence_iobes
    with open(fn_out, 'w') as fout:
        for fn_in in fn_list:
            with open(fn_in) as fin:
                for line in fin:
                    pattern, sentence = line.decode('utf8').strip().split('\t')[-2:]
                    tagged = fn(pattern, sentence)
                    tagged = '\t'.join([w + ' ' + t for w, t in tagged])
                    print >> fout, tagged.encode('utf8')


def gen_word_list(fn_in, fn_out):
    count = dict()
    with open(fn_in) as fin:
        for line in fin:
            # sentence = line.decode('utf8').strip().split('\t')[-1]
            sentence, normalized = split_sentence(line.decode('utf8').strip().split('\t')[-1])

            # for w in normalize_word(sentence).split():
            for w in normalized:
                if w not in count:
                    count[w] = 0
                count[w] += 1
    count = sorted(count.items(), key=lambda x:x[-1], reverse=True)
    with open(fn_out, 'w') as fout:
        for w, c in count:
            print >> fout, ('%s\t%s' % (w, c)).encode('utf8')

def gen_character_list(fn_in, fn_out):
    count = dict()
    with open(fn_in) as fin:
        for line in fin:
            sentence = line.decode('utf8').strip().split('\t')[-1]
            for w in sentence.split():
                if w.endswith("'s"):
                    w = w[:-2]
                if w.endswith(","):
                    w = w[:-1]
                for c in w:
                    if c not in count:
                        count[c] = 0
                    count[c] += 1
    count = sorted(count.items(), key=lambda x:x[-1], reverse=True)
    with open(fn_out, 'w') as fout:
        for ch, cnt in count:
            print >> fout, ("%s\t%s" % (ch, cnt)).encode('utf8')

def get_max_length(fn_in):
    max_len = 0
    with open(fn_in) as fin:
        for line in fin:
            sentence = line.decode('utf8').strip().split('\t')
            max_len = max(max_len, len(sentence))
    print max_len

def get_max_word_length(fn_in):
    max_len = 0
    with open(fn_in) as fin:
        for line in fin:
            sentence = line.decode('utf8').strip().split('\t')
            for w in sentence:
                w = w.split()[0]
                max_len = max(max_len, len(normalize_word(w)))
    print max_len

def merge_file(fn_list, fn_out):
    lines = []
    for fn in fn_list:
        with open(fn) as fin:
            for line in fin:
                lines.append(line.strip())

    random.shuffle(lines)
    with open(fn_out, 'w') as fout:
        for line in lines:
            print >> fout, line

def gen_simple_question_tag_data():
    fn_simple_test = "../data/simple.test.el.v2"
    fn_simple_train = "../data/simple.train.dev.el.v2"

    fn_simple_test_new = "../data/simple.test.el.v2.new"
    fn_simple_train_new = "../data/simple.train.dev.el.v2.new"

    fn_simple_test_iob = "../data/simple.test.el.v2.iob"
    fn_simple_test_iobes = "../data/simple.test.el.v2.iobes"
    fn_simple_train_iob = "../data/simple.train.el.v2.iob"
    fn_simple_train_iobes = "../data/simple.train.el.v2.iobes"

    replace_placeholder(fn_simple_test, fn_simple_test_new)
    replace_placeholder(fn_simple_train, fn_simple_train_new)

    gen_tagged_sentence(fn_simple_train_new, fn_simple_train_iob, "iob")
    gen_tagged_sentence(fn_simple_train_new, fn_simple_train_iobes, "iobs")
    gen_tagged_sentence(fn_simple_test_new, fn_simple_test_iob, "iob")
    gen_tagged_sentence(fn_simple_test_new, fn_simple_test_iobes, "iobs")
    fn_word = "../data/tag.word.list"
    fn_char = "../data/tag.char.list"

    gen_word_list(fn_simple_train_new, fn_word)
    gen_character_list(fn_simple_train_new, fn_char)

    get_max_length(fn_simple_test_iob)
    get_max_length(fn_simple_train_iob)

    get_max_word_length(fn_simple_test_iob)
    get_max_word_length(fn_simple_train_iob)

def transform_webquestion(fn_in, fn_out):
    with open(fn_out, 'w') as fout:
        wq = json.load(open(fn_in), encoding='utf8')
        for data in wq:
            pattern = data['sentence'].rstrip('?')
            utterance = data['utterance'].rstrip('?')
            pattern = replace_placeholder_one(pattern, utterance)
            print >> fout, '%s\t%s' % (pattern, utterance)

def gen_webquestion_tag_data():
    fn_wq_test = '../data/wq.test.complete.v2'
    fn_wq_train = '../data/wq.train.complete.v2'
    fn_wq_dev = '../data/wq.train.complete.v2'

    fn_wq_test_new = '../data/wq.test.complete.v2.new'
    fn_wq_train_new = '../data/wq.train.complete.v2.new'
    fn_wq_dev_new = '../data/wq.dev.complete.v2.new'

    fn_wq_test_iob = '../data/wq.test.complete.v2.iob'
    fn_wq_train_iob = '../data/wq.train.complete.v2.iob'

    fn_wq_test_iobes = '../data/wq.test.complete.v2.iobes'
    fn_wq_train_iobes = '../data/wq.train.complete.v2.iobes'

    transform_webquestion(fn_wq_test, fn_wq_test_new)
    transform_webquestion(fn_wq_train, fn_wq_train_new)
    transform_webquestion(fn_wq_dev, fn_wq_dev_new)

    gen_tagged_sentence([fn_wq_test_new], fn_wq_test_iob, 'iob')
    gen_tagged_sentence([fn_wq_train_new, fn_wq_dev_new], fn_wq_train_iob, 'iob')

    gen_tagged_sentence([fn_wq_test_new], fn_wq_test_iobes, 'iobes')
    gen_tagged_sentence([fn_wq_train_new, fn_wq_dev_new], fn_wq_train_iobes, 'iobes')




if __name__ == '__main__':
    # gen_simple_question_tag_data()
    # gen_webquestion_tag_data()

    fn_wq_train_iob = '../data/wq.train.complete.v2.iob'
    fn_wq_train_iobes = '../data/wq.train.complete.v2.iobes'


    fn_simple_train_iob = "../data/simple.train.el.v2.iob"
    fn_simple_train_iobes = "../data/simple.train.el.v2.iobes"

    fn_train_iob = '../data/tag.train.iob'
    fn_train_iobes = '../data/tag.train.iobes'
    merge_file([fn_wq_train_iob, fn_simple_train_iob], fn_train_iob)
    merge_file([fn_wq_train_iobes, fn_simple_train_iobes], fn_train_iobes)