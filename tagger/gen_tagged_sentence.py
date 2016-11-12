#coding=utf8
import sys
sys.path.insert(0, '..')
import json
from tagger.utils import naive_split, normalize_word
import globals
from corenlp_parser.parser import CoreNLPParser
from corenlp_parser.local_parser import NLPParser
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
                    pattern = pattern.lower().split()
                    sentence = sentence.lower().split()

                    tagged = fn(pattern, sentence)
                    tagged = '\t'.join([w + ' ' + t for w, t in tagged])
                    print >> fout, tagged.encode('utf8')

def gen_tagged_sentence_plus_pos(fn_list, fn_out, scheme):
    if scheme == 'iob':
        fn = tag_sentence_iob
    else:
        fn = tag_sentence_iobes

    with open(fn_out, 'w') as fout:
        for fn_in in fn_list:
            with open(fn_in) as fin:
                for line in fin:
                    ll = line.decode('utf8').strip().split('\t')
                    pattern, _, pos, sentence = ll[-4:]
                    entity = ll[0]
                    pos = pos.split()
                    tagged = fn(pattern.lower().split(), sentence.lower().split())
                    tags = ' '.join([t[1] for t in tagged])
                    sentence = ' '.join(t[0] for t in tagged)
                    pos = ' '.join(pos)
                    print >> fout, '%s\t%s\t%s\t%s' % (entity, sentence, tags, pos)

def gen_word_list(fn_list, fn_out):
    count = dict()
    for fn_in in fn_list:
        with open(fn_in) as fin:
            for line in fin:
                sentence = line.decode('utf8').strip().split('\t')[-1].lower().split()
                for w in sentence:
                    if w not in count:
                        count[w] = 0
                    count[w] += 1
    count = sorted(count.items(), key=lambda x:x[-1], reverse=True)
    with open(fn_out, 'w') as fout:
        for w, c in count:
            print >> fout, ('%s\t%s' % (w, c)).encode('utf8')

def gen_character_list(fn_list, fn_out):
    count = dict()
    for fn_in in fn_list:
        with open(fn_in) as fin:
            for line in fin:
                sentence = line.decode('utf8').strip().lower().split('\t')[-1]
                for w in sentence.split():
                    for c in w:
                        if c not in count:
                            count[c] = 0
                        count[c] += 1
    count = sorted(count.items(), key=lambda x:x[-1], reverse=True)
    with open(fn_out, 'w') as fout:
        for ch, cnt in count:
            print >> fout, ("%s\t%s" % (ch, cnt)).encode('utf8')

def gen_pos_list(fn_list, fn_out):
    poss = set()
    for fn_in in fn_list:
        with open(fn_in) as fin:
            for line in fin:
                pos = line.decode('utf8').strip().split('\t')[-1].split()
                poss.update(pos)

    with open(fn_out, 'w') as fout:
        for p in poss:
            print >> fout, p.encode('utf8')

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
            print >> fout, '%s\t%s\t%s' % (data['mid1'], pattern, utterance)

def gen_webquestion_tag_data():
    fn_wq_test = '../data/wq.test.complete.v2'
    fn_wq_train = '../data/wq.train.complete.v2'
    fn_wq_dev = '../data/wq.dev.complete.v2'

    fn_wq_test_new = '../data/wq.test.complete.v2.new'
    fn_wq_train_new = '../data/wq.train.complete.v2.new'
    fn_wq_dev_new = '../data/wq.dev.complete.v2.new'

    fn_wq_test_iob = '../data/wq.test.complete.v2.iob'
    fn_wq_train_iob = '../data/wq.train.complete.v2.iob'

    fn_wq_test_iobes = '../data/wq.test.complete.v2.iobes'
    fn_wq_train_iobes = '../data/wq.train.complete.v2.iobes'

    # transform_webquestion(fn_wq_test, fn_wq_test_new)
    transform_webquestion(fn_wq_train, fn_wq_train_new)
    # transform_webquestion(fn_wq_dev, fn_wq_dev_new)

    # gen_tagged_sentence([fn_wq_test_new], fn_wq_test_iob, 'iob')
    # gen_tagged_sentence([fn_wq_train_new, fn_wq_dev_new], fn_wq_train_iob, 'iob')
    #
    # gen_tagged_sentence([fn_wq_test_new], fn_wq_test_iobes, 'iobes')
    # gen_tagged_sentence([fn_wq_train_new, fn_wq_dev_new], fn_wq_train_iobes, 'iobes')

def isCapital(word):
    for c in word.split():
        if c < 'A' or c > 'Z':
            return False
    return True

def merge_splited_word(sentence1, sentence2, poss):
    j = 0
    sentence2_ = []
    poss_ = []
    LR = {'-LSB-': '[', '-RSB-': ']', '-LCB-':'{', '-RCB-':'}', '-LRB-': '(', '-RRB-': ')'}
    for w, p in zip(sentence2, poss):
        if w in LR:
            sentence2_.append(LR[w])
            poss_.append('NN')
            continue
        ll = w.split()
        if len(ll) > 1:
            poss_.extend([p] * len(ll))
            sentence2_.extend(ll)
        else:
            poss_.append(p)
            sentence2_.append(w)

    n = len(sentence2_)
    new_poss = []

    try:
        for i, w in enumerate(sentence1):
            if sentence1[i] == sentence2_[j]:
                new_poss.append(poss_[j])
                j += 1
            elif sentence2_[j] == '':
                new_poss.append('NN')
                j += 1
            else:
                word = sentence2_[j]
                j += 1
                isCD = word.isdigit()

                while j < n and sentence1[i].startswith(word + sentence2_[j]):
                    if sentence2_[j] == '':
                        word = sentence1[i][:len(word) + 1]
                    else:
                        word += sentence2_[j]
                    isCD = isCD | sentence2_[j].isdigit()
                    j += 1
                if word == sentence1[i]:
                    new_poss.append('CD' if isCD else 'NN')
                else:
                    print sentence1
                    print sentence2_
                    return None, None
    except:
        print "error occurs in [merge_splited_word]", sentence1, sentence2
    return sentence1, new_poss

def add_pos_feature_remote(fn_in, fn_out, parser):
    with open(fn_out, 'w') as fout, open(fn_in) as fin:
        for line in fin:
            line = line.decode('utf8').strip().split('\t')
            pattern, sentence = line[-2:]
            pattern = naive_split(pattern)
            sentence = naive_split(sentence)
            line[-2] = ' '.join(pattern)
            tokens = parser.parse(' '.join(sentence))
            poss = [t.pos for t in tokens]
            tokens = [t.token for t in tokens]
            _, poss = merge_splited_word(sentence, tokens, poss)
            if poss == None:
                continue
            if len(sentence) != len(poss):
                print '|'.join(sentence)
                print '|'.join(tokens)

            poss = [u'<START>'] + poss + [u'<END>']
            # line = '\t'.join([w + ' ' + p for w, p in zip(line, poss)])
            line.append(' '.join(poss))
            line.append(' '.join(tokens))
            line = '\t'.join(line)
            print >> fout, line.encode('utf8')


def add_pos_feature_one(params):
    index, fn, start, end = params
    lno = 0
    ret = []
    with open(fn) as fin:
        while lno < start:
            fin.readline()
            lno += 1

        parser = NLPParser()
        for i in xrange(start, end):
            line = fin.readline()
            line = line.decode('utf8').strip().split('\t')
            pattern, sentence = line[-2:]
            pattern = naive_split(pattern)
            sentence = naive_split(sentence)
            line[-2] = ' '.join(pattern)
            tokens, poss = parser.tag_pos(' '.join(sentence))
            poss = [u'<START>'] + poss + [u'<END>']
            # line = '\t'.join([w + ' ' + p for w, p in zip(line, poss)])
            line.append(' '.join(poss))
            line.append(' '.join(tokens))
            line = ('\t'.join(line)).encode('utf8')
            ret.append(line)
    return ret

def add_pos_feature(fn_in, fn_out):
    from multiprocessing import Pool

    MAX_POOL_NUM = 4
    num_line = 0
    with open(fn_in) as fin:
        for _ in fin:
            num_line += 1
    print "There are %d lines to process." % num_line
    chunk_size = 100
    parameters = []
    i = 0
    while i * chunk_size < num_line:
        parameters.append((i, fn_in, i * chunk_size, min(num_line, (i + 1) * chunk_size)))
        i += 1

    pool = Pool(MAX_POOL_NUM)
    ret_list = pool.imap_unordered(add_pos_feature_one, parameters)
    pool.close()
    with open(fn_out, 'w') as fout:
        for l in ret_list:
            for s in l:
                print >> fout, s
    pool.join()


def merge_data():
    fn_wq_train_iob = '../data/wq.train.complete.v2.iob'
    fn_wq_train_iobes = '../data/wq.train.complete.v2.iobes'

    fn_simple_train_iob = "../data/simple.train.el.v2.iob"
    fn_simple_train_iobes = "../data/simple.train.el.v2.iobes"

    fn_train_iob = '../data/tag.train.iob'
    fn_train_iobes = '../data/tag.train.iobes'
    merge_file([fn_wq_train_iob, fn_simple_train_iob], fn_train_iob)
    merge_file([fn_wq_train_iobes, fn_simple_train_iobes], fn_train_iobes)

def gen_pos_data():
    globals.read_configuration('../config.cfg')

    fn_sq_train = '../data/simple.train.dev.el.v2.new'
    fn_sq_test = '../data/simple.test.el.v2.new'
    fn_sq_train_pos = '../data/simple.train.dev.el.v2.pos'
    fn_sq_test_pos = '../data/simple.test.el.v2.pos'

    fn_wq_train = '../data/wq.train.complete.v2.new'
    fn_wq_test = '../data/wq.test.complete.v2.new'
    fn_wq_dev = '../data/wq.dev.complete.v2.new'

    fn_wq_train_pos = '../data/wq.train.complete.v2.pos'
    fn_wq_test_pos = '../data/wq.test.complete.v2.pos'
    fn_wq_dev_pos = '../data/wq.dev.complete.v2.pos'

    fn_wq_train_pos_iob = '../data/wq.train.complete.v2.pos.iob'
    fn_wq_test_pos_iob = '../data/wq.test.complete.v2.pos.iob'

    fn_sq_train_pos_iob = '../data/simple.train.dev.el.v2.pos.iob'
    fn_sq_test_pos_iob = '../data/simple.test.el.v2.pos.iob'

    fn_train_pos_iob = '../data/tag.train.pos.iob'
    fn_word = "../data/tag.word.list"
    fn_char = '../data/tag.char.list'
    fn_pos = '../data/pos.list'
    # parser = CoreNLPParser.init_from_config()

    # add_pos_feature(fn_sq_train, fn_sq_train_pos+ '.tmp')
    # add_pos_feature(fn_sq_test, fn_sq_test_pos+'.tmp')

    add_pos_feature(fn_wq_train, fn_wq_train_pos +'.tmp')
    # add_pos_feature(fn_wq_dev, fn_wq_dev_pos + '.tmp')
    # add_pos_feature(fn_wq_test, fn_wq_test_pos + '.tmp')

    gen_tagged_sentence_plus_pos([fn_wq_train_pos +'.tmp', fn_wq_dev_pos +'.tmp'], fn_wq_train_pos_iob, 'iob')
    # gen_tagged_sentence_plus_pos([fn_wq_test_pos +'.tmp'], fn_wq_test_pos_iob, 'iob')

    # gen_tagged_sentence_plus_pos([fn_sq_train_pos +'.tmp'], fn_sq_train_pos_iob, 'iob')
    # gen_tagged_sentence_plus_pos([fn_sq_test_pos +'.tmp'], fn_sq_test_pos_iob, 'iob')

    merge_file([fn_wq_train_pos_iob, fn_sq_train_pos_iob], fn_train_pos_iob)
    # gen_word_list([fn_sq_train_pos, fn_wq_train_pos], fn_word)
    # gen_character_list([fn_sq_train_pos, fn_wq_train_pos], fn_char)
    # get_max_length(fn_train_pos_iob)
    # get_max_word_length(fn_train_pos_iob)
    # gen_pos_list([fn_train_pos_iob, fn_sq_test_pos_iob, fn_wq_test_pos_iob], fn_pos)

if __name__ == '__main__':
    # gen_simple_question_tag_data()
    gen_webquestion_tag_data()
    # merge_data()
    gen_pos_data()

