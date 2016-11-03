#coding=utf8

def normalize_word(name):
    name = name.lower()
    name = name.replace('!', ' ')
    name = name.replace('.', ' ')
    name = name.replace(',', ' ')
    name = name.replace('-', ' ')
    name = name.replace('_', ' ')
    # name = name.replace(' ', '')
    name = name.replace(u"\’", ' ')
    name = name.replace('\'', ' ')
    name = name.replace('\"', ' ')
    name = name.replace('\\', ' ')
    name = name.replace('/', ' ')
    name = name.replace(u'–', ' ')
    name = name.replace(u'—', ' ')
    #add
    name = name.replace(':', ' ')
    name = name.replace('(', ' ')
    name = name.replace(')', ' ')
    return name

def replace_placeholder(fn_in, fn_out):
    with open(fn_out, 'w') as fout:
        with open(fn_in) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                pattern, sentence = ll[-2:]
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
                ll[3] = ' '.join(new_pattern)
                line = '\t'.join(ll)
                print >> fout, line.encode('utf8')


def tag_sentence_iob(pattern, sentence):

    pattern = [w for w in normalize_word(pattern).split()]
    sentence = [w for w in normalize_word(sentence).split()]
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

def gen_tagged_sentence(fn_in, fn_out, scheme):
    if scheme == "iob":
        fn = tag_sentence_iob
    else:
        fn = tag_sentence_iobes
    with open(fn_out, 'w') as fout:
        with open(fn_in) as fin:
            for line in fin:
                pattern, sentence = line.decode('utf8').strip().split('\t')[3:]
                tagged = fn(pattern, sentence)
                tagged = '\t'.join([w + ' ' + t for w, t in tagged])
                print >> fout, tagged.encode('utf8')

def gen_word_list(fn_in, fn_out):
    count = dict()
    with open(fn_in) as fin:
        for line in fin:
            sentence = line.decode('utf8').strip().split('\t')[-1]
            for w in normalize_word(sentence).split():
                # if w.endswith("'s"):
                #     w = w[:-2]
                # if w.endswith(","):
                #     w = w[:-1]
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
                max_len = max(max_len, len(w.split()[0]))
    print max_len

if __name__ == '__main__':
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
    gen_tagged_sentence(fn_simple_train_new, fn_simple_test_iobes, "iobs")
    fn_word = "../data/tag.word.list"
    fn_char = "../data/tag.char.list"

    gen_word_list(fn_simple_train_new, fn_word)
    gen_character_list(fn_simple_train_new, fn_char)

    get_max_length(fn_simple_test_iob)
    get_max_length(fn_simple_train_iob)


    get_max_word_length(fn_simple_test_iob)
    get_max_word_length(fn_simple_train_iob)