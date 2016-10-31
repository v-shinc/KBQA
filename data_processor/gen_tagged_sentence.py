
def normalize_word(name):
    name = name.lower()
    name = name.replace('!', '')
    name = name.replace('.', '')
    name = name.replace(',', '')
    name = name.replace('-', '')
    name = name.replace('_', '')
    name = name.replace(' ', '')
    name = name.replace('\'', '')

    #add
    name = name.replace(':', '')
    name = name.replace('(', '')
    name = name.replace(')', '')
    return name

def replace_placeholder(fn_in, fn_out):


def tag_sentence_iob(pattern, sentence):

    pattern = [w[:-2] if w.endswith("'s") else w for w in pattern.split()]
    sentence = [w[:-2] if w.endswith("'s") else w for w in sentence.split()]
    j = 0
    tagged = [['<START>', 'START']]
    for i, w in enumerate(sentence):
        if j >= len(pattern):
            tagged.append([w, 'I'])
        elif w == pattern[j]:
            j += 1
            tagged.append([w, 'O'])
        elif pattern[j] == '_' or pattern[j] == "_'s":
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
            for w in sentence.split():
                if w.endswith("'s"):
                    w = w[:-2]
                if w.endswith(","):
                    w = w[:-1]
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

if __name__ == '__main__':
    fn_simple_test = "../data/simple.test.el.v2"
    fn_simple_iob = "../data/simple.test.el.v2.iob"
    fn_simple_iobes = "../data/simple.test.el.v2.iobes"
    fn_simple_train = "../data/simple.train.dev.el.v2"
    # gen_tagged_sentence(fn_simple, fn_simple_iob, "iob")
    # gen_tagged_sentence(fn_simple, fn_simple_iobes, "iobs")

    fn_word = "../data/tag.word.list"
    fn_char = "../data/tag.char.list"

    gen_word_list(fn_simple_train, fn_word)
    gen_character_list(fn_simple_train, fn_char)
