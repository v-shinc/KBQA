#coding=utf8
import sys
def normalize_word(name):
    if name == '<START>' or name == '<END>':
        return name
    name = name.lower()
    if name.endswith("'s"):
        return name

    name = name.replace('!', '')
    name = name.replace('.', '')
    name = name.replace(',', '')
    name = name.replace('-', '')

    name = name.replace(u"’", '')
    name = name.replace("'", '')
    name = name.replace('"', '')
    name = name.replace('\\', '')
    name = name.replace('/', '')
    name = name.replace(':', '')
    name = name.replace('(', '')
    name = name.replace(')', '')
    return name

def split_sentence(text):
    text = text.replace(u'–', ' ')
    text = text.replace(u'—', ' ')
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    text = text.replace('/', ' ')
    text = text.replace('(', ' (')
    text = text.replace(')', ') ')
    text = text.replace(':', ': ')
    new_words = []
    normalized = []
    for w in text.split():
        nw = normalize_word(w)
        if len(nw) > 0:
            normalized.append(nw)
            new_words.append(w)
    return new_words, normalized

def naive_split(text):
    """ The placeholder of pattern should be <$> if text is a pattern """
    text = text.lower()
    # text = strip_accents(text)  # TODO: remove this line, and transform special punctuation to English punctuation
    to_blank = {'\\', '-', u'–', u'—', '-', '_', ':', '(', ')', '!', '?', ',', '.', '/', ';', '"', '...', u'“', u'”', u'…'}
    # to_blank = set(['\\', '-', '-', '_', ':', '(', ')', '!', '?', ',', '.', '/', ';', '"', '...'])
    sentence = ''
    l = len(text)
    i = 0

    while i < l:
        c = text[i]
        # if c == '!' or c == '?':
        #     continue
        if c == '#':
            sentence += ' # '
            i += 1
        elif i + 4 <= l and text[i:i+4] == "n't ":
            sentence += ' ' + text[i:i + 4]
            i += 4
        elif c == "'" or c == u"’":
            if i + 3 <= l and (text[i:i+3] == "'s " or text[i:i+3] == "'m " or text[i:i+3] == "'d "):
                sentence += ' '+text[i:i+3]
                i += 3
            elif i + 4 <= l and (text[i:i+4] == "'ll " or text[i:i+4] == "'re "):
                sentence += ' ' + text[i:i+4]
                i += 4
            else:
                sentence += " ' "
                i += 1

        elif c in to_blank:
            sentence += ' '
            i += 1
        else:
            sentence += c
            i += 1

    return sentence.split()


import unicodedata
def strip_accents(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    return s

def get_all_unichar():
    return ' '.join([unichr(i) for i in range(sys.maxunicode)])


def map_word(x, old_word='_', new_word='<$>'):
    if x.startswith(old_word) or x.endswith(old_word):
        return x.replace(old_word, new_word)
    else:
        return x


def replace_mention_holder(sentence, old_one='_', new_one="<$>"):
    return ' '.join([map_word(w, old_one, new_one) for w in sentence.split()])