#coding=utf8
def normalize_word(name):
    if name == '<START>' or name == '<END>':
        return name
    name = name.lower()
    if name.endswith(u"’s") or name.endswith('\'s'):
        name = name[:-2]

    name = name.replace('!', '')
    name = name.replace('.', '')
    name = name.replace(',', '')

    name = name.replace(u"\’", '')
    name = name.replace('\'', '')
    name = name.replace('\"', '')
    name = name.replace('\\', '')
    # name = name.replace('/', '')

    #add
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

# def normalize_word(name):
#     name = name.lower()
#     name = name.replace('!', ' ')
#     name = name.replace('.', ' ')
#     name = name.replace(',', ' ')
#     name = name.replace('-', ' ')
#     name = name.replace('_', ' ')
#     # name = name.replace(' ', '')
#     name = name.replace(u"’", ' ')
#     name = name.replace('\'', ' ')
#     name = name.replace('"', ' ')
#     name = name.replace('\\', ' ')
#     name = name.replace('/', ' ')
#     name = name.replace(u'–', ' ')
#     name = name.replace(u'—', ' ')
#     #add
#     name = name.replace(':', ' ')
#     name = name.replace('(', ' ')
#     name = name.replace(')', ' ')
#     return name